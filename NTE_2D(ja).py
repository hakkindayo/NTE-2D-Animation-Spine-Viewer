import base64
import http.server
import io
import json
import shutil
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

# ---------- 0. 設定(json保存、フォルダ自動生成) ----------

CONFIG_PATH = Path(__file__).resolve().parent / "nte_config.json"

# 設定ファイルが無い場合の初期値(空なら初回入力時に必須入力になる)
SOURCE_ROOT = Path("")
OUTPUT_ROOT = Path("")


def load_or_create_config():
    """nte_config.json(スクリプトと同じフォルダ)から読み込み元/保存先を読み込む。
    無ければ初回実行として、コンソールでパスを直接入力してもらい新規作成する。
    保存先フォルダが存在しなければ自動生成する。"""
    global SOURCE_ROOT, OUTPUT_ROOT

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            SOURCE_ROOT = Path(cfg["source_root"])
            OUTPUT_ROOT = Path(cfg["output_root"])
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[警告] {CONFIG_PATH} の読み込みに失敗、初回設定をやり直す: {e}")

    print(f"初回実行のようです。設定ファイル({CONFIG_PATH.name})が無いのでパスを入力してください。")
    print("(何も入力せずEnterのみなら [ ] 内のデフォルト値を使います。デフォルトが空の項目は入力必須です)")

    while True:
        src_in = input(f"FModel書き出し元フォルダのパス [{SOURCE_ROOT}]: ").strip().strip('"')
        if src_in:
            SOURCE_ROOT = Path(src_in)
            break
        if str(SOURCE_ROOT):
            break
        print("  -> 空にはできません。パスを入力してください。")

    while True:
        out_in = input(f"保存先フォルダのパス [{OUTPUT_ROOT}]: ").strip().strip('"')
        if out_in:
            OUTPUT_ROOT = Path(out_in)
            break
        if str(OUTPUT_ROOT):
            break
        print("  -> 空にはできません。パスを入力してください。")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_config()
    print(f"設定を {CONFIG_PATH} に保存した。次回からはこのファイルの値が自動で使われる。")


def save_config():
    """現在のSOURCE_ROOT/OUTPUT_ROOTを nte_config.json(スクリプトの隣)に保存する。
    保存先フォルダが無ければ自動生成する。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"source_root": str(SOURCE_ROOT), "output_root": str(OUTPUT_ROOT)},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


SPINE_WEBGL_VERSION = "4.2.39"  # NTEのskelは4.2.x形式(鳴潮は4.1.x)。patchでAPIが変わることがあるので実バージョンにピン留め
FPS = 30
CANVAS_DIM = 1200         # レンダリング解像度(正方形、px)
MARGIN = 1.15             # skeleton boundsの余白倍率
MAKE_VIDEO = True         # ffmpegがあればmp4/movも書き出す
MAKE_ALPHA_MOV = True     # 背景透過版(*_alpha.mov, PNGコーデック)も書き出す


def get_free_port() -> int:
    """localhostの空いてるポートをOSに割り当ててもらう(固定ポートで衝突しないように)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


HTTP_PORT = None  # main()内で自動的に空きポートが入る


# ---------- 1. 抽出処理 ----------

DONE_MANIFEST_NAME = "_done.json"  # OUTPUT_ROOT直下にまとめて置く完了マーカー(全アセット共通の1ファイル)


def done_manifest_path() -> Path:
    return OUTPUT_ROOT / DONE_MANIFEST_NAME


def load_done_manifest() -> dict:
    p = done_manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_done_manifest(manifest: dict):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    done_manifest_path().write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def try_extract_spine_pair(atlas_json_path: Path):
    """atlas_json_path: SpineAtlasAssetのjson (例: SP_YLYYZ1.json)。
    同じフォルダの <stem>_Data.json (SpineSkeletonDataAsset) と対にして読む。
    戻り値: (atlas_text, skel_bytes) または None
    """
    try:
        with open(atlas_json_path, "r", encoding="utf-8") as f:
            atlas_data = json.load(f)
        if not atlas_data or atlas_data[0].get("Type") != "SpineAtlasAsset":
            return None
        atlas_raw = atlas_data[0]["Properties"]["RawData"]
        if not isinstance(atlas_raw, str):
            return None

        data_json_path = atlas_json_path.parent / f"{atlas_json_path.stem}_Data.json"
        if not data_json_path.exists():
            return None
        with open(data_json_path, "r", encoding="utf-8") as f:
            skel_data = json.load(f)
        if not skel_data or skel_data[0].get("Type") != "SpineSkeletonDataAsset":
            return None
        skel_raw = skel_data[0]["Properties"]["RawData"]
        if not isinstance(skel_raw, list):
            return None

        return atlas_raw.replace("\\n", "\n"), bytes(bytearray(skel_raw))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, UnicodeDecodeError, OSError):
        return None


def atlas_page_images(atlas_text: str):
    lines = atlas_text.splitlines()
    images = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or ":" in s:
            continue
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_is_size = (i + 1 < len(lines)) and lines[i + 1].strip().startswith("size:")
        if prev_blank and next_is_size:
            images.append(s)
    return images


def find_png(char_dir: Path, image_name: str):
    stem = Path(image_name).stem
    for tex_dir_name in ("Textures", "textures"):
        tex_dir = char_dir / tex_dir_name
        if tex_dir.is_dir():
            hit = list(tex_dir.glob(f"{stem}.png"))
            if hit:
                return hit[0]
    hit = list(char_dir.rglob(f"{stem}.png"))
    return hit[0] if hit else None


def stage_textures(atlas_text: str, images: list, char_dir: Path, stem: str, out_dir: Path):
    """テクスチャpngをout_dirにコピーし、atlas内のページ名も書き換える。

    同じ出力フォルダ(out_dir)に複数のアセット(stem違い)が混在する場合、
    テクスチャのファイル名が同じでも中身が別物というケースがあり得る。
    そのまま同名コピーすると "if not dest.exists(): skip" で先に処理した方の
    画像を後から処理したアセットが誤って使ってしまう(画像の順番/取り違えバグ)。
    これを避けるため、コピー先のファイル名に必ず stem を付けて一意にし、
    atlas_text 内のページ名の行もその新しい名前に書き換える。

    戻り値: (書き換え後のatlas_text, コピーしたPathのリスト, 見つからなかった元画像名のリスト)
    """
    lines = atlas_text.splitlines()
    copied = []
    missing = []
    for img in images:
        src = find_png(char_dir, img)
        if src is None:
            missing.append(img)
            continue
        new_name = f"{stem}__{src.name}"
        dest = out_dir / new_name
        if not dest.exists():
            shutil.copy2(src, dest)
        copied.append(dest)
        for i, line in enumerate(lines):
            if line.strip() == img:
                lines[i] = new_name
    return "\n".join(lines), copied, missing


def extract_all(done_manifest: dict):
    """戻り値: [(出力フォルダ名, skelファイル名, atlasファイル名, マニフェストキー), ...]"""
    entries = []
    if not SOURCE_ROOT.exists():
        print(f"見つからない: {SOURCE_ROOT}")
        return entries

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for json_path in SOURCE_ROOT.rglob("*.json"):
        # _Data.json は skel側なので、atlas側(本体)からペアで処理する
        if json_path.stem.endswith("_Data"):
            continue
        # Textures配下のテクスチャ用jsonはスキップ(Type判定でも弾かれるが念のため)
        if json_path.parent.name.lower() == "textures":
            continue

        result = try_extract_spine_pair(json_path)
        if result is None:
            continue
        atlas_text, skel_bytes = result
        char_dir = json_path.parent
        stem = json_path.stem
        manifest_key = f"{char_dir.name}__{stem}"

        if manifest_key in done_manifest:
            print(f"[スキップ] {manifest_key} : 完了マーカーあり(処理済み)")
            continue

        images = atlas_page_images(atlas_text)
        out_dir = OUTPUT_ROOT / char_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        new_atlas_text, pngs, missing = stage_textures(atlas_text, images, char_dir, stem, out_dir)

        if not pngs:
            print(f"[スキップ] {json_path.relative_to(SOURCE_ROOT)} : テクスチャが見つからない ({', '.join(images)})")
            continue
        if missing:
            print(f"[注意] {json_path.relative_to(SOURCE_ROOT)} : 見つからないテクスチャ {missing}")

        skel_name, atlas_name = f"{stem}.skel", f"{stem}.atlas"
        (out_dir / skel_name).write_bytes(skel_bytes)
        (out_dir / atlas_name).write_text(new_atlas_text, encoding="utf-8")

        entries.append((char_dir.name, skel_name, atlas_name, manifest_key))
        print(f"[抽出OK] {manifest_key} : png x{len(pngs)}")

    return entries


# ---------- 2. レンダリング用HTMLハーネス(内部用、ユーザーは開かない) ----------

HARNESS_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://unpkg.com/@esotericsoftware/spine-webgl@__SPINE_VERSION__/dist/iife/spine-webgl.js"></script>
</head><body style="margin:0">
<canvas id="canvas" width="__DIM__" height="__DIM__"></canvas>
<script>
window.ready = false;
window.loadError = null;
window.animNames = [];

window.addEventListener('error', (e) => {
  window.loadError = 'window.onerror: ' + e.message;
  console.error('CAUGHT', e.message, e.filename, e.lineno);
});

if (typeof spine === 'undefined') {
  window.loadError = 'spine-webgl script did not load (CDN blocked/offline?)';
}

const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl", {alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: true});
if (!gl) {
  window.loadError = 'WebGL context could not be created';
}
let renderer, assetManager;
if (gl && typeof spine !== 'undefined') {
  renderer = new spine.SceneRenderer(canvas, gl, true);
  assetManager = new spine.AssetManager(gl, "");
  assetManager.loadBinary("__SKEL__");
  assetManager.loadTextureAtlas("__ATLAS__");
}

let skeleton, animations = {};

function poll() {
  if (window.loadError) return;
  if (!assetManager) { setTimeout(poll, 30); return; }
  if (assetManager.hasErrors && assetManager.hasErrors()) {
    window.loadError = 'asset load error: ' + JSON.stringify(assetManager.getErrors());
    return;
  }
  if (assetManager.isLoadingComplete()) {
    const atlas = assetManager.get("__ATLAS__");
    const atlasLoader = new spine.AtlasAttachmentLoader(atlas);
    const skeletonBinary = new spine.SkeletonBinary(atlasLoader);
    const skeletonData = skeletonBinary.readSkeletonData(assetManager.get("__SKEL__"));
    skeleton = new spine.Skeleton(skeletonData);
    skeletonData.animations.forEach(a => { animations[a.name] = a; window.animNames.push(a.name); });

    skeleton.setToSetupPose();
    skeleton.updateWorldTransform(spine.Physics.update);

    // デバッグ用: 描画順(スロット順)・アタッチメント・ブレンドモードを一覧出力する。
    // 変な白い板などが出た時、どのスロットが原因か特定するのに使う。
    console.log('SLOTS(draw order): ' + skeleton.slots.map((s, i) => {
      const att = s.getAttachment();
      return `#${i}:${s.data.name}=[${att ? att.name : 'null'}]/blend=${s.data.blendMode}`;
    }).join(' | '));

    window.ready = true;
  } else {
    setTimeout(poll, 30);
  }
}
if (gl && typeof spine !== 'undefined') poll();

// ゲーム内で動的に差し込まれる「カウンター/データ表示」などのプレースホルダースロットは、
// 静止画として書き出すと中身が空(白い板など)のまま目立って表示されてしまうことがある。
// 名前にこれらのキーワードを含むスロットは、アニメーション適用のたびに強制的に非表示にする。
const HIDE_SLOT_KEYWORDS = ['数字', '数据', 'shuju'];
function hideDebugSlots() {
  skeleton.slots.forEach(s => {
    if (HIDE_SLOT_KEYWORDS.some(kw => s.data.name.includes(kw))) {
      s.setAttachment(null);
    }
  });
}

window.renderFrame = function(animName, time, bgR, bgG, bgB) {
  // bgR/bgG/bgBは0〜1の背景色。加算/スクリーン合成のVFXは透明キャンバス上だと
  // 正しく合成されない(色や不透明度がおかしくなる)ため、常に不透明な背景で描画する。
  // 透過が必要な場合は、黒背景版と白背景版を2回描画して差分からアルファを計算する
  // (Python側の compose_alpha を参照)。
  const anim = animations[animName];
  skeleton.setToSetupPose();
  anim.apply(skeleton, 0, time, false, null, 1, spine.MixBlend.setup, spine.MixDirection.mixIn);
  hideDebugSlots();
  skeleton.updateWorldTransform(spine.Physics.update);
  gl.clearColor(bgR, bgG, bgB, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  renderer.begin();
  renderer.drawSkeleton(skeleton, false);
  renderer.end();
  return canvas.toDataURL("image/png");
};

window.getAnimDuration = function(animName) {
  return animations[animName].duration;
};

// アニメーション全体(複数フレーム)をサンプリングしてbounds範囲の合計を求め、
// そのアニメーション用にカメラを合わせ直す。セットアップポーズだけを見て
// カメラを固定すると、動きの大きいVFXなどでは実際の描画位置とズレてしまうため。
window.fitCameraToAnim = function(animName, samples) {
  const anim = animations[animName];
  const n = Math.max(1, samples || 40);
  const offset = new spine.Vector2(), size = new spine.Vector2();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i <= n; i++) {
    const t = (anim.duration * i) / n;
    skeleton.setToSetupPose();
    anim.apply(skeleton, 0, t, false, null, 1, spine.MixBlend.setup, spine.MixDirection.mixIn);
    hideDebugSlots();
    skeleton.updateWorldTransform(spine.Physics.update);
    skeleton.getBounds(offset, size, []);
    minX = Math.min(minX, offset.x);
    minY = Math.min(minY, offset.y);
    maxX = Math.max(maxX, offset.x + size.x);
    maxY = Math.max(maxY, offset.y + size.y);
  }
  const width = maxX - minX, height = maxY - minY;
  renderer.camera.position.set(minX + width / 2, minY + height / 2, 0);
  const squareDim = Math.max(width, height) * __MARGIN__;
  renderer.camera.viewportWidth = squareDim;
  renderer.camera.viewportHeight = squareDim;
  renderer.camera.update();
};
</script>
</body></html>
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass


def serve_output_root(port: int):
    def handler(*a, **kw):
        return QuietHandler(*a, directory=str(OUTPUT_ROOT), **kw)
    httpd = socketserver.TCPServer(("localhost", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def decode_frame_rgb(data_url: str) -> np.ndarray:
    """canvas.toDataURL()のPNG data URLをデコードしてRGBのfloat32配列にする"""
    png_bytes = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float32)


def compose_alpha_from_black_white(black_arr: np.ndarray, white_arr: np.ndarray) -> Image.Image:
    """黒背景/白背景の2枚のレンダリング結果から、本来のアルファ値と色を復元する(差分マット)。

    スクリーン/加算合成などのVFXは透明キャンバス上に直接描くと正しく合成されない
    (色や不透明度が崩れる)ため、常に不透明な背景で2回描画し、その差分から逆算する。
    「over」合成を仮定: white - black = (1-alpha) となる(背景色の寄与分の差)ので、
    alpha = 1 - (white - black)、straight_color = black / alpha で復元する。

    ただしこの割り算は、アルファがごく低い(ほぼ見えない)領域だとノイズに弱く、
    本来ほぼ透明で見えないはずの部分が色だけ異常に増幅されて、変な色のシミ/滲みが
    浮き出てしまうことがある。それを防ぐため:
      - 色の増幅率に上限をかける(ALPHA_FLOORより低いアルファは色をそれ以上増幅しない)
      - さらに低い(ALPHA_CUTOFF未満の)アルファは完全に透明として扱う
    """
    ALPHA_FLOOR = 48.0   # これ未満のアルファでは色の増幅率をこの値で頭打ちにする
    ALPHA_CUTOFF = 40.0  # これ未満のアルファはノイズとみなして完全透明(alpha=0)にする

    diff = white_arr - black_arr
    alpha = 255.0 - np.clip(diff.max(axis=2), 0.0, 255.0)
    alpha_safe = np.maximum(alpha, ALPHA_FLOOR)
    color = np.clip(black_arr * 255.0 / alpha_safe[..., None], 0.0, 255.0)
    alpha = np.where(alpha < ALPHA_CUTOFF, 0.0, alpha)
    rgba = np.dstack([color, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cleanup_keep_only_deliverables(out_dir: Path):
    """out_dir配下から *.mp4 / *_alpha.mov 以外を全部削除する
    (frames連番png、抽出した.skel/.atlas/テクスチャpngなど)。"""
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.suffix.lower() != ".mp4" and not item.name.endswith("_alpha.mov"):
            item.unlink(missing_ok=True)


def encode_outputs(flat_pattern: str, rgba_pattern: str, out_dir: Path, name: str):
    """flat_pattern(黒背景に正しく合成済みのRGB連番png)から通常再生用の<name>.mp4を、
    rgba_pattern(差分マットで復元したRGBA連番png)から背景透過用の<name>_alpha.mov
    (PNGコーデック+アルファ, MAKE_ALPHA_MOV有効時のみ)を書き出す。

    透過は当初VP9(webm)で試したが、VLC/Discord/YMM4など多くのプレイヤーで
    アルファチャンネルが正しくデコードされないことが確認されたため、より確実な
    PNGコーデック入りの.mov(可逆・透過対応)に変更した。ファイルサイズは大きくなるが
    再生側の対応状況に左右されにくい。
    mp4側は4:2:0のクロマ間引きで細かい反射・ハイライトの色がぼやけるのを避けるため
    yuv444p(色情報を間引かない)+高めの画質設定(CRF低め)にしてある。

    戻り値: (mp4_ok, mov_ok)。ファイルサイズとffmpegの終了コードで成功を判定する
    (単に存在するかだけだと、壊れた/再生できないファイルを成功扱いしてしまうため)。
    """
    mp4_path = out_dir / f"{name}.mp4"
    r1 = subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", flat_pattern,
        "-pix_fmt", "yuv444p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "16", "-preset", "slow",
        "-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
        str(mp4_path)
    ], check=False, capture_output=True)
    mp4_ok = mp4_path.exists() and mp4_path.stat().st_size > 1024 and r1.returncode == 0
    if not mp4_ok:
        mp4_path.unlink(missing_ok=True)
        print(f"    [ffmpeg mp4 エラー] {name} : {r1.stderr.decode(errors='ignore')[-800:]}")

    mov_ok = False
    if MAKE_ALPHA_MOV:
        mov_path = out_dir / f"{name}_alpha.mov"
        r2 = subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", rgba_pattern,
            "-c:v", "png",
            str(mov_path)
        ], check=False, capture_output=True)
        mov_ok = mov_path.exists() and mov_path.stat().st_size > 1024 and r2.returncode == 0
        if not mov_ok:
            mov_path.unlink(missing_ok=True)
            print(f"    [ffmpeg mov(透過)エラー] {name} : {r2.stderr.decode(errors='ignore')[-800:]}")

    return mp4_ok, mov_ok


def render_character(page, char_dir_name, skel_name, atlas_name):
    """戻り値: 全アニメーションのmp4化まで成功したらTrue"""
    out_dir = OUTPUT_ROOT / char_dir_name
    harness_path = out_dir / "_harness.html"
    html = (HARNESS_TEMPLATE
            .replace("__SPINE_VERSION__", SPINE_WEBGL_VERSION)
            .replace("__SKEL__", skel_name)
            .replace("__ATLAS__", atlas_name)
            .replace("__DIM__", str(CANVAS_DIM))
            .replace("__MARGIN__", str(MARGIN)))
    harness_path.write_text(html, encoding="utf-8")

    page.goto(f"http://localhost:{HTTP_PORT}/{char_dir_name}/_harness.html")
    try:
        page.wait_for_function(
            "() => window.ready === true || window.loadError !== null",
            timeout=30000,
        )
    except Exception:
        load_error = page.evaluate("window.loadError")
        raise RuntimeError(f"読み込みタイムアウト(30秒)。window.loadError={load_error!r}")

    load_error = page.evaluate("window.loadError")
    if load_error:
        raise RuntimeError(f"読み込み失敗: {load_error}")

    anim_names = page.evaluate("window.animNames")
    all_ok = True

    for anim in anim_names:
        duration = page.evaluate("(a) => window.getAnimDuration(a)", anim)
        if not duration or duration <= 0:
            continue
        # このアニメーション全体の動きに合わせてカメラを合わせ直す(位置ズレ対策)
        page.evaluate("([a, s]) => window.fitCameraToAnim(a, s)", [anim, 40])
        n_frames = max(1, int(duration * FPS))
        frame_dir = out_dir / "frames" / anim
        frame_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_frames):
            t = i / FPS
            black_url = page.evaluate("([a, t]) => window.renderFrame(a, t, 0, 0, 0)", [anim, t])
            white_url = page.evaluate("([a, t]) => window.renderFrame(a, t, 1, 1, 1)", [anim, t])
            black_arr = decode_frame_rgb(black_url)
            white_arr = decode_frame_rgb(white_url)
            rgba_img = compose_alpha_from_black_white(black_arr, white_arr)
            rgba_img.save(frame_dir / f"frame_{i:04d}.png")
            Image.fromarray(black_arr.astype(np.uint8), mode="RGB").save(frame_dir / f"flat_{i:04d}.png")

        print(f"[書き出しOK] {char_dir_name} / {anim} : {n_frames}フレーム -> {frame_dir}")

        if MAKE_VIDEO and shutil.which("ffmpeg"):
            mp4_ok, mov_ok = encode_outputs(
                str(frame_dir / "flat_%04d.png"), str(frame_dir / "frame_%04d.png"), out_dir, anim
            )
            if mp4_ok:
                print(f"    動画も書き出し: {out_dir / (anim + '.mp4')}"
                      + (f" / {out_dir / (anim + '_alpha.mov')}" if mov_ok else ""))
                shutil.rmtree(frame_dir, ignore_errors=True)
            else:
                print(f"    [エラー] {char_dir_name} / {anim} : mp4化に失敗、連番pngのまま残す -> {frame_dir}")
                all_ok = False
        else:
            all_ok = False

    harness_path.unlink(missing_ok=True)

    if all_ok:
        cleanup_keep_only_deliverables(out_dir)
        print(f"    [クリーンアップ] {char_dir_name} : mp4/mov以外(frames連番png・抽出skel/atlas/テクスチャ)を削除")
    else:
        print(f"    [注意] {char_dir_name} : 一部失敗のため中間ファイルは残す(次回また再挑戦される)")

    return all_ok


def main():
    load_or_create_config()
    done_manifest = load_done_manifest()

    entries = extract_all(done_manifest)
    if not entries:
        print("変換できるアセットが見つからなかった(または全部処理済み)")
        return

    global HTTP_PORT
    HTTP_PORT = get_free_port()
    httpd = serve_output_root(HTTP_PORT)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": CANVAS_DIM, "height": CANVAS_DIM})
            page.on("console", lambda msg: print(f"    [ブラウザconsole] {msg.type}: {msg.text}"))
            page.on("pageerror", lambda exc: print(f"    [ブラウザpageerror] {exc}"))
            for char_dir_name, skel_name, atlas_name, manifest_key in entries:
                try:
                    ok = render_character(page, char_dir_name, skel_name, atlas_name)
                    if ok:
                        done_manifest[manifest_key] = {
                            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        save_done_manifest(done_manifest)
                except Exception as e:
                    print(f"[エラー] {char_dir_name} : {e}")
            browser.close()
    finally:
        httpd.shutdown()

    print(f"\nDone. Output written under {OUTPUT_ROOT} "
          f"(<anim>.mp4 for normal playback, <anim>_alpha.mov for transparent background).")


if __name__ == "__main__":
    main()
