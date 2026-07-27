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

# ---------- 0. Settings (JSON saving, auto-folder generation) ----------

CONFIG_PATH = Path(__file__).resolve().parent / "nte_config.json"

# Initial values if the config file does not exist (if empty, required on first run)
SOURCE_ROOT = Path("")
OUTPUT_ROOT = Path("")


def load_or_create_config():
    """Loads source/output paths from nte_config.json (in the same folder as the script).
    If missing, treats it as the first run, prompts the user to input paths directly in the console, and creates a new file.
    Auto-creates the output directory if it doesn't exist."""
    global SOURCE_ROOT, OUTPUT_ROOT

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            SOURCE_ROOT = Path(cfg["source_root"])
            OUTPUT_ROOT = Path(cfg["output_root"])
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[WARNING] Failed to load {CONFIG_PATH}, restarting initial setup: {e}")

    print(f"First run detected. Config file ({CONFIG_PATH.name}) not found, please enter the paths.")
    print("(Pressing Enter without input uses default values inside [ ]. Items with empty defaults are required)")

    while True:
        src_in = input(f"FModel export source folder path [{SOURCE_ROOT}]: ").strip().strip('"')
        if src_in:
            SOURCE_ROOT = Path(src_in)
            break
        if str(SOURCE_ROOT):
            break
        print("  -> Cannot be empty. Please enter a path.")

    while True:
        out_in = input(f"Destination folder path [{OUTPUT_ROOT}]: ").strip().strip('"')
        if out_in:
            OUTPUT_ROOT = Path(out_in)
            break
        if str(OUTPUT_ROOT):
            break
        print("  -> Cannot be empty. Please enter a path.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_config()
    print(f"Settings saved to {CONFIG_PATH}. These values will be used automatically from next time.")


def save_config():
    """Saves current SOURCE_ROOT/OUTPUT_ROOT to nte_config.json (next to the script).
    Auto-creates the output directory if it doesn't exist."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"source_root": str(SOURCE_ROOT), "output_root": str(OUTPUT_ROOT)},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


SPINE_WEBGL_VERSION = "4.2.39"  # NTE skel uses 4.2.x format (Wuthering Waves uses 4.1.x). Pinned to actual version as patches can change API
FPS = 30
CANVAS_DIM = 1200         # Rendering resolution (square, px)
MARGIN = 1.15             # Skeleton bounds margin multiplier
MAKE_VIDEO = True         # Also export mp4/mov if ffmpeg is available
MAKE_ALPHA_MOV = True     # Also export transparent background version (*_alpha.mov, PNG codec)


def get_free_port() -> int:
    """Asks OS to assign an available port on localhost (to avoid conflicts with fixed ports)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


HTTP_PORT = None  # Free port is automatically assigned inside main()


# ---------- 1. Extraction Process ----------

DONE_MANIFEST_NAME = "_done.json"  # Completion marker placed directly under OUTPUT_ROOT (single file shared across all assets)


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
    """atlas_json_path: SpineAtlasAsset JSON (e.g., SP_YLYYZ1.json).
    Reads as a pair with <stem>_Data.json (SpineSkeletonDataAsset) in the same folder.
    Returns: (atlas_text, skel_bytes) or None
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
    """Copies texture PNGs to out_dir and rewrites page names within the atlas.

    If multiple assets (different stems) coexist in the same output folder (out_dir),
    texture filenames might be identical while their contents differ.
    If copied with the same name, "if not dest.exists(): skip" would cause later processed assets
    to mistakenly use the images processed earlier (image order/mix-up bug).
    To prevent this, the destination filename is always prefixed with the stem to make it unique,
    and the page name lines inside atlas_text are also rewritten to this new name.

    Returns: (rewritten atlas_text, list of copied Paths, list of missing original image names)
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
    """Returns: [(output folder name, skel filename, atlas filename, manifest key), ...]"""
    entries = []
    if not SOURCE_ROOT.exists():
        print(f"Not found: {SOURCE_ROOT}")
        return entries

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for json_path in SOURCE_ROOT.rglob("*.json"):
        # _Data.json belongs to skel side, processed as a pair from the atlas side (main)
        if json_path.stem.endswith("_Data"):
            continue
        # Skip texture JSONs under Textures (caught by Type check too, but just in case)
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
            print(f"[SKIP] {manifest_key} : Completion marker exists (already processed)")
            continue

        images = atlas_page_images(atlas_text)
        out_dir = OUTPUT_ROOT / char_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        new_atlas_text, pngs, missing = stage_textures(atlas_text, images, char_dir, stem, out_dir)

        if not pngs:
            print(f"[SKIP] {json_path.relative_to(SOURCE_ROOT)} : Textures not found ({', '.join(images)})")
            continue
        if missing:
            print(f"[NOTICE] {json_path.relative_to(SOURCE_ROOT)} : Missing textures {missing}")

        skel_name, atlas_name = f"{stem}.skel", f"{stem}.atlas"
        (out_dir / skel_name).write_bytes(skel_bytes)
        (out_dir / atlas_name).write_text(new_atlas_text, encoding="utf-8")

        entries.append((char_dir.name, skel_name, atlas_name, manifest_key))
        print(f"[EXTRACT OK] {manifest_key} : png x{len(pngs)}")

    return entries


# ---------- 2. Rendering HTML Harness (Internal, user does not open) ----------

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

    // Debug: Output draw order (slot order), attachments, and blend modes as a list.
    // Used to identify which slot causes issues like unexpected white boxes.
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

// Placeholder slots such as "counters/data displays" dynamically injected inside the game 
// may remain empty (white boxes, etc.) and stand out when exported as still images.
// Slots containing these keywords in their names are forcibly hidden whenever animations are applied.
const HIDE_SLOT_KEYWORDS = ['数字', '数据', 'shuju'];
function hideDebugSlots() {
  skeleton.slots.forEach(s => {
    if (HIDE_SLOT_KEYWORDS.some(kw => s.data.name.includes(kw))) {
      s.setAttachment(null);
    }
  });
}

window.renderFrame = function(animName, time, bgR, bgG, bgB) {
  // bgR/bgG/bgB are background colors from 0 to 1. Additive/screen blend VFX do not composite correctly
  // on a transparent canvas (colors or opacity become incorrect), so they are always rendered on an opaque background.
  // If transparency is needed, black-background and white-background versions are rendered twice, and alpha is calculated from the difference
  // (refer to compose_alpha on the Python side).
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

// Samples the entire animation (multiple frames) to find the total bounds range,
// and readjusts the camera for that animation. If the camera is fixed looking only at the setup pose,
// it will misalign with actual rendering positions for heavy motion VFX, etc.
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
    """Decodes PNG data URL from canvas.toDataURL() into an RGB float32 array"""
    png_bytes = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float32)


def compose_alpha_from_black_white(black_arr: np.ndarray, white_arr: np.ndarray) -> Image.Image:
    """Restores original alpha values and colors from two rendering results (black background / white background) via difference matting.

    VFX such as screen/additive blending do not composite properly when drawn directly on a transparent canvas
    (colors or opacity degrade), so they are rendered twice with opaque backgrounds and reverse-calculated from the difference.
    Assuming "over" compositing: white - black = (1-alpha) (difference in background color contribution),
    so alpha = 1 - (white - black), and straight_color = black / alpha.

    However, this division is vulnerable to noise in regions with very low alpha (almost invisible),
    causing areas that should be transparent to abnormally amplify colors and show strange color spots/bleeds.
    To prevent this:
      - Limits the maximum color amplification rate (alphas lower than ALPHA_FLOOR will not amplify color further)
      - Treats alphas lower than ALPHA_CUTOFF as completely transparent
    """
    ALPHA_FLOOR = 48.0   # Caps color amplification at this value for alphas below this
    ALPHA_CUTOFF = 40.0  # Treats alphas below this as noise and makes them fully transparent (alpha=0)

    diff = white_arr - black_arr
    alpha = 255.0 - np.clip(diff.max(axis=2), 0.0, 255.0)
    alpha_safe = np.maximum(alpha, ALPHA_FLOOR)
    color = np.clip(black_arr * 255.0 / alpha_safe[..., None], 0.0, 255.0)
    alpha = np.where(alpha < ALPHA_CUTOFF, 0.0, alpha)
    rgba = np.dstack([color, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cleanup_keep_only_deliverables(out_dir: Path):
    """Deletes everything under out_dir except *.mp4 / *_alpha.mov
    (frames sequence PNGs, extracted .skel/.atlas/texture PNGs, etc.)."""
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.suffix.lower() != ".mp4" and not item.name.endswith("_alpha.mov"):
            item.unlink(missing_ok=True)


def encode_outputs(flat_pattern: str, rgba_pattern: str, out_dir: Path, name: str):
    """Exports standard playback <name>.mp4 from flat_pattern (RGB sequence PNGs correctly composited on black background),
    and transparent background <name>_alpha.mov (PNG codec + alpha, only when MAKE_ALPHA_MOV is enabled) 
    from rgba_pattern (RGBA sequence PNGs restored via difference matting).

    Transparency was initially tested with VP9 (webm), but since many players like VLC/Discord/YMM4 
    failed to decode the alpha channel correctly, it was changed to a more reliable 
    .mov container with PNG codec (lossless, supports transparency). Although file size is larger,
    it is less dependent on player compatibility.
    The MP4 uses yuv444p (preserves color information without sub-sampling) + higher quality settings (lower CRF)
    to avoid blurring fine reflections and highlights due to 4:2:0 chroma sub-sampling.

    Returns: (mp4_ok, mov_ok). Success is judged by file size and ffmpeg exit code
    (checking only existence might treat corrupted/unplayable files as successful).
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
        print(f"    [ffmpeg mp4 error] {name} : {r1.stderr.decode(errors='ignore')[-800:]}")

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
            print(f"    [ffmpeg mov (transparent) error] {name} : {r2.stderr.decode(errors='ignore')[-800:]}")

    return mp4_ok, mov_ok


def render_character(page, char_dir_name, skel_name, atlas_name):
    """Returns: True if all animations were successfully exported to MP4"""
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
        raise RuntimeError(f"Load timeout (30s). window.loadError={load_error!r}")

    load_error = page.evaluate("window.loadError")
    if load_error:
        raise RuntimeError(f"Load failed: {load_error}")

    anim_names = page.evaluate("window.animNames")
    all_ok = True

    for anim in anim_names:
        duration = page.evaluate("(a) => window.getAnimDuration(a)", anim)
        if not duration or duration <= 0:
            continue
        # Readjust camera according to the motion of this entire animation (prevents positional misalignment)
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

        print(f"[EXPORT OK] {char_dir_name} / {anim} : {n_frames} frames -> {frame_dir}")

        if MAKE_VIDEO and shutil.which("ffmpeg"):
            mp4_ok, mov_ok = encode_outputs(
                str(frame_dir / "flat_%04d.png"), str(frame_dir / "frame_%04d.png"), out_dir, anim
            )
            if mp4_ok:
                print(f"    Video also exported: {out_dir / (anim + '.mp4')}"
                      + (f" / {out_dir / (anim + '_alpha.mov')}" if mov_ok else ""))
                shutil.rmtree(frame_dir, ignore_errors=True)
            else:
                print(f"    [ERROR] {char_dir_name} / {anim} : MP4 conversion failed, keeping sequence PNGs -> {frame_dir}")
                all_ok = False
        else:
            all_ok = False

    harness_path.unlink(missing_ok=True)

    if all_ok:
        cleanup_keep_only_deliverables(out_dir)
        print(f"    [CLEANUP] {char_dir_name} : Deleted non-mp4/mov files (frames sequence PNGs, extracted skel/atlas/textures)")
    else:
        print(f"    [NOTICE] {char_dir_name} : Intermediate files kept due to partial failure (will retry next time)")

    return all_ok


def main():
    load_or_create_config()
    done_manifest = load_done_manifest()

    entries = extract_all(done_manifest)
    if not entries:
        print("No convertible assets found (or all already processed)")
        return

    global HTTP_PORT
    HTTP_PORT = get_free_port()
    httpd = serve_output_root(HTTP_PORT)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": CANVAS_DIM, "height": CANVAS_DIM})
            page.on("console", lambda msg: print(f"    [Browser console] {msg.type}: {msg.text}"))
            page.on("pageerror", lambda exc: print(f"    [Browser pageerror] {exc}"))
            for char_dir_name, skel_name, atlas_name, manifest_key in entries:
                try:
                    ok = render_character(page, char_dir_name, skel_name, atlas_name)
                    if ok:
                        done_manifest[manifest_key] = {
                            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        save_done_manifest(done_manifest)
                except Exception as e:
                    print(f"[ERROR] {char_dir_name} : {e}")
            browser.close()
    finally:
        httpd.shutdown()

    print(f"\nDone. Output written under {OUTPUT_ROOT} "
          f"(<anim>.mp4 for normal playback, <anim>_alpha.mov for transparent background).")


if __name__ == "__main__":
    main()