import base64
import gc
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

# ---------- 0. Settings (JSON saving, automatic folder generation) ----------

CONFIG_PATH = Path(__file__).resolve().parent / "nte_config.json"

# Initial values if the configuration file does not exist (if empty, required on first input)
SOURCE_ROOT = Path("")
OUTPUT_ROOT = Path("")


def load_or_create_config():
    """Load the source/destination paths from nte_config.json (same folder as the script).
    If it does not exist, treat it as the first run, prompt the user to input the paths directly in the console, and create a new one.
    Automatically generate the destination folder if it does not exist."""
    global SOURCE_ROOT, OUTPUT_ROOT

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            SOURCE_ROOT = Path(cfg["source_root"])
            OUTPUT_ROOT = Path(cfg["output_root"])
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[Warning] Failed to load {CONFIG_PATH}, redoing initial setup: {e}")

    print(f"It looks like the first run. The configuration file ({CONFIG_PATH.name}) does not exist, so please enter the paths.")
    print("(If you press Enter without typing anything, the default value inside [ ] will be used. Items with empty defaults are required)")

    while True:
        src_in = input(f"Path to the FModel export source folder [{SOURCE_ROOT}]: ").strip().strip('"')
        if src_in:
            SOURCE_ROOT = Path(src_in)
            break
        if str(SOURCE_ROOT):
            break
        print("  -> Cannot be empty. Please enter a path.")

    while True:
        out_in = input(f"Path to the destination folder [{OUTPUT_ROOT}]: ").strip().strip('"')
        if out_in:
            OUTPUT_ROOT = Path(out_in)
            break
        if str(OUTPUT_ROOT):
            break
        print("  -> Cannot be empty. Please enter a path.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_config()
    print(f"Settings saved to {CONFIG_PATH}. From next time onwards, the values in this file will be used automatically.")


def save_config():
    """Save the current SOURCE_ROOT/OUTPUT_ROOT to nte_config.json (next to the script).
    Automatically generate the destination folder if it does not exist."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"source_root": str(SOURCE_ROOT), "output_root": str(OUTPUT_ROOT)},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


SPINE_WEBGL_VERSION = "4.2.39"  # NTE skel uses the 4.2.x format (Wuthering Waves uses 4.1.x). Pinned to the actual version because patches can change the API
FPS = 30
CANVAS_DIM = 2400         # Rendering resolution (square, px)
MARGIN = 1.15             # Margin multiplier for skeleton bounds
MAKE_VIDEO = True         # Export mp4/mov as well if ffmpeg is available
MAKE_ALPHA_MOV = True     # Also export background-transparent versions (*_alpha.mov, PNG codec)
SHOW_BROWSER_LOGS = False  # If True, displays all browser-side console.log outputs (such as slot lists).
                            # Warnings/errors are always displayed even if False. It is good to set to True only when debugging.


def get_free_port() -> int:
    """Have the OS assign an available port on localhost (to avoid conflicts with a fixed port)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


HTTP_PORT = None  # An available port is automatically assigned inside main()


# ---------- 1. Extraction Process ----------

DONE_MANIFEST_NAME = "_done.json"  # Completion marker placed collectively directly under OUTPUT_ROOT (one shared file for all assets)


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
    """atlas_json_path: SpineAtlasAsset json (e.g., SP_YLYYZ1.json).
    Read as a pair with <stem>_Data.json (SpineSkeletonDataAsset) in the same folder.
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
    """Copy texture PNGs to out_dir and rewrite page names inside the atlas.

    If multiple assets (with different stems) coexist in the same output folder (out_dir),
    there is a case where texture filenames are identical but contents are different.
    If copied with the same name, the asset processed later might mistakenly use the image
    from the asset processed earlier via "if not dest.exists(): skip" (image order/mix-up bug).
    To avoid this, always append the stem to the destination filename to make it unique,
    and rewrite the page name lines inside atlas_text to that new name.

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
        # _Data.json is on the skel side, so process as a pair from the atlas side (main body)
        if json_path.stem.endswith("_Data"):
            continue
        # Skip texture-related JSONs under Textures (though also filtered by Type judgment, just in case)
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
            print(f"[Skip] {manifest_key} : Completion marker exists (already processed)")
            continue

        images = atlas_page_images(atlas_text)
        out_dir = OUTPUT_ROOT / char_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        new_atlas_text, pngs, missing = stage_textures(atlas_text, images, char_dir, stem, out_dir)

        if not pngs:
            print(f"[Skip] {json_path.relative_to(SOURCE_ROOT)} : Texture not found ({', '.join(images)})")
            continue
        if missing:
            print(f"[Notice] {json_path.relative_to(SOURCE_ROOT)} : Missing textures {missing}")

        skel_name, atlas_name = f"{stem}.skel", f"{stem}.atlas"
        (out_dir / skel_name).write_bytes(skel_bytes)
        (out_dir / atlas_name).write_text(new_atlas_text, encoding="utf-8")

        entries.append((char_dir.name, skel_name, atlas_name, manifest_key))
        print(f"[Extraction OK] {manifest_key} : png x{len(pngs)}")

    return entries


# ---------- 2. Rendering HTML Harness (Internal use, users do not open) ----------

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
const gl = canvas.getContext("webgl", {alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: true});
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

    // For debugging: list draw order (slot order), attachments, and blend modes.
    // Used to identify which slot causes weird white plates, etc.
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

// Placeholder slots such as "counters/data displays" dynamically inserted inside games
// may stand out and appear empty (white plates, etc.) when exported as still images.
// Slots whose names contain these keywords are forcibly hidden every time an animation is applied.
const HIDE_SLOT_KEYWORDS = ['数字', '数据', 'shuju'];
// Diagnostic: if true, hides all slots whose blend mode is Screen (=3).
// A flag to isolate whether the cause of "white plates" not eliminated by keyword specification
// lies in overall Screen composition. Once the cause is identified, it should ideally revert to individual slot name specification.
const DEBUG_HIDE_ALL_SCREEN_BLEND = true;
function hideDebugSlots() {
  skeleton.slots.forEach(s => {
    const nameHit = HIDE_SLOT_KEYWORDS.some(kw => s.data.name.includes(kw));
    const screenHit = DEBUG_HIDE_ALL_SCREEN_BLEND && s.data.blendMode === 3;
    if (nameHit || screenHit) {
      s.setAttachment(null);
    }
  });
}

window.renderFrame = function(animName, time, bgR, bgG, bgB) {
  // bgR/bgG/bgB are background colors from 0 to 1. Additive/screen blend VFX do not composite correctly
  // on a transparent canvas (colors or opacity become broken), so they are always rendered on an opaque background.
  // When transparency is required, render twice (black background version and white background version) and calculate alpha from the difference
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

// For optimization: process a range of frames together in a single browser invocation.
// If Python <-> browser round-trips occur frame by frame, that communication overhead accumulates
// by frame count x 2 (black background / white background) and becomes the dominant bottleneck,
// so render frames in the specified range together and return them in an array.
// Intermediate captures use PNG (lossless) (JPEG caused a bug where low-alpha determinations flickered
// per frame due to compression noise, causing transparent videos to blink).
// When needAlpha is false, skip white background rendering entirely (optimization when transparent mov is unnecessary.
// In this case, use the black background rendering result directly as opaque).
window.renderFrameBatch = function(animName, startFrame, count, fps, needAlpha) {
  const anim = animations[animName];
  const blackUrls = [];
  const whiteUrls = [];
  for (let k = 0; k < count; k++) {
    const t = (startFrame + k) / fps;
    skeleton.setToSetupPose();
    anim.apply(skeleton, 0, t, false, null, 1, spine.MixBlend.setup, spine.MixDirection.mixIn);
    hideDebugSlots();
    skeleton.updateWorldTransform(spine.Physics.update);

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    renderer.begin();
    renderer.drawSkeleton(skeleton, false);
    renderer.end();
    blackUrls.push(canvas.toDataURL("image/png"));

    if (needAlpha) {
      gl.clearColor(1, 1, 1, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      renderer.begin();
      renderer.drawSkeleton(skeleton, false);
      renderer.end();
      whiteUrls.push(canvas.toDataURL("image/png"));
    }
  }
  return { black: blackUrls, white: whiteUrls };
};

window.getAnimDuration = function(animName) {
  return animations[animName].duration;
};

// Sample the entire animation (multiple frames) to find the sum of bounds ranges,
// and readjust the camera for that animation. If the camera is fixed looking only at the setup pose,
// it will drift from the actual rendering position in high-movement VFX, etc.
window.fitCameraToAnim = function(animName, samples) {
  const anim = animations[animName];
  const n = Math.max(1, samples || 20);
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
    """Decode the PNG data URL from canvas.toDataURL() into an RGB float32 array"""
    png_bytes = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float32)


def compose_alpha_from_black_white(black_arr: np.ndarray, white_arr: np.ndarray) -> Image.Image:
    """Restore the original alpha value and color from two rendering results on black/white backgrounds (difference matte).

    VFX such as screen/additive blending do not composite correctly when drawn directly on a transparent canvas
    (colors or opacity collapse), so they are always rendered twice on opaque backgrounds, and reverse-calculated from the difference.
    Assuming "over" composition: white - black = (1-alpha) (difference in background color contribution),
    so alpha = 1 - (white - black), and straight_color = black / alpha is used for restoration.

    However, this division is vulnerable to noise in regions with very low alpha (almost invisible),
    causing parts that should be nearly transparent and invisible to have only their color abnormally amplified,
    raising weird color stains/bleeds. To prevent this:
      - Place an upper limit on color amplification (alphas lower than ALPHA_FLOOR do not amplify color any further)
      - Furthermore, treat alphas lower than (less than ALPHA_CUTOFF) as completely transparent.
    """
    ALPHA_FLOOR = 48.0   # Cap color amplification at this value for alphas below this
    ALPHA_CUTOFF = 40.0  # Treat alphas below this as noise and make them completely transparent (alpha=0)

    diff = white_arr - black_arr
    alpha = 255.0 - np.clip(diff.max(axis=2), 0.0, 255.0)
    alpha_safe = np.maximum(alpha, ALPHA_FLOOR)
    color = np.clip(black_arr * 255.0 / alpha_safe[..., None], 0.0, 255.0)
    alpha = np.where(alpha < ALPHA_CUTOFF, 0.0, alpha)
    rgba = np.dstack([color, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def opaque_rgba_from_black(black_arr: np.ndarray) -> Image.Image:
    """For cases where transparency is unnecessary (MAKE_ALPHA_MOV=False): omit white background rendering,
    and use the black background rendering result directly as an opaque image with alpha 255."""
    alpha = np.full(black_arr.shape[:2], 255.0, dtype=np.float32)
    rgba = np.dstack([black_arr, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cleanup_keep_only_deliverables(out_dir: Path):
    """Delete everything under out_dir except *.mp4 / *_alpha.mov
    (sequential frame PNGs, extracted .skel/.atlas/texture PNGs, etc.)."""
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.suffix.lower() != ".mp4" and not item.name.endswith("_alpha.mov"):
            item.unlink(missing_ok=True)


def encode_outputs(rgba_pattern: str, out_dir: Path, name: str, canvas_dim: int):
    """Export both <name>.mp4 for normal playback (composited on black background, opaque) and <name>_alpha.mov
    for background transparency (PNG codec + alpha, only when MAKE_ALPHA_MOV is enabled) from a single type of rgba_pattern (RGBA sequential PNGs restored via difference matte).

    Previously, a separate sequence (flat_) directly drawn on a black background was saved for mp4,
    but since it was verified to yield numerically identical results to "compositing RGBA onto a black background using the overlay filter then converting to yuv420p",
    it was unified to rely solely on 1 type of frame_ (RGBA) (reducing saved files and simplifying things).

    Transparency was initially tried with VP9 (webm), but it was confirmed that alpha channels were not decoded correctly in many players like VLC/Discord/YMM4,
    so it was changed to the more reliable .mov containing the PNG codec. However, because PNG requires decoding frame by frame, it placed a heavy load on playback software, causing dropped frames/flickering,
    so it was changed to ProRes 4444 which supports proper video compression while also supporting alpha.
    The mp4 side is kept straightforward using yuv420p + simple settings to avoid color shift
    (yuv444p + full range specification was tried, but instead changed colors in some playback environments, so it was reverted).

    Returns: (mp4_ok, mov_ok). Success is judged by file size and ffmpeg exit code
    (checking only existence might treat a broken/unplayable file as a success).
    """
    mp4_path = out_dir / f"{name}.mp4"
    r1 = subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=black:s={canvas_dim}x{canvas_dim}:r={FPS}",
        "-framerate", str(FPS), "-i", rgba_pattern,
        "-filter_complex", "[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "15", "-preset", "slow",
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
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
            str(mov_path)
        ], check=False, capture_output=True)
        mov_ok = mov_path.exists() and mov_path.stat().st_size > 1024 and r2.returncode == 0
        if not mov_ok:
            mov_path.unlink(missing_ok=True)
            print(f"    [ffmpeg mov (transparent) error] {name} : {r2.stderr.decode(errors='ignore')[-800:]}")

    return mp4_ok, mov_ok


def render_character(page, char_dir_name, skel_name, atlas_name):
    """Returns: True if conversion to mp4 for all animations succeeds"""
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
            timeout=600000,
        )
    except Exception:
        load_error = page.evaluate("window.loadError")
        raise RuntimeError(f"Load timeout (10 minutes). window.loadError={load_error!r}")

    load_error = page.evaluate("window.loadError")
    if load_error:
        raise RuntimeError(f"Load failed: {load_error}")

    anim_names = page.evaluate("window.animNames")
    all_ok = True

    for anim in anim_names:
        duration = page.evaluate("(a) => window.getAnimDuration(a)", anim)
        if not duration or duration <= 0:
            continue
        # Readjust camera according to the movement of this entire animation (countermeasure for positional drift)
        page.evaluate("([a, s]) => window.fitCameraToAnim(a, s)", [anim, 20])
        n_frames = max(1, int(duration * FPS))
        frame_dir = out_dir / "frames" / anim
        frame_dir.mkdir(parents=True, exist_ok=True)

        BATCH_SIZE = 20  # Unit of frames sent to the browser together (reduces round-trips to speed up)
        idx = 0
        t_start = time.time()
        for start in range(0, n_frames, BATCH_SIZE):
            count = min(BATCH_SIZE, n_frames - start)
            result = page.evaluate(
                "([a, s, c, fps, na]) => window.renderFrameBatch(a, s, c, fps, na)",
                [anim, start, count, FPS, MAKE_ALPHA_MOV]
            )
            for k in range(count):
                black_arr = decode_frame_rgb(result["black"][k])
                if MAKE_ALPHA_MOV:
                    white_arr = decode_frame_rgb(result["white"][k])
                    rgba_img = compose_alpha_from_black_white(black_arr, white_arr)
                    del white_arr
                else:
                    rgba_img = opaque_rgba_from_black(black_arr)
                rgba_img.save(frame_dir / f"frame_{idx:04d}.png")
                del black_arr, rgba_img
                idx += 1
            del result
            gc.collect()

            elapsed = time.time() - t_start
            pct = idx / n_frames * 100
            eta = (elapsed / idx) * (n_frames - idx) if idx > 0 else 0
            print(f"    [Progress] {char_dir_name} / {anim} : {idx}/{n_frames} frames "
                  f"({pct:.0f}%) Elapsed {elapsed:.0f}s Remaining approx. {eta:.0f}s", flush=True)

        print(f"[Export OK] {char_dir_name} / {anim} : {n_frames} frames -> {frame_dir}")

        if MAKE_VIDEO and shutil.which("ffmpeg"):
            mp4_ok, mov_ok = encode_outputs(
                str(frame_dir / "frame_%04d.png"), out_dir, anim, CANVAS_DIM
            )
            if mp4_ok:
                print(f"    Video also exported: {out_dir / (anim + '.mp4')}"
                      + (f" / {out_dir / (anim + '_alpha.mov')}" if mov_ok else ""))
                shutil.rmtree(frame_dir, ignore_errors=True)
            else:
                print(f"    [Error] {char_dir_name} / {anim} : Failed to convert to mp4, leaving sequential PNGs as is -> {frame_dir}")
                all_ok = False
        else:
            all_ok = False

    harness_path.unlink(missing_ok=True)

    if all_ok:
        cleanup_keep_only_deliverables(out_dir)
        print(f"    [Cleanup] {char_dir_name} : Deleted everything except mp4/mov (sequential frame PNGs, extracted skel/atlas/texture)")
    else:
        print(f"    [Notice] {char_dir_name} : Intermediate files kept due to partial failure (will retry next time)")

    return all_ok


BROWSER_RESTART_EVERY = 2   # Restart the browser and reset memory every time this many characters are processed


def new_browser_and_page(p):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": CANVAS_DIM, "height": CANVAS_DIM})
    page.on("console", lambda msg: (
        print(f"    [Browser console] {msg.type}: {msg.text}")
        if SHOW_BROWSER_LOGS or msg.type in ("warning", "error")
        else None
    ))
    page.on("pageerror", lambda exc: print(f"    [Browser pageerror] {exc}"))
    return browser, page


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
            browser, page = new_browser_and_page(p)
            for i, (char_dir_name, skel_name, atlas_name, manifest_key) in enumerate(entries):
                if i > 0 and i % BROWSER_RESTART_EVERY == 0:
                    print(f"    [Maintenance] Processed {BROWSER_RESTART_EVERY} characters, restarting browser to free memory")
                    browser.close()
                    gc.collect()
                    browser, page = new_browser_and_page(p)
                try:
                    ok = render_character(page, char_dir_name, skel_name, atlas_name)
                    if ok:
                        done_manifest[manifest_key] = {
                            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        save_done_manifest(done_manifest)
                except Exception as e:
                    print(f"[Error] {char_dir_name} : {e}")
            browser.close()
    finally:
        httpd.shutdown()

    print(f"\nDone. Output written under {OUTPUT_ROOT} "
          f"(<anim>.mp4 for normal playback, <anim>_alpha.mov for transparent background).")


if __name__ == "__main__":
    main()