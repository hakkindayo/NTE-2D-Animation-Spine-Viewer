# NTE-2D-Animation-Spine-Viewer

A Python script that automatically extracts, reconstructs, and batch-exports Spine animations (PNG sequences and videos) from FModel-exported JSON files, tailored for NTE assets.

The browser is operated automatically in the background via Playwright (headless Chromium), so no manual browser operation is required.

---

## Features

1. **Asset Reconstruction:**
   - Reconstructs `.skel` and `.atlas` files from FModel-exported JSONs under the `UISpine` folder.
   - Groups them together with their corresponding PNG textures into asset-specific folders inside the output directory.
2. **Headless Deterministic Rendering:**
   - Renders all animations for each asset frame-by-frame using `spine-webgl`.
3. **Dual Video Output (via FFmpeg):**
   - **Standard Playback:** `<anim_name>.mp4` (Opaque, high quality, `yuv444p` to prevent color/highlight blurring).
   - **Transparent Background:** `<anim_name>_alpha.mov` (Lossless PNG codec + Alpha channel support).
4. **Automatic Cleanup:**
   - Automatically deletes intermediate sequence PNGs and extracted raw assets after successful conversion (keeps sequence PNGs as a fallback only if FFmpeg fails).
5. **Config Persistence:**
   - Source and destination paths are automatically saved to `nte_config.json` in the same directory.

---

## NTE Specific Handling

NTE differs from Wuthering Waves in its FModel export format:
- `SpineAtlasAsset` and `SpineSkeletonDataAsset` are split into separate JSON files (e.g., `SP_YLYYZ1.json` + `SP_YLYYZ1_Data.json`).
- The atlas property name is `RawData` (capitalized) instead of `rawData`.
- This script natively handles extraction supporting this two-file structure.

---

## Prerequisites

Run the following commands in your Command Prompt (first time only):

```bash
pip install playwright numpy pillow
playwright install chromium
