# Pipeline Audit — Honest State of Each Step

## Pipeline Steps & I/O

| # | Step | Script | Input | Output |
|---|---|---|---|---|
| 1 | Extract keyframes | `matchmove_ai.py prepare` | video.mp4 | `work/frames/*.jpg`, `perception_context.json` |
| 2 | VLM scene understanding | `qwen_vl_scene.py` | frames | `scene_brief.json` |
| 3 | Object detection | `grounding_dino_detect.py` | frames | `grounding_dino_detections.json` |
| 4 | Instance segmentation | `sam2_segment_boxes.py` | frames + detections | `sam2_segments.json`, `sam2_masks/*.png` |
| 5 | Merge perception | `merge_perception_outputs.py` | context + segments + scene_brief | `perception_context.json` (updated) |
| 6 | Annotate frames | `annotate_frames.py` | context + masks | `annotated_frames/*.jpg`, `spatial_summary.txt` |
| 7 | Camera tracking | `pycolmap_sfm.py` | frames | `tracking/camera_poses.json` |
| 8 | Story planning | `qwen_story_planner.py` | annotated frames + scene_brief + spatial_summary | `story_plan.json` |
| 9 | Asset search | `asset_search.py` | story_plan (visual_description per character) | `asset_manifest.json`, `assets/*.glb` |
| 10 | Plan | `matchmove_ai.py plan` | context + config + story_plan | `cgi_plan.json` |
| 11 | Render | `blender/insert_cgi.py` | perception_context + cgi_plan + GLBs | `result.mp4` |

---

## Step-by-Step Audit

### Step 2 — VLM Scene Understanding
**What it does:** Qwen2.5-VL reads keyframes, outputs scene description, CGI suggestion, insertion regions.  
**Hardcoded:** Prompt schema, max 12 frames, `user_location` must be passed manually.  
**Problem:** Qwen sometimes outputs non-English text. Scene misidentification (e.g. "pedestrian bridge" for a highway overpass). Output quality depends heavily on Qwen 7B's limitations.  
**Result:** `scene_brief.json` — usually correct at a coarse level, unreliable for precise scene type.

---

### Step 3 — Grounding DINO Detection
**What it does:** Open-vocabulary detection on keyframes using text prompts.  
**Hardcoded:** Prompt string (`"person. car. road. building. sign. sky. tree. walkway..."`) — **must be manually updated per scene type**.  
**Problem:** For new videos (shibuya_10s, shibuya_30s), this step was **skipped initially**, causing `instances=0` in context, making all downstream spatial reasoning useless.  
**Result:** `grounding_dino_detections.json` — bounding boxes per frame per label.

---

### Step 4 — SAM2 Segmentation
**What it does:** Produces pixel-level masks for each detected box.  
**Hardcoded:** Label filter list in `--labels` argument.  
**Problem:** Also **skipped for new videos**. Masks are generated but never used by Blender compositor for occlusion — the compositing node in `insert_cgi.py` does not read SAM2 masks. Segmentation output is currently unused beyond annotated frame visualization.  
**Result:** `sam2_masks/*.png` — exists but **disconnected from render**.

---

### Step 6 — Annotate Frames
**What it does:** Draws bounding boxes and masks on keyframes, produces `spatial_summary.txt` with insertion gaps.  
**Hardcoded:** Color scheme (red/green/cyan), gap detection threshold (`y > 0.55`, min 2 columns wide).  
**Problem:** Gap analysis is coarse (10-column grid). Gaps show `x=[0.0, 1.0]` for most frames — no real spatial specificity when scene has no detected ground label.  
**Result:** Annotated images are fed to Qwen in step 8. Useful only if step 3/4 produced real detections.

---

### Step 7 — Camera Tracking (SfM)
**What it does:** pycolmap reconstructs camera poses from keyframes.  
**Hardcoded:** Nothing — fully parametric.  
**Problem:** Only 24/80 frames registered for miyashita (30%). **Not run at all for shibuya videos.** `apply_sfm_camera_motion()` in `insert_cgi.py` reads the poses and animates the Blender camera — but with sparse coverage and no correspondence to the composited background video, objects still slide against the background.  
**Result:** `tracking/camera_poses.json` — exists for miyashita only, not wired to screen→world projection.

---

### Step 8 — Story Planning
**What it does:** Qwen reads annotated frames + obstacle map → generates narrative + characters + screen paths.  
**Hardcoded:**
- `_diversify_paths()` — mechanically remaps x-ranges into equal-width thirds. **Overrides whatever Qwen generates.**
- `_normalise()` — clamps y: ground ≥ 0.70, air ∈ [0.28, 0.65]. **All objects at fixed altitude.**
- Path point count fixed at 4–6 regardless of video duration or scene complexity.
- "2-3 objects" hardcoded in prompt.

**Problem:** Routes are always horizontal sweeps at constant y. Qwen's spatial reasoning is insufficient for temporal 3D path planning. Obstacle map (`person@(0.35,0.52)`) is screen-space and has no depth — Qwen cannot route around a 3D obstacle from a 2D centroid.  
**Result:** `story_plan.json` — narrative is scene-aware, **paths are not**.

---

### Step 9 — Asset Search
**What it does:** CLIP text embedding of `visual_description` → cosine similarity against 1156 Objaverse LVIS category names → download candidate GLBs → rank by annotation text similarity.  
**Hardcoded:** Nothing — fully semantic via CLIP.  
**Problem:**
- CLIP text-to-text for annotation ranking is weak (CLIP was trained for cross-modal, not text-text similarity).
- Category matching is coarse (1156 categories for 46K objects). Within a category, candidates are near-random.
- No visual verification — downloaded GLB may not resemble the text description at all.
- Object selection (what character to invent) is driven by Qwen with no semantic constraint → "butterfly drone" for a Shibuya highway.

**Result:** `asset_manifest.json` + `assets/*.glb` — GLBs are sometimes semantically correct (butterfly score 0.686), sometimes random.

---

### Step 10 — Plan
**What it does:** Merges story_plan into cgi_plan, stamps video metadata.  
**Hardcoded:** Nothing significant.  
**Result:** `cgi_plan.json` — correct.

---

### Step 11 — Render
**What it does:** Blender imports GLBs or builds procedural geometry, animates along screen paths, composites over background video.  
**Hardcoded:**
- `screen_to_world()` — fixed constants `(5.8, -1.4, 2.1, -0.35, 0.55)` not derived from camera intrinsics.
- Camera position `(0, -6, 2.4)` and rotation `68°` fixed regardless of actual shot.
- Shadow catcher removed (was causing grey bottom) but SAM2 occlusion masks still not used.
- Emissive tint overrides all GLB materials — real asset textures discarded.

**Problem:** Without correct camera projection, screen-space paths do not correspond to scene geometry. Objects slide against the moving background (no camera tracking). GLBs normalized to unit bounding box but material override removes original texture/appearance.  
**Result:** `result.mp4` — objects visible, compositing works, but objects are not matched to scene.

---

## Summary of Hardcoded Elements

| Location | Hardcoded thing | Effect |
|---|---|---|
| `qwen_story_planner.py` | `_diversify_paths()` | Forces same horizontal route structure every run |
| `qwen_story_planner.py` | y-clamp in `_normalise()` | Fixes object altitude to two bands |
| `qwen_story_planner.py` | "2-3 objects" in prompt | Object count not emergent from scene |
| `insert_cgi.py` | `screen_to_world()` constants | Wrong 3D placement for any camera that isn't the hardcoded one |
| `insert_cgi.py` | Camera position/rotation | Fixed camera, no calibration from SfM |
| `insert_cgi.py` | Material override (emissive tint) | GLB textures discarded |
| `grounding_dino_detect.py` | Detection prompt labels | Must be manually tuned per scene type |
| `pycolmap_sfm.py` | Not integrated into routing | SfM poses exist but screen→world still uses hardcoded constants |
| Compositor | SAM2 masks not loaded | Occlusion compositing not implemented |

---

## What Is Missing for the Pipeline to Actually Work

1. **Camera calibration wired to rendering**: SfM intrinsics (focal length, principal point) + extrinsics per frame → replace `screen_to_world()` with proper ray-cast onto estimated ground plane.

2. **Object tracking across frames**: DINO gives per-frame boxes. Need tracking (DeepSORT, ByteTrack) to get per-object trajectories → Qwen can then route CGI relative to where real objects move.

3. **SAM2 masks in compositor**: Per-frame mask images loaded as Blender image sequences → CGI goes behind foreground objects.

4. **Stronger planner or structured routing**: Qwen 7B cannot generate correct 3D paths from screen-space hints. Either use a stronger model (GPT-4o / Claude with vision) or replace path generation with a geometry-based planner that uses the SfM ground plane + tracked obstacle positions.

5. **Object selection grounded in scene type**: Story planner needs a scene-type classifier output ("highway", "park", "indoor") that constrains what objects are plausible before story generation.

6. **Agent feedback loop**: Render a draft → VLM evaluates → revise story/paths → re-render.
