# Figure 1 Candidates — narrative-friendly samples for the concept illustration

**Goal.** Figure 1 illustrates *“local context is ambiguous, but a wider global context resolves the building”*. Candidates are ranked by a narrative-suitability heuristic combining foreground sweet-spot density, multiple connected components, small objects, boundary complexity, bright non-building surfaces (potential roads / parking lots), road-like long bright strips, and shadows next to buildings. We do **not** rank by per-model accuracy here.

**Source.** WHU test (manifest = `data/meta/whu_test.csv`), 2416 images scanned.
**Boxes-per-candidate target.** 4.
**Prior qualitative IDs reused (boost = +0.60).** 18 IDs.

## Candidates (ranked by narrative score)


| #   | sample id | feature tags                                                                                                      | rec.     | score | fg ratio | #cc | small cc | bndry cplx | bright bg | road bg | shadow | prior? | preview                           |
| --- | --------- | ----------------------------------------------------------------------------------------------------------------- | -------- | ----- | -------- | --- | -------- | ---------- | --------- | ------- | ------ | ------ | --------------------------------- |
| 001 | `2_256`   | small buildings, complex boundary, ambiguous bright surface, road interference, building shadows                  | **high** | 8.07  | 0.189    | 58  | 9        | 0.144      | 0.244     | 3       | 0.0346 | —      | `candidate_001/preview_panel.png` |
| 002 | `2_745`   | small buildings, complex boundary, ambiguous bright surface, road interference, building shadows                  | **high** | 8.06  | 0.127    | 49  | 15       | 0.143      | 0.276     | 2       | 0.0355 | —      | `candidate_002/preview_panel.png` |
| 003 | `2_210`   | dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows | **high** | 8.06  | 0.219    | 68  | 13       | 0.143      | 0.238     | 4       | 0.0473 | —      | `candidate_003/preview_panel.png` |
| 004 | `2_776`   | small buildings, complex boundary, ambiguous bright surface, road interference, building shadows                  | **high** | 8.05  | 0.166    | 57  | 8        | 0.146      | 0.197     | 2       | 0.0491 | —      | `candidate_004/preview_panel.png` |
| 005 | `2_435`   | small buildings, complex boundary, ambiguous bright surface, road interference, building shadows                  | **high** | 8.05  | 0.084    | 31  | 7        | 0.151      | 0.231     | 3       | 0.0186 | —      | `candidate_005/preview_panel.png` |
| 006 | `2_268`   | dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows | **high** | 8.03  | 0.203    | 62  | 17       | 0.139      | 0.208     | 2       | 0.0387 | —      | `candidate_006/preview_panel.png` |
| 007 | `2_655`   | small buildings, complex boundary, ambiguous bright surface, road interference, building shadows                  | **high** | 8.03  | 0.147    | 50  | 12       | 0.138      | 0.203     | 3       | 0.0373 | —      | `candidate_007/preview_panel.png` |
| 008 | `2_549`   | dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows | **high** | 8.03  | 0.216    | 67  | 16       | 0.138      | 0.249     | 3       | 0.0384 | —      | `candidate_008/preview_panel.png` |


## Per-candidate narrative

### candidate_001 — id `2_256`  (high)

- **Score**: 8.07
- **Feature tags**: small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.189, num_cc=58, small_cc=9, boundary_complexity=0.144, bright_bg_ratio=0.244, long_thin_bg_count=3, dark_near_building_ratio=0.0346.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=28, r1=32, c1=113))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=223, c0=99, r1=423, c1=299))
  - Patch 3: **road-like region** (source: bright_bg, bbox=(r0=207, c0=308, r1=387, c1=488))
  - Patch 4: **bright non-building surface** (source: bright_bg, bbox=(r0=67, c0=385, r1=219, c1=473))
- Files: `candidate_001/image.png`, `candidate_001/gt.png`, `candidate_001/image_with_boxes.png`, `candidate_001/patch_*.png`, `candidate_001/preview_panel.png`, `candidate_001/meta.json`.

### candidate_002 — id `2_745`  (high)

- **Score**: 8.06
- **Feature tags**: small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.127, num_cc=49, small_cc=15, boundary_complexity=0.143, bright_bg_ratio=0.276, long_thin_bg_count=2, dark_near_building_ratio=0.0355.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=0, r1=47, c1=36))
  - Patch 2: **bright non-building surface** (source: bright_bg, bbox=(r0=0, c0=160, r1=98, c1=367))
  - Patch 3: **visually similar to building roof** (source: bright_bg, bbox=(r0=99, c0=112, r1=299, c1=312))
  - Patch 4: **road-like region** (source: bright_bg, bbox=(r0=342, c0=0, r1=508, c1=133))
- Files: `candidate_002/image.png`, `candidate_002/gt.png`, `candidate_002/image_with_boxes.png`, `candidate_002/patch_*.png`, `candidate_002/preview_panel.png`, `candidate_002/meta.json`.

### candidate_003 — id `2_210`  (high)

- **Score**: 8.06
- **Feature tags**: dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.219, num_cc=68, small_cc=13, boundary_complexity=0.143, bright_bg_ratio=0.238, long_thin_bg_count=4, dark_near_building_ratio=0.0473.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=3, c0=305, r1=71, c1=371))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=301, c0=0, r1=512, c1=163))
  - Patch 3: **bright non-building surface** (source: bright_bg, bbox=(r0=0, c0=113, r1=108, c1=248))
  - Patch 4: **road-like region** (source: bright_bg, bbox=(r0=358, c0=358, r1=512, c1=512))
- Files: `candidate_003/image.png`, `candidate_003/gt.png`, `candidate_003/image_with_boxes.png`, `candidate_003/patch_*.png`, `candidate_003/preview_panel.png`, `candidate_003/meta.json`.

### candidate_004 — id `2_776`  (high)

- **Score**: 8.05
- **Feature tags**: small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.166, num_cc=57, small_cc=8, boundary_complexity=0.146, bright_bg_ratio=0.197, long_thin_bg_count=2, dark_near_building_ratio=0.0491.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=470, r1=37, c1=512))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=454, c0=125, r1=512, c1=274))
  - Patch 3: **shadowed structure** (source: shadow, bbox=(r0=12, c0=186, r1=113, c1=267))
  - Patch 4: **road-like region** (source: bright_bg, bbox=(r0=266, c0=21, r1=402, c1=95))
- Files: `candidate_004/image.png`, `candidate_004/gt.png`, `candidate_004/image_with_boxes.png`, `candidate_004/patch_*.png`, `candidate_004/preview_panel.png`, `candidate_004/meta.json`.

### candidate_005 — id `2_435`  (high)

- **Score**: 8.05
- **Feature tags**: small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.084, num_cc=31, small_cc=7, boundary_complexity=0.151, bright_bg_ratio=0.231, long_thin_bg_count=3, dark_near_building_ratio=0.0186.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=46, r1=57, c1=109))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=92, c0=64, r1=229, c1=186))
  - Patch 3: **bright non-building surface** (source: bright_bg, bbox=(r0=0, c0=344, r1=119, c1=512))
  - Patch 4: **road-like region** (source: bright_bg, bbox=(r0=367, c0=63, r1=512, c1=139))
- Files: `candidate_005/image.png`, `candidate_005/gt.png`, `candidate_005/image_with_boxes.png`, `candidate_005/patch_*.png`, `candidate_005/preview_panel.png`, `candidate_005/meta.json`.

### candidate_006 — id `2_268`  (high)

- **Score**: 8.03
- **Feature tags**: dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.203, num_cc=62, small_cc=17, boundary_complexity=0.139, bright_bg_ratio=0.208, long_thin_bg_count=2, dark_near_building_ratio=0.0387.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=148, r1=33, c1=213))
  - Patch 2: **road-like region** (source: bright_bg, bbox=(r0=0, c0=370, r1=131, c1=452))
  - Patch 3: **bright non-building surface** (source: bright_bg, bbox=(r0=168, c0=126, r1=368, c1=326))
  - Patch 4: **visually similar to building roof** (source: bright_bg, bbox=(r0=283, c0=277, r1=357, c1=392))
- Files: `candidate_006/image.png`, `candidate_006/gt.png`, `candidate_006/image_with_boxes.png`, `candidate_006/patch_*.png`, `candidate_006/preview_panel.png`, `candidate_006/meta.json`.

### candidate_007 — id `2_655`  (high)

- **Score**: 8.03
- **Feature tags**: small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.147, num_cc=50, small_cc=12, boundary_complexity=0.138, bright_bg_ratio=0.203, long_thin_bg_count=3, dark_near_building_ratio=0.0373.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=155, r1=36, c1=238))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=240, c0=238, r1=440, c1=438))
  - Patch 3: **road-like region** (source: bright_bg, bbox=(r0=48, c0=0, r1=203, c1=92))
  - Patch 4: **shadowed structure** (source: shadow, bbox=(r0=174, c0=146, r1=299, c1=238))
- Files: `candidate_007/image.png`, `candidate_007/gt.png`, `candidate_007/image_with_boxes.png`, `candidate_007/patch_*.png`, `candidate_007/preview_panel.png`, `candidate_007/meta.json`.

### candidate_008 — id `2_549`  (high)

- **Score**: 8.03
- **Feature tags**: dense buildings, small buildings, complex boundary, ambiguous bright surface, road interference, building shadows
- **Stats**: fg_ratio=0.216, num_cc=67, small_cc=16, boundary_complexity=0.138, bright_bg_ratio=0.249, long_thin_bg_count=3, dark_near_building_ratio=0.0384.
- **Auto-detected ambiguous regions**:
  - Patch 1: **small building / small object** (source: small_fg, bbox=(r0=0, c0=172, r1=36, c1=236))
  - Patch 2: **visually similar to building roof** (source: bright_bg, bbox=(r0=156, c0=142, r1=356, c1=342))
  - Patch 3: **small building / small object** (source: small_fg, bbox=(r0=0, c0=431, r1=45, c1=497))
  - Patch 4: **visually similar to building roof** (source: bright_bg, bbox=(r0=310, c0=168, r1=389, c1=255))
- Files: `candidate_008/image.png`, `candidate_008/gt.png`, `candidate_008/image_with_boxes.png`, `candidate_008/patch_*.png`, `candidate_008/preview_panel.png`, `candidate_008/meta.json`.

## Top-3 recommended candidates for Figure 1

### Top 1: candidate_001  (id `2_256`, score 8.07)

- **Why this works for Figure 1**:
  - contains large bright non-building surfaces that look like roofs in a small window
  - has long thin road-like bright strips that compete with buildings under local context
  - contains small buildings that are easy to miss without surrounding cues
  - has buildings with complex boundaries that benefit from a wider context window
  - has shadows adjacent to buildings, useful for shadow-vs-roof discussion
- **Suggested local-context windows** (each is one of the auto-detected red boxes; feel free to refine manually):
  - Patch 1: small building / small object
  - Patch 2: visually similar to building roof
  - Patch 3: road-like region
  - Patch 4: bright non-building surface
- **Files to start from**: `candidate_001/preview_panel.png` and `candidate_001/image_with_boxes.png`.

### Top 2: candidate_002  (id `2_745`, score 8.06)

- **Why this works for Figure 1**:
  - contains large bright non-building surfaces that look like roofs in a small window
  - has long thin road-like bright strips that compete with buildings under local context
  - contains small buildings that are easy to miss without surrounding cues
  - has buildings with complex boundaries that benefit from a wider context window
  - has shadows adjacent to buildings, useful for shadow-vs-roof discussion
- **Suggested local-context windows** (each is one of the auto-detected red boxes; feel free to refine manually):
  - Patch 1: small building / small object
  - Patch 2: bright non-building surface
  - Patch 3: visually similar to building roof
  - Patch 4: road-like region
- **Files to start from**: `candidate_002/preview_panel.png` and `candidate_002/image_with_boxes.png`.

### Top 3: candidate_003  (id `2_210`, score 8.06)

- **Why this works for Figure 1**:
  - contains large bright non-building surfaces that look like roofs in a small window
  - has long thin road-like bright strips that compete with buildings under local context
  - contains small buildings that are easy to miss without surrounding cues
  - has buildings with complex boundaries that benefit from a wider context window
  - has shadows adjacent to buildings, useful for shadow-vs-roof discussion
  - has multiple buildings + free-space, ideal for showing a global layout cue
- **Suggested local-context windows** (each is one of the auto-detected red boxes; feel free to refine manually):
  - Patch 1: small building / small object
  - Patch 2: visually similar to building roof
  - Patch 3: bright non-building surface
  - Patch 4: road-like region
- **Files to start from**: `candidate_003/preview_panel.png` and `candidate_003/image_with_boxes.png`.

## How to use these files when building the final Figure 1

1. Pick **one** candidate from the Top-3 above.
2. Open `<candidate_dir>/image.png` (clean) and `<candidate_dir>/image_with_boxes.png` (red-boxed) side-by-side.
3. Refine the local-window crops with a vector editor (Inkscape / Illustrator):
  - Use 2–3 of the auto-detected red boxes to label `local context` ambiguity.
  - Add a larger crop covering the same scene to label `global context`.
  - Annotate with arrows showing how the larger context disambiguates the local windows.
4. The `preview_panel.png` is for at-a-glance review, not the final figure; you will redraw it.

**Overview grid:** `all_candidates_overview.png`
**Per-image cached stats:** `scan_stats.json`