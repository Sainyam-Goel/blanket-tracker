# CH27 Ground Truth Labeling Memory

This folder contains the short clips and manual ground-truth labels used to move
the CH27 taping counter beyond the v2 F1 ceiling.

## Goal

The v2 taping counter was stuck around F1=0.85. We decided the next useful step
was more labeled data, not more manual threshold tuning. These labels will be
clustered into action windows and used to train/evaluate the next classifier.

## Labeling Convention

- `A` means `load`.
  - This corresponds to the lower table ROI.
  - In the labeler this is shown with the darker table color.
- `D` means `toss`.
  - This corresponds to the upper air ROI.
  - In the labeler this is shown with the lighter air color.
- Multiple adjacent frames may be marked for one physical load/toss action.
  - These should not be treated as independent events.
  - Training should cluster nearby frame labels into action windows.

## File Types

### `.mp4`

These are the 5-minute source clips to review manually.

### `.labels.json`

These are the important manual label files saved from the labeler. Use these for
training and evaluation.

### `.v2_detections.json`

These are only auto-suggestion caches from the old v2 detector. They help the
labeler pre-populate suggested tosses. They are not manual ground truth.

## Clips

| Priority | Clip | Source slice | Purpose |
|---|---|---|---|
| 1 | `gt_clip1_morning.mp4` | 10:35-10:40 | Peak production, both tables busy, restock pattern |
| 2 | `gt_clip2_prelunch.mp4` | 12:25-12:30 | Pre-lunch slowdown and break-edge behavior |
| 3 | `gt_clip3_postlunch.mp4` | 14:15-14:20 | Post-lunch ramp-up and first-toss weakness |
| 4 | `gt_clip4_afternoon_dark.mp4` | 15:30-15:35 | Afternoon/darker blanket SKU variation |
| 5 | `gt_clip5_endofday.mp4` | 18:45-18:50 | End-of-day slower/casual motion patterns |

## Current Saved Labels

As of the last review:

| File | Status |
|---|---|
| `gt_clip1_morning.labels.json` | Saved |
| `gt_clip2_prelunch.labels.json` | Saved |
| `gt_clip3_postlunch.labels.json` | Saved |
| `gt_clip4_afternoon_dark.labels.json` | Saved after cleanup and resumed labeling |
| `gt_clip5_endofday.labels.json` | Not yet saved unless created later |

## Clip 4 Cleanup Note

For `gt_clip4_afternoon_dark.labels.json`, a mistaken right-table section was
removed:

- Removed right-table labels from `52.40s` through `79.88s`.
- Everything else was kept.
- The clip was then reopened and additional labels were saved.

## Labeler

Use the browser labeler:

```bash
python3 gt_labeler_web.py gt_clips/<clip-name>.mp4 --port <port>
```

Example:

```bash
python3 gt_labeler_web.py gt_clips/gt_clip5_endofday.mp4 --port 8769
```

Open the printed local URL in Chrome. Save with `Cmd+S` or the Save button.

## Next Training Step

When training:

1. Load every `*.labels.json`.
2. Cluster adjacent labels by `table`, `type`, and time proximity.
3. Treat each cluster as one action window.
4. Match candidate detector pulses against those windows.
5. Use the deleted/uncertain v2 suggestions and unmatched candidates as useful
   negative examples where appropriate.

