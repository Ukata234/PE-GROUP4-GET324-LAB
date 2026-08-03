# PE-GROUP4-GET324-LAB

# Concrete Bridge Deck Crack Detection

Cracked vs Non-cracked classifier for SDNET2018 bridge-deck image patches. Two models
trained and compared under an identical, leakage-safe evaluation protocol.

## Results (test set, 447 images)

| Model | Accuracy | Balanced Acc. | F1 (Cracked) | Recall (Cracked) | Precision (Cracked) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Custom CNN | 0.506 | 0.500 | 0.000 | 0.00 | 0.00 | 0.351 | 0.477 |
| **EfficientNetV2-B0** (fine-tuned) | **0.689** | **0.686** | **0.559** | 0.40 | 0.94 | **0.773** | **0.816** |

`best_model.keras` = EfficientNetV2-B0. It's the only usable model here — the Custom CNN
collapsed to predicting Non-cracked for essentially every image (ROC-AUC 0.35, below
random chance) and is included only as a documented failed baseline, not a working option.

## What this model actually can't do yet

- **Misses roughly 6 in 10 real cracks.** Recall on the Cracked class is 0.40. When it does
  flag something as Cracked it's usually right (precision 0.94), but it's far too
  conservative to rely on for catching defects — don't deploy this as a pass/fail gate
  without lowering the decision threshold and re-validating the new tradeoff.
- **The Custom CNN baseline is broken, not just weak** — treat any comparison against it as
  informational only, not evidence that transfer learning "barely helps." It needs a real
  debugging pass (learning rate, whether class balancing actually reached its training call)
  before it's a fair comparison point.
- **Deck-only, patch-scale images.** Not validated on pavements, walls, or full-photo (not
  cropped-patch) inputs.

## How the evaluation avoids the usual traps

- **Group-aware splitting.** SDNET2018's 13,620 image patches trace back to only 66 source
  photographs. A random split would let near-duplicate patches from the same source photo
  land in both train and test, inflating metrics without the model generalizing. Every split
  here is done by `GroupShuffleSplit` on an extracted source-photo group ID — verified zero
  leakage across train/val/test.
- **Undersampling done at the group level**, not per-image, so no group is left partially
  represented across splits. Target ratio 2:1 (Non-cracked:Cracked), achieved 2.03:1.
  `class_weight` for training on the full imbalanced set was also computed and is available
  as an alternative — not used by default.
- **Metrics beyond accuracy.** Balanced accuracy, F1, ROC-AUC, and PR-AUC are all reported
  specifically because accuracy alone would hide the Custom CNN's collapse (0.506 accuracy
  looks almost passable; 0.00 F1 on the class that matters does not).

## Reproducing this

```bash
# Requires kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY set
jupyter notebook concrete_deck_crack_detection.ipynb
```

Run top to bottom on a fresh runtime. Dataset auto-downloads via Kaggle API in Section 2.
GPU strongly recommended — training runs (40 + 15 + 20 epochs across two models) took
several minutes per epoch on a T4.

## Output artifacts

- `best_model.keras` — EfficientNetV2-B0, best checkpoint by validation AUC
- `best_custom_cnn.keras` — Custom CNN checkpoint (broken baseline, see above)
- `label_map.json` — class index mapping
- `test_set_results.csv` — per-sample test predictions

