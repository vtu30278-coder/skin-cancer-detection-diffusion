# AI-Based Early Skin Cancer Detection and Medical Image Generation

 project: skin lesion classification (HAM10000, 7 classes) with EfficientNet-B0, a class-conditional diffusion model for minority-class image generation, Grad-CAM explainability, and a Gradio demo.

**Research prototype. Not a medical device.**

## Results (test set, lesion-level split, no leakage)
| Experiment | Bal. acc | Macro F1 | AUC | Melanoma recall |
|---|---|---|---|---|
| E1 baseline | 0.672 | 0.640 | 0.944 | 0.604 |
| E2 oversampling | 0.701 | 0.658 | 0.955 | 0.594 |
| E3 oversampling + synthetic | 0.699 | 0.655 | 0.947 | 0.508 |

Oversampling helped. The 167 filtered synthetic images did not improve results and reduced melanoma recall.

## Dataset
HAM10000 (Tschandl et al., 2018), CC-BY-NC-SA 4.0. Kaggle: `kmader/skin-cancer-mnist-ham10000`. Images are not included in this repo.

## Limitations
Single runs, small diffusion model trained from scratch at 64x64, low yield after filtering, melanoma/nevus confusion.
