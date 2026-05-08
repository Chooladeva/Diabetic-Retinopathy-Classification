> **Note:** GitHub often fails to render large Jupyter Notebooks. To view the full project with all plots and training results, please use the DagsHub link below:
>
> [![View on DagsHub](https://dagshub.com/static/badge.svg)](https://dagshub.com/Chooladeva/Diabetic-Retinopathy-Classification)

## Diabetic Retinopathy Grading — Model Evaluation & Results

### Overview

This document covers the complete evaluation of both models — EfficientNet-B4 (transfer learning) and RetinalNet (custom CNN trained from scratch) — on the diabetic retinopathy grading task. Evaluation goes beyond standard metrics to include interpretability analysis, calibration assessment, and a real-world prevalence alignment test on a large unlabelled dataset of approximately 53,000 retinal images.

### Evaluation Metrics Used

| Metric                                 | Why It Was Used                                                                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QWK (Quadratic Weighted Kappa)         | Primary metric — accounts for ordinal class ordering and class imbalance. Predicting Grade 4 as Grade 0 is penalised far more than predicting Grade 1 as Grade 0 |
| Accuracy                               | Reported but used cautiously — misleading due to 73.5% Grade 0 prevalence                                                                                        |
| MCC (Matthews Correlation Coefficient) | Reliable single-number summary for imbalanced multiclass problems                                                                                                |
| Precision / Recall / F1                | Per-class breakdown to identify where each model succeeds and fails                                                                                              |
| ROC AUC                                | General screening ability — distinguishing diseased from healthy                                                                                                 |
| Average Precision (AP)                 | PR curve summary — more informative than ROC for imbalanced data                                                                                                 |
| Reliability Diagrams                   | Model calibration — how well predicted probabilities reflect true outcomes                                                                                       |

### Overall Results

RetinalNet outperforms EfficientNet-B4 in 6 out of 9 evaluation categories. However, EfficientNet retains a critical advantage on clinically important metrics — particularly Grade 4 recall, which directly determines how many of the most severe, vision-threatening cases are caught.

| Metric          | EfficientNet-B4 |      RetinalNet | Better Model    |
| --------------- | --------------: | --------------: | --------------- |
| QWK             |           0.611 |          0.6399 | RetinalNet      |
| Accuracy        |           73.7% |           76.5% | RetinalNet      |
| MCC             |          0.3328 |          0.3856 | RetinalNet      |
| ROC AUC         |          0.7877 |          0.7719 | EfficientNet-B4 |
| Grade 4 Recall  |            0.74 |            0.60 | EfficientNet-B4 |
| Grade 2 AP      |            0.44 |            0.55 | RetinalNet      |
| Calibration     |   Overconfident | Well-calibrated | RetinalNet      |
| Grade 1 Recall  |            0.11 |            0.04 | EfficientNet-B4 |
| Inference Speed |    Slower (TTA) |          Faster | RetinalNet      |

#### Per-Class Performance

**EfficientNet-B4 (with Test-Time Augmentation)**

| Grade             | Precision | Recall |   F1 | Support |
| ----------------- | --------: | -----: | ---: | ------: |
| 0 — Healthy       |      0.82 |   0.91 | 0.86 |   5,158 |
| 1 — Mild          |      0.13 |   0.11 | 0.12 |     503 |
| 2 — Moderate      |      0.58 |   0.21 | 0.31 |   1,048 |
| 3 — Severe        |      0.40 |   0.57 | 0.47 |     176 |
| 4 — Proliferative |      0.41 |   0.74 | 0.52 |     141 |

**RetinalNet (Standard Single-Pass Inference)**

| Grade             | Precision | Recall |   F1 | Support |
| ----------------- | --------: | -----: | ---: | ------: |
| 0 — Healthy       |      0.83 |   0.94 | 0.88 |   5,158 |
| 1 — Mild          |      0.10 |   0.04 | 0.05 |     503 |
| 2 — Moderate      |      0.71 |   0.28 | 0.40 |   1,048 |
| 3 — Severe        |      0.30 |   0.61 | 0.40 |     176 |
| 4 — Proliferative |      0.50 |   0.60 | 0.55 |     141 |

**Grade-by-Grade Interpretation**

- Grade 0 — Healthy

Both models perform well, as expected given the abundance of training data. RetinalNet has slightly higher recall (0.94 vs 0.91), meaning it correctly identifies more healthy eyes. However this also means it is slightly more likely to call a diseased eye healthy.

- Grade 1 — Mild DR

The weakest class for both models. EfficientNet catches 11% of mild cases while RetinalNet catches only 4%. Both are clinically concerning — mild DR is the earliest detectable stage and the optimal point for intervention. The failure here is primarily due to visual similarity with healthy eyes and limited training data.

- Grade 2 — Moderate DR

RetinalNet is meaningfully better — precision of 0.71 versus EfficientNet's 0.58. When RetinalNet predicts moderate DR, it is correct 71% of the time. Both models still have low recall (0.21 and 0.28), missing most moderate cases.

- Grade 3 — Severe DR

EfficientNet has higher precision (0.40 vs 0.30) while RetinalNet has marginally higher recall (0.61 vs 0.57). EfficientNet's predictions are more trustworthy, but RetinalNet catches slightly more true cases. Both models perform relatively well here because severe DR has visually distinctive features.

- Grade 4 — Proliferative DR

EfficientNet's recall of 0.74 is its most clinically valuable result — it catches nearly three quarters of the most urgent cases. RetinalNet's 0.60 recall, while respectable, misses 40% of these critical patients. From a deployment standpoint, EfficientNet's higher sensitivity for the most severe grade is a significant clinical advantage.

**Test-Time Augmentation (TTA)**

TTA was applied to EfficientNet-B4 during final evaluation. Each test image is passed through the model in four orientations — original, horizontal flip, vertical flip, and 90° rotation — and the probability distributions are averaged before the final class is selected.

Why TTA was not applied to RetinalNet:

TTA was tested on RetinalNet but decreased performance. EfficientNet-B4, trained with ImageNet pretraining, produces consistent probability estimates across augmented views because its features are orientation-stable. RetinalNet, trained from scratch on a smaller domain-specific dataset, is more sensitive to image orientation and produces inconsistent outputs across views. Averaging inconsistent distributions degrades rather than improves predictions.

#### Model Interpretability — Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualise which regions of each retinal image each model focused on when making its prediction. This provides insight into whether the models are attending to clinically meaningful areas or being distracted by irrelevant features.

**EfficientNet-B4**

- Shows fine-grained, localised attention — focuses on small specific regions rather than broad areas
- For correct predictions in Grades 1 and 3, attention concentrates on small lesion regions, reflecting the pretrained backbone's ability to detect subtle textures
- Failure mode: In some incorrect predictions, attention shifts to image edges and borders rather than the retinal region, suggesting the model can be distracted by imaging artifacts
- For Grade 4, attention spreads across multiple retinal regions, consistent with the widespread abnormalities characteristic of proliferative DR

**RetinalNet**

- Shows broader, more structured attention — often forms circular or halo-like patterns centred on the retinal region
- For correct predictions in Grades 3 and 4, attention concentrates on the central retina and major blood vessels — clinically appropriate regions where haemorrhages are typically found
- Failure mode: Some incorrect predictions display horizontal striping patterns in the heatmaps, indicating a spatial bias likely introduced during preprocessing (cropping and padding). In Grade 2
- cases misclassified as Grade 0, attention shifts to image edges rather than the central retinal region, causing the model to miss pathological features

**Key insight: Both models are learning meaningful retinal patterns rather than random noise. EfficientNet is better at detecting fine local features but more vulnerable to edge artifacts. RetinalNet captures broader retinal structure better but can be affected by preprocessing-induced spatial bias.**

#### Model Calibration — Reliability Diagram Analysis

Reliability diagrams measure how well a model's predicted probabilities reflect actual outcomes. A perfectly calibrated model's confidence curve follows the diagonal exactly — if it predicts 70% probability, the true frequency should be 70%.

**EfficientNet-B4**

- The reliability curve lies below the diagonal across the lower probability range (0.0–0.5)
- This indicates overconfidence — the model assigns higher probabilities than the actual observed frequencies
- Example: when EfficientNet predicts 40% probability of disease, the true occurrence may be closer to 30%
- Clinically, this can lead to overestimation of risk in borderline cases and more false positives

**RetinalNet**

- The reliability curve closely follows the diagonal across most probability ranges
- This indicates strong calibration — predicted probabilities align well with actual outcomes
- When RetinalNet assigns a confidence score to a prediction, that score is genuinely informative
- Clinically valuable — doctors can rely on the model's probability outputs when making decisions

**Why RetinalNet is better calibrated: Both models use the same label smoothing (0.05), but RetinalNet's domain-specific from-scratch training, combined with its two-stage dropout and BatchNorm in the classifier head, naturally produces better-calibrated outputs. EfficientNet's stronger pretrained features can sometimes produce overconfident predictions that label smoothing alone cannot fully counteract.**

#### Prevalence Alignment Analysis

Both models were used to generate predictions on a large unlabelled dataset of approximately 53,000 retinal images (~53GB). The predicted class distributions were compared against the known training distribution to test whether the models behave realistically on naturally distributed real-world data.

**What Was Measured:** A well-calibrated model should produce a prediction distribution that roughly mirrors the true population prevalence. Significant deviation — particularly over-predicting the majority class — indicates inference-time bias.

**Results:**

| Grade             | Training Prevalence | EfficientNet Predicted | RetinalNet Predicted |
| ----------------- | ------------------: | ---------------------: | -------------------: |
| 0 — Healthy       |               73.5% |         88.8% (+15.3%) |       83.6% (+10.1%) |
| 1 — Mild          |                6.9% |           1.2% (−5.7%) |         2.7% (−4.2%) |
| 2 — Moderate      |               15.1% |          3.0% (−12.1%) |         6.1% (−9.0%) |
| 3 — Severe        |                2.5% |                   3.7% |                 4.9% |
| 4 — Proliferative |                2.0% |                   3.2% |                 2.7% |

**Key Observations**

- Both models over-predict Grade 0 significantly above its true prevalence — EfficientNet by 15.3 percentage points and RetinalNet by 10.1 points. This is not a design failure. It is a well-documented and expected consequence of training on severely imbalanced DR datasets, confirmed by published research using the same EyePACS data.
- Grade 1 nearly disappears from both prediction distributions — dropping from 6.9% true prevalence to 1.2% (EfficientNet) and 2.7% (RetinalNet). This is consistent with published findings that Grade 1 is the most severely under-predicted class across DR models trained on imbalanced data, due to its visual similarity to Grade 0 and limited training examples.
- RetinalNet shows less bias — its Grade 0 over-prediction is 5.2 percentage points lower than EfficientNet, and its Grade 2 predictions (6.1%) are approximately double EfficientNet's (3.0%), which directly explains RetinalNet's superior Grade 2 precision and AP scores.

**Why Training Techniques Cannot Fully Eliminate This Bias**

The weighted sampler creates balanced batches during training but does not change the true data distribution the model encounters at inference time. When the model sees real-world data where 73.5% of images are genuinely Grade 0, its learned prior toward the majority class reasserts itself regardless of the training-time balancing strategies applied.

This phenomenon is formally known as label shift or prevalence shift in the medical imaging literature. Research by Godau et al. (2025) confirms that addressing it requires prevalence-aware recalibration at inference time — adjusting predicted probabilities to better reflect the true population distribution — rather than training-level interventions alone.

**Clinical Implications**

In a real screening scenario:

| Model           | Patients Flagged for Review | Estimated DR Cases Missed |
| --------------- | --------------------------: | ------------------------: |
| EfficientNet-B4 |        ~11.2% of population |     ~57% of true DR cases |
| RetinalNet      |        ~16.4% of population |     ~38% of true DR cases |
| Ideal target    |        ~26.5% of population |     <10% of true DR cases |

Neither model is yet suitable as a standalone clinical screening tool, but both demonstrate genuine learning — QWK scores well above chance — and could function effectively as a triage system to prioritise the most urgent cases for specialist review.

**Which Model to Use**

The right choice depends on the clinical objective:

Choose EfficientNet-B4 if:

- The priority is catching the most severe cases (Grade 4 recall: 0.74)
- Deploying as a screening tool where missing severe disease is the primary risk
- Higher sensitivity is preferred even at the cost of more false positives

Choose RetinalNet if:

- The priority is reliable probability estimates for clinical decision-making (better calibration)
- Identifying moderate DR (Grade 2) with high precision is important
- Faster inference without TTA is required
- A more interpretable, domain-specific model is preferred

**Ideal approach — Ensemble both models: Averaging the probability outputs of both models before making a final prediction would likely outperform either individually. The two models make systematically different errors — EfficientNet is better at the severity extremes, RetinalNet is better at intermediate grades — meaning their errors are unlikely to be highly correlated. An ensemble would produce a more balanced and robust prediction profile across all five DR grades.**

**Shared Limitations**

Both models share limitations that should be considered before any deployment:

| Limitation                       | Detail                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------- |
| Grade 1 detection failure        | Both models miss most mild DR cases (recall: 0.11 and 0.04)                       |
| Training-test QWK gap            | Indicates overfitting — partly from weighted sampling creating distribution shift |
| Majority class bias at inference | Both over-predict Grade 0 on real-world data                                      |
| Grade 2 low recall               | Both miss majority of moderate DR cases                                           |
| Not clinically validated         | Results are from a research dataset, not a validated clinical trial               |

#### Recommendations for Improvement

| Recommendation                    | Expected Impact                                                   |
| --------------------------------- | ----------------------------------------------------------------- |
| CLAHE preprocessing               | Improve Grade 1 detection through better microaneurysm visibility |
| Ensemble both models              | 3–5 point QWK improvement from complementary error profiles       |
| Prevalence-aware recalibration    | Reduce majority class bias at inference time                      |
| DR-specific pretraining           | Largest potential improvement — close the gap toward 0.90 QWK     |
| Ordinal loss function             | Directly aligned with QWK optimisation                            |
| Two-stage classification pipeline | Separate healthy vs. any DR detection from severity grading       |
| Tighter retinal cropping          | Address edge artifact failure modes in both models                |
