## Diabetic Retinopathy Grading — EfficientNet-B4

### Overview

This module implements an automated diabetic retinopathy (DR) grading pipeline using EfficientNet-B4 with transfer learning. Given a retinal fundus photograph, the model classifies it into one of five severity grades:
0No DR (Healthy)
- 1- Mild DR
- 2- Moderate DR
- 3- Severe DR
- 4- Proliferative DR

The pipeline is designed to handle the severe class imbalance inherent in real-world DR datasets, where healthy eyes (Grade 0) represent over 73% of all images while proliferative DR (Grade 4) accounts for less than 2%.

### Model Architecture

Base Model: EfficientNet-B4 pretrained on ImageNet

Modification: The original 1000-class classification head is replaced with a 5-class DR-specific head:

EfficientNet-B4 Backbone (pretrained, frozen initially)
         ↓
    Dropout (p=0.5)
         ↓
  Linear (1792 → 5)
         ↓
   5 DR Grade Logits

### Training Strategy: Two-phase transfer learning

- Phase 1 (Epochs 1–3): Backbone frozen, only the classification head trains
- Phase 2 (Epoch 4+): Backbone unfrozen with discriminative learning rates — backbone at 1e-5, head at 1e-4

This prevents the randomly initialised head's noisy early gradients from corrupting the pretrained ImageNet features.

### Dataset & Class Imbalance Handling

The dataset contains 28,100 training images and 7,026 test images with the following distribution:

| Grade             | Train Count | Train % |
| ----------------- | ----------: | ------: |
| 0 — Healthy       |      20,652 |   73.5% |
| 1 — Mild          |       1,940 |    6.9% |
| 2 — Moderate      |       4,244 |   15.1% |
| 3 — Severe        |         697 |    2.5% |
| 4 — Proliferative |         567 |    2.0% |

Four mechanisms work together to address this imbalance:

**1. Weighted Random Sampler**

Each image is assigned a sampling weight equal to the inverse of its class frequency. Grade 4 images are approximately 36× more likely to be selected per batch than Grade 0 images, producing balanced batches throughout training.

**2. Class-Aware Data Augmentation**

- Grade 0: Random horizontal flip only
- Grades 1–4: Horizontal flip + random rotation (0°, 90°, 120°, 180°, 270°) + colour jitter (brightness/contrast ±0.2)

**3. Focal Loss with Class Weights**

Focal Loss = (1 - p_t)^γ × CrossEntropy

- γ = 2.0 — downweights easy, confident predictions; focuses on hard minority examples
- Class weights [1.0, 6.0, 4.0, 2.0, 2.0] — Grade 1 penalised most heavily as the hardest to detect
- Label smoothing = 0.05 — prevents overconfident predictions on repeatedly-seen minority images

**4. Mixup Augmentation**

Pairs of minority class images (Grades 1–4) are blended together during training using a Beta-distributed coefficient (α=0.3). This creates synthetic training examples and smooths decision boundaries between adjacent grades. Grade 0 is excluded to avoid creating misleading healthy-labelled composites.

### Hyperparameter Search

A lightweight 3-epoch proxy search was used to identify the best class weight configuration before committing to the full training run. Three configurations were evaluated using Quadratic Weighted Kappa (QWK) as the selection metric:

| Config            | Class Weights [0,1,2,3,4] | Result                                       |
| ----------------- | ------------------------- | -------------------------------------------- |
| Low Minority      | [1.0, 1.0, 1.0, 1.0, 1.0] | Baseline — underperforms on minority classes |
| High Minority     | [0.3, 5.0, 4.0, 6.0, 7.0] | Over-aggressive — hurts Grade 2 accuracy     |
| Balanced Weighted | [1.0, 6.0, 4.0, 2.0, 2.0] | Winner — best overall QWK                    |

The winning configuration was automatically selected and carried forward into the full training run.

### Training Configuration

| Hyperparameter           | Value            | Reason                                               |
| ------------------------ | ---------------- | ---------------------------------------------------- |
| Learning rate (head)     | 1e-4             | Standard fine-tuning rate                            |
| Learning rate (backbone) | 1e-5             | Slow adaptation to preserve pretrained features      |
| Batch size               | 16               | GPU memory limit for 384×384 images                  |
| Epochs                   | 25 (max)         | Early stopping prevents over-training                |
| Unfreeze epoch           | 4                | 3 warm-up epochs before backbone fine-tuning         |
| Dropout                  | 0.5              | Regularisation against minority class memorisation   |
| Weight decay             | 1e-4             | L2 regularisation via AdamW                          |
| Scheduler                | Cosine annealing | Smooth LR decay across training phases               |
| Optimiser                | AdamW            | Decoupled weight decay for consistent regularisation |

### Evaluation Metric
The primary metric is Quadratic Weighted Kappa (QWK), not accuracy. QWK is preferred because:

- It accounts for the ordinal nature of DR grades — predicting Grade 4 for a Grade 0 eye is penalised far more than predicting Grade 1
- It is robust to class imbalance — a model predicting only Grade 0 would score near zero despite high accuracy
- It is the standard benchmark metric used in the Kaggle DR grading competition and published literature

A QWK of 0 indicates performance no better than random chance. A score of 1.0 indicates perfect agreement.

**Test-Time Augmentation (TTA)** is applied during final evaluation — each test image is passed through the model in four orientations (original, horizontal flip, vertical flip, 90° rotation) and the probability distributions are averaged before the final prediction, improving prediction stability.

### Regularisation Techniques

| Technique            | Details                                            |
| -------------------- | -------------------------------------------------- |
| Dropout              | p = 0.5 in classification head                     |
| Weight decay         | 1e-4 via AdamW (L2 regularisation)                 |
| Label smoothing      | 0.05 — softens hard targets                        |
| Data augmentation    | Rotation, flip, colour jitter for minority classes |
| Mixup                | Blends minority class image pairs                  |
| Early stopping       | Patience of 7 epochs on test QWK                   |
| Best model saving    | Saves weights only when test QWK improves          |
| Progressive freezing | Protects pretrained backbone during warm-up        |

### Checkpointing & Resume Support

The training loop saves the complete training state after every epoch:

torch.save({
    'epoch',
    'model_state_dict',
    'optimizer_state_dict',
    'scheduler_state_dict',
    'scaler_state_dict',
    'test_kappa',
    'best_kappa',
    'backbone_unfrozen',
    'early_stopping_state',
}, CHECKPOINT_PATH)

If a checkpoint exists at startup, training resumes exactly where it left off — including the epoch count, best QWK, early stopping patience counter, and whether the backbone has already been unfrozen. This makes the pipeline robust to session interruptions on cloud compute platforms such as Kaggle.

A separate best model file is saved whenever test QWK improves, ensuring the final evaluation always uses peak-performance weights regardless of what happens in later epochs.


