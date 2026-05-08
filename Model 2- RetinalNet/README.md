> **Note:** GitHub often fails to render large Jupyter Notebooks. To view the full project with all plots and training results, please use the DagsHub link below:
>
> [![View on DagsHub](https://dagshub.com/static/badge.svg)](https://dagshub.com/Chooladeva/Diabetic-Retinopathy-Classification/src/main/Model%202-%20RetinalNet)

## Diabetic Retinopathy Grading — RetinalNet (Custom CNN)

### Overview

This module implements a purpose-built convolutional neural network called RetinalNet for automated diabetic retinopathy (DR) grading from retinal fundus photographs. 
Unlike transfer learning approaches that adapt pretrained models, RetinalNet is trained entirely from scratch — every parameter is shaped solely by retinal imaging data, making it a fully domain-specific classifier.Given a retinal photograph, the model classifies it into one of five severity grades:

- 0- No DR (Healthy)
- 1- Mild DR
- 2- Moderate DR
- 3- Severe DR
- 4- Proliferative DR

### Model Architecture

RetinalNet follows a hierarchical feature extraction design: a stem block for initial processing, four residual stages with progressive channel doubling, global average pooling, and a two-layer classification head.

Input (384×384 RGB)
        ↓
   Stem Block          7×7 conv → BN → ReLU → MaxPool
   384×384 → 96×96    64 channels
        ↓
   Stage 1             2× ResBlock (64ch, no SE)
   96×96               No spatial reduction
        ↓
   Stage 2             2× ResBlock + SE Attention (128ch)
   96×96 → 48×48       Spatial downsampling
        ↓
   Stage 3             3× ResBlock + SE Attention (256ch)
   48×48 → 24×24       Spatial downsampling
        ↓
   Stage 4             2× ResBlock + SE Attention (512ch)
   24×24 → 12×12       Spatial downsampling
        ↓
Global Average Pool    12×12 → 512-dimensional vector
        ↓
   Classifier Head
   Dropout(0.5) → Linear(512→256) → BN → ReLU → Dropout(0.3) → Linear(256→5)
        ↓
   5 DR Grade Logits

Total Parameters: ~11 million (all trainable from epoch 1)

### Key Architectural Components

- #### Residual Blocks (ResBlock)

The core building unit of RetinalNet. Each block contains two 3×3 convolutions with a skip connection that adds the block's input directly to its output:

Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+Input) → ReLU

Skip connections solve the vanishing gradient problem — gradients can flow directly through the skip path without passing through convolution layers, allowing the network to train stably despite its depth. When spatial dimensions or channel counts change, a 1×1 projection convolution aligns the skip connection dimensions.

- #### Squeeze-and-Excitation Blocks (SEBlock)

Applied in Stages 2, 3, and 4. SE blocks add channel-wise attention — they learn which feature maps are most important for a given input and rescale them accordingly:

Feature Map → Global Avg Pool → FC(compress) → ReLU → FC(expand) → Sigmoid → Scale

For retinal images this is particularly valuable. Lesion signals such as haemorrhages, hard exudates, and neovascularisation appear in specific colour channels. The SE block gives the model a mechanism to actively upweight those channels and suppress background noise, without any manual feature engineering.

Stage 1 does not use SE attention — early-stage features (basic edges and textures) are not yet semantically rich enough to benefit from channel re-weighting.

- #### Classification Head

RetinalNet uses a two-layer MLP head rather than a single linear layer:

Flatten → Dropout(0.5) → Linear(512→256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256→5)

The two dropout layers at different strengths (0.5 then 0.3) provide aggressive regularisation at two levels of abstraction. The BatchNorm between the linear layers stabilises the intermediate 256-dimensional representation, which is especially important when training on a heavily imbalanced dataset where minority class examples are seen repeatedly.

- #### Kaiming Weight Initialisation

All convolutional and linear layers are initialised using Kaiming Normal initialisation, specifically designed for ReLU networks. This keeps activation variance stable across layers from the very first forward pass, preventing gradient explosion or vanishing before any learning occurs.

### Dataset & Class Imbalance Handling

The dataset contains 28,100 training images with severe class imbalance:

| Grade             | Train Count | Train % |
| ----------------- | ----------: | ------: |
| 0 — Healthy       |      20,652 |   73.5% |
| 1 — Mild          |       1,940 |    6.9% |
| 2 — Moderate      |       4,244 |   15.1% |
| 3 — Severe        |         697 |    2.5% |
| 4 — Proliferative |         567 |    2.0% |

Four mechanisms are stacked to address this imbalance:

**1. Weighted Random Sampler**

Each image is assigned a sampling weight equal to 1 / class_count. Grade 4 images are approximately 36× more likely to be selected per batch than Grade 0 images, ensuring balanced class representation throughout training.

**2. Class-Aware Data Augmentation**

The augmentation strategy is intentionally asymmetric:

- Grade 0: Random horizontal flip only — healthy eyes are already abundant
- Grades 1–4: Horizontal flip + random rotation (0°, 90°, 120°, 180°, 270°) + colour jitter (brightness/contrast ±0.2)

More aggressive augmentation for minority classes creates variety from a limited pool of images, reducing the risk of memorisation.

**3. Focal Loss with Class Weights**

Focal Loss = (1 - p_t)^γ × CrossEntropy

- γ = 2.0 — easy, well-classified examples contribute almost nothing to the loss; hard minority examples retain full loss
- Class weights [1.0, 6.0, 4.0, 2.0, 2.0] — Grade 1 (Mild) is penalised most heavily as it is the most visually similar to Grade 0 and the most commonly missed
- Label smoothing = 0.05 — softens hard one-hot targets to prevent overconfident predictions on repeatedly-seen minority images

**4. Mixup Augmentation**

Pairs of minority class images (Grades 1–4) are blended together at the batch level using a Beta-distributed coefficient (α = 0.3):

new_image = λ × image_A + (1-λ) × image_B

The blending coefficient is always kept above 0.5 so the dominant image contributes more than the secondary one, and the label stays as the dominant image's grade. Grade 0 is completely excluded from mixing to avoid creating misleading composites.

### Hyperparameter Configuration

RetinalNet reuses the BalancedWeighted class weight configuration identified by the EfficientNet-B4 hyperparameter search in Model 1. Since both models face identical class imbalance, the winning configuration transfers directly — no separate search was run.

| Config                   | Class Weights [0,1,2,3,4] |
| ------------------------ | ------------------------- |
| Balanced Weighted (used) | [1.0, 6.0, 4.0, 2.0, 2.0] |

### Training Configuration

All hyperparameters differ from EfficientNet-B4 because RetinalNet trains from random weights rather than pretrained ones:

| Hyperparameter | Value     | Reason                                                              |
| -------------- | --------- | ------------------------------------------------------------------- |
| Learning rate  | 1e-3      | Random weights need larger steps than pretrained weights            |
| Batch size     | 32        | Larger batches give more stable gradients for from-scratch training |
| Epochs         | 35 (max)  | More time needed — no pretrained features to build on               |
| Warmup epochs  | 5         | Stabilises random initialisation before full LR is applied          |
| Dropout (head) | 0.5 + 0.3 | Two-stage regularisation in classifier                              |
| Weight decay   | 1e-4      | L2 regularisation via AdamW                                         |
| Optimiser      | AdamW     | Decoupled weight decay for consistent regularisation                |
| Focal gamma    | 2.0       | Standard value for imbalanced classification                        |

No backbone freezing phase — since all weights start randomly, there is nothing to protect. Every layer trains from epoch 1.

### Learning Rate Schedule

A custom two-phase schedule is used, combining linear warmup and cosine decay:

Epochs 1–5  (Warmup):  LR ramps from 0.2 × 1e-3  →  1e-3
             (20% → 40% → 60% → 80% → 100% of target LR)

Epochs 6–35 (Cosine):  LR decays smoothly from 1e-3  →  ~0

Why warmup? In epoch 1, weights are random and BatchNorm layers have no meaningful running statistics. Applying the full learning rate immediately causes large, noisy gradient updates that can push weights into bad regions. The 5-epoch warmup lets the network stabilise gradually before full training begins.
EfficientNet-B4 did not need this because its pretrained weights already start in a good region.

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs          # Linear ramp

    progress = (
        (epoch - warmup_epochs)
        / (total_epochs - warmup_epochs)
    )

    return 0.5 * (1 + np.cos(np.pi * progress))    # Cosine decay
```

### Evaluation Metric

The primary metric is Quadratic Weighted Kappa (QWK), not accuracy. QWK is preferred because:

- It accounts for the ordinal nature of DR grades — predicting Grade 4 for a Grade 0 eye is penalised far more than predicting Grade 1
- It is robust to class imbalance — a model predicting only Grade 0 would score near zero despite high accuracy
- It is the standard benchmark metric in DR grading research and the Kaggle DR competition

A QWK of 0 indicates no better than random. A score of 1.0 is perfect agreement.

Note: **TTA (Test-Time Augmentation)** was tested on RetinalNet but decreased performance. This is because RetinalNet — trained from scratch on a smaller domain-specific dataset — produces inconsistent probability estimates across augmented views. Averaging inconsistent outputs degrades rather than improves predictions. Standard single-pass inference is used for final evaluation.

### Regularisation Techniques

| Technique              | Details                                            |
| ---------------------- | -------------------------------------------------- |
| Two-stage dropout      | p = 0.5 then p = 0.3 in classifier head            |
| BatchNorm in head      | Stabilises intermediate 256-d features             |
| Weight decay           | 1e-4 via AdamW                                     |
| Label smoothing        | 0.05 — prevents overconfident predictions          |
| Data augmentation      | Rotation, flip, colour jitter for minority classes |
| Mixup                  | Blends minority class image pairs at batch level   |
| Early stopping         | Patience of 7 epochs on test QWK                   |
| Best model saving      | Saves weights only when test QWK improves          |
| Kaiming initialisation | Stable starting point for ReLU networks            |
| Linear warmup          | Prevents destructive early-epoch updates           |

### Checkpointing & Resume Support

The complete training state is saved to disk after every epoch:

```python
torch.save(
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'test_kappa': test_kappa,
        'best_kappa': best_kappa,
        'early_stopping_state': early_stopping_state,
    },
    CHECKPOINT_PATH
)
```

If a checkpoint exists at startup, training resumes from exactly where it left off — including epoch count, best QWK, and early stopping patience counter. A separate best model file is saved whenever test QWK improves, ensuring final evaluation always uses peak-performance weights.
