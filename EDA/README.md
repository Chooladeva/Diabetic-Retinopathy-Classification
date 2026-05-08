## Diabetic Retinopathy Grading — Exploratory Data Analysis & Preprocessing

### Overview

Before any model was built, a thorough Exploratory Data Analysis (EDA) was conducted to understand the structure, quality, and challenges of the dataset. The findings directly shaped the preprocessing pipeline and model design decisions that followed. This document summarises the key observations from the EDA and the preprocessing steps applied to prepare the data for training.Dataset DescriptionThe dataset consists of high-resolution retinal fundus photographs collected under varying imaging conditions. Each patient contributes two images — one for the left eye and one for the right eye — with filenames in the format {patient_id}_left.jpeg and {patient_id}_right.jpeg. Each image has been clinically graded on a five-level diabetic retinopathy severity scale:

- 0- No DR (Healthy)
- 1- Mild DR
- 2- Moderate DR
- 3- Severe DR
- 4- Proliferative DR

### Exploratory Data Analysis

**I. Class Distribution Analysis**

The first and most critical finding from the EDA is the severe class imbalance in the dataset. The bar chart analysis revealed that Grade 0 (healthy) accounts for the overwhelming majority of images, while the clinically important severe and proliferative grades are represented by far fewer samples.



This imbalance is not just a statistical concern — it has direct clinical implications. A model that ignores the imbalance would learn to predict healthy for most images and achieve superficially high accuracy while failing to detect the disease cases that matter most. This finding motivated the use of weighted sampling, Focal Loss, and Mixup augmentation in both models.

**II. Symmetry Analysis — Left vs. Right Eye**

Since each patient contributes both a left and right eye image, it was important to understand how consistent DR severity grades are between the two eyes. This has two implications — clinical and methodological.

Key findings:

- 87.25% of patients have the same DR grade in both eyes, indicating that the disease typically progresses similarly in both eyes
- Grade 0 symmetry: ~94% — if one eye is healthy, the other almost certainly is too
- Grade 3 and 4 symmetry: ~75% — advanced disease tends to affect both eyes, though some asymmetry exists
- Grade 1 symmetry: ~50% — the lowest symmetry, with many cases where one eye is mild while the other is still healthy (36% of Grade 1 cases)

The low symmetry in Grade 1 is clinically meaningful — it suggests that mild DR often represents the earliest observable stage of the disease, where one eye has just begun to show signs while the other has not yet progressed. This also explains why Grade 1 is the hardest grade to classify reliably.

Methodological implication: The high inter-eye correlation (87.25%) meant that naive random splitting of images into train and test sets would cause data leakage — a patient's left eye image could appear in training while their right eye appears in testing. This was addressed through patient-level data splitting, described in the preprocessing section below.

**III. Brightness Distribution by Class**

The brightness analysis examined whether DR severity correlates with image brightness — which would suggest that disease progression affects how images are captured or appear visually.

Key findings:

- Brightness levels are largely consistent across all five DR grades — the central 50% of brightness values overlap significantly between classes
- Grade 0 shows the most outliers — some healthy eye images appear excessively bright or washed out, likely due to imaging artifacts
- There is wide within-class brightness variation across all grades, reflecting the diverse imaging conditions under which the dataset was collected

The consistent brightness across grades confirms that the model cannot rely on overall image brightness as a shortcut for grading — it must learn actual pathological features. The wide brightness variation motivated the inclusion of colour jitter augmentation (±0.2 brightness and contrast) to make both models robust to these real-world imaging inconsistencies.

**IV. Image Resolution and Shape Analysis**

Understanding the resolution and shape characteristics of the raw images was essential for designing the preprocessing pipeline.

Key findings:

- No images are perfectly square — all images are rectangular with a landscape (wider than tall) orientation
- Average aspect ratio: ~1.46 — images are noticeably wider than tall on average
- Large resolution variation — some images are as small as 400×300 pixels while others exceed 5,000 pixels in width
- Small images may lack the fine detail needed to identify subtle lesions (microaneurysms, small haemorrhages)
- Very large images carry more detail but are computationally expensive to process

These findings made it clear that a structured normalisation pipeline was essential — directly resizing the raw rectangular images to a square input format would introduce distortion, and the large resolution variation would make lesion sizes inconsistent between images even within the same DR grade.

### Data Preprocessing

The EDA findings motivated a five-stage preprocessing pipeline designed to standardise image quality, size, and retinal scale before any model training.

**1. Image Preprocessing Pipeline**

- **Stage I — Automated Cropping and Noise Removal**

Many raw images contained large black borders around the retinal region, along with edge artifacts that carry no clinical information. An automated cropping step detected the central retinal region and removed these borders, ensuring the model focuses exclusively on relevant content from the first pixel.

- **Stage II — Standardised Radius Scaling**

Because images were captured at varying distances, the apparent size of the retina varies significantly between images — a lesion that looks large in one image may look tiny in another despite representing the same clinical severity. To fix this, a radius-based scaling method was applied:

- The radius of the retina was estimated from the middle row of pixels in each image
- All images were resized so that the retinal radius was standardised to 384 pixels

This ensures that anatomical structures (the optic disc, blood vessels, lesion regions) appear at a consistent scale across all images, regardless of the original imaging distance.

**Stage III — Square Padding**

After radius scaling, images were placed onto a square canvas with a neutral grey (128) background. This step preserves the original proportions of the retina — avoiding the distortion that direct rectangular-to-square resizing would introduce — while producing a uniform square input size ready for the neural network.

**Stage IV — Ben Graham Contrast Enhancement**

The dataset included images with varying visual quality — haze, uneven illumination, and colour inconsistencies that make subtle features like microaneurysms and small haemorrhages difficult to see. Ben Graham's contrast enhancement method was applied:

Enhanced = Image - GaussianBlur(Image) + 128

This subtracts a blurred version of the image from the original, which effectively removes background illumination gradients and makes fine structures such as blood vessels, lesions, and exudates stand out more clearly against the retinal background.

- **Stage V — Circular Masking**

After contrast enhancement, some edge artifacts became more pronounced at the image borders. A circular mask was applied to retain only the central retinal region while filling the outer areas with a neutral grey background. This ensures the model attends exclusively to clinically relevant regions and is not confused by processing artifacts at the image edges.

Preprocessing progression:

Raw Image
    ↓
Automated Cropping (remove black borders)
    ↓
Radius Scaling (standardise retinal size to 384px radius)
    ↓
Square Padding (grey canvas, preserve aspect ratio)
    ↓
Ben Graham Contrast Enhancement (improve lesion visibility)
    ↓
Circular Masking (focus on retinal region only)
    ↓
Preprocessed Image (ready for model training)

**Data Splitting**

The symmetry analysis finding — that 87.25% of patients have matching grades in both eyes — made conventional random image splitting inappropriate. If both eyes from the same patient were split across training and test sets, the model would effectively be evaluated on patients it had already seen, producing artificially optimistic performance estimates.

A patient-level splitting strategy was applied instead:

- Both images from the same patient are always assigned to the same split (either both in training or both in testing)
- This completely eliminates patient overlap between training and test sets
- The dataset was split 80% training / 20% testing

Final split distribution:

| Grade     | Description     | Train Count | Test Count |
| --------- | --------------- | ----------: | ---------: |
| 0         | No DR (Healthy) |      20,652 |      5,158 |
| 1         | Mild            |       1,940 |        503 |
| 2         | Moderate        |       4,244 |      1,048 |
| 3         | Severe          |         697 |        176 |
| 4         | Proliferative   |         567 |        141 |
| **Total** |                 |  **28,100** |  **7,026** |

**Data Augmentation and Normalisation**

To further improve model generalisation and address the class imbalance identified in the EDA, a targeted augmentation strategy was applied dynamically during training (on-the-fly) rather than as a fixed offline preprocessing step. This means the model sees a different variation of each image on every epoch, effectively expanding the training set without storing additional data.

Augmentation strategy by class:

| Grade          | Augmentations Applied                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| 0 (Healthy)    | Random horizontal flip only                                                                              |
| 1–4 (Diseased) | Horizontal flip + Random rotation (0°, 90°, 120°, 180°, 270°) + Colour jitter (±0.2 brightness/contrast) |

Minority classes receive more aggressive augmentation because they have fewer unique training images and are at higher risk of being memorised by the model rather than generalised from.
After augmentation, all images were:

- Converted to PyTorch tensors — pixel values scaled from [0, 255] to [0, 1]
- Normalised using ImageNet statistics — mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] per channel

ImageNet normalisation is applied even to RetinalNet (trained from scratch) because the statistics provide a reasonable and consistent input distribution given that retinal images are captured under visible-light conditions similar to natural photographs.


