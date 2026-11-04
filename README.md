# Diffusion-Based-Data-Augmentation-for-Pothole-Detection-Using-YOLO11n
A computer vision project that combines YOLO11n and DDPM diffusion models for pothole detection. It generates synthetic pothole images, selects useful samples, and uses them to augment real road images. The project evaluates the impact of synthetic data augmentation on detection performance using the RDD2022 India dataset.

# Diffusion-Based Data Augmentation for Pothole Detection Using YOLO11n

## Overview

This project combines **Deep Learning, Computer Vision, and Diffusion Models** to improve pothole detection through synthetic data augmentation.

It uses a DDPM (Denoising Diffusion Probabilistic Model) to generate synthetic pothole images and YOLO11n to detect potholes in road images. Generated samples are ranked using a detector-based usefulness score and selected samples are blended with real road images to create an augmented training dataset.

The performance of the baseline YOLO11n model is compared with the diffusion-augmented YOLO11n model.

## Objectives

- Detect potholes in road images using YOLO11n.
- Generate synthetic pothole samples using a DDPM.
- Evaluate synthetic samples using a usefulness score.
- Augment real training images with selected synthetic samples.
- Compare detection performance before and after augmentation.

## Technologies Used

- Python
- PyTorch
- YOLO11n (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Diffusion Models (DDPM)

## Dataset

**RDD2022 – India Dataset**

The project focuses on the D40 pothole class and prepares a pothole-only dataset for training and evaluation.

Dataset structure:

```text
RDD2022/
└── India/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

The dataset must be downloaded separately.

## Project Workflow

1. **Dataset Preparation:** Extract pothole images and annotations from RDD2022.
2. **Baseline Training:** Train YOLO11n using real pothole images.
3. **Crop Extraction:** Extract pothole regions from annotated images.
4. **Diffusion Training:** Train a small U-Net-based DDPM on pothole crops.
5. **Synthetic Generation:** Generate new pothole image patches.
6. **Usefulness Selection:** Rank generated samples using the baseline detector's confidence-based score.
7. **Dataset Augmentation:** Blend selected synthetic patches into real road images using OpenCV seamless cloning.
8. **Augmented Training:** Train another YOLO11n model using the augmented dataset.
9. **Evaluation:** Compare both models using Precision, Recall, mAP50, and mAP50-95.

## Installation

Clone the repository:

```bash
git clone https://github.com/sreekartulluri/Diffusion-Based-Data-Augmentation-for-Pothole-Detection-Using-YOLO11n.git
```

Navigate to the project:

```bash
cd Diffusion-Based-Data-Augmentation-for-Pothole-Detection-Using-YOLO11n
```

Install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib pyyaml
```

## Execution

Run the main Python script:

```bash
python road_damage.py
```

Make sure the dataset path in the script points to your RDD2022 India dataset location.

## Evaluation Metrics

The project evaluates:

| Metric | Purpose |
|---|---|
| Precision | Measures correct pothole predictions |
| Recall | Measures detected potholes |
| mAP50 | Detection accuracy at IoU 0.50 |
| mAP50-95 | Detection performance across IoU thresholds |

## Expected Outputs

The project generates:

- Baseline YOLO11n model
- Diffusion model weights
- Synthetic pothole images
- Selected synthetic samples
- Augmented training images
- Detection comparison visualizations
- Training loss and performance graphs
- Metrics CSV and summary report

## Applications

- Road condition monitoring
- Automated pothole detection
- Smart transportation systems
- Road maintenance and infrastructure monitoring

## Future Improvements

- Improve synthetic image quality.
- Experiment with larger diffusion architectures.
- Increase dataset diversity.
- Explore real-time pothole detection.
- Evaluate performance on additional road datasets.

---
<img width="3550" height="1774" alt="6_augmented_examples" src="https://github.com/user-attachments/assets/237412de-0c51-4edc-b8a4-983a769ff4c0" />



<img width="2675" height="4518" alt="7_detection_comparison" src="https://github.com/user-attachments/assets/156af2b1-a050-41cb-b550-525a4bfbdf79" />


<img width="3181" height="1284" alt="1_forward_diffusion" src="https://github.com/user-attachments/assets/813da4a7-1da8-4966-80d2-57c5b95c2948" />

<img width="2780" height="1670" alt="3_reverse_diffusion" src="https://github.com/user-attachments/assets/def2f32e-0fed-461a-ba21-77705eec2e86" />

<img width="1369" height="819" alt="4_real_vs_generated" src="https://github.com/user-attachments/assets/26352038-68fb-458d-a150-e2dc1c8d00e3" />

<img width="3180" height="1009" alt="5_selected_samples" src="https://github.com/user-attachments/assets/e34c0028-67d3-4718-a460-94033ec12464" />

<img width="1579" height="878" alt="2_diffusion_loss" src="https://github.com/user-attachments/assets/a1e88549-2fb1-43ff-9520-20bc9a9b3bfc" />

<img width="2581" height="878" alt="9_yolo_training_curves" src="https://github.com/user-attachments/assets/f38c1357-72f7-43f2-bd9a-4810a2d67224" />

<img width="1979" height="1078" alt="8_metrics_comparison" src="https://github.com/user-attachments/assets/c228d8ad-c15a-4423-a374-0d9a99862ae9" />

-




















*This project explores the use of diffusion-based synthetic data augmentation to support object detection in road infrastructure monitoring.*
