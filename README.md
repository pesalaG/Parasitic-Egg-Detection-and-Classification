# Parasitic Egg Detection using RTMDet in MMDetection

This repository contains the code and resources for a machine learning model designed to detect parasitic eggs in images. The model is built using RTMDet within the MMDetection framework.

## Approach

### Problem Description

Intestinal parasitic infections remain among the leading causes of morbidity worldwide, especially in tropical and sub-tropical areas with high temperate climates. According to WHO, approximately 1.5 billion people, or 24% of the world's population, were infected with soil-transmitted helminth infections (STH), and more than 800 million children worldwide required preventive chemotherapy for STH in 2020. Most infections can cause diarrhea and other symptoms, such as malnutrition and anemia, particularly in children, who may suffer from growth failure. Diagnosis of intestinal parasites is usually based on direct examination in the laboratory. Unfortunately, this time-consuming method shows low sensitivity and requires an experienced and skilled medical laboratory technologist, being thus impractical for on-site use.

### Custom Configuration

To tackle the problem of parasitic egg detection, a custom configuration script for RTMDet has been prepared. This configuration script is designed to work with the specific dataset and requirements of the task. Key customizations include:

- Dataset root: The dataset is located at '/content/output_directory/5-fold/fold-0/'.
- Training batch size and workers: Adjusted for optimal performance.
- Learning rate: Customized learning rate schedule for training.
- Classes and color palette: The model is trained to detect specific parasitic egg among the 11 classes, each associated with a unique color in the output.

The custom configuration script is saved as 'config_parasitic' and is written to '/content/mmdetection/configs/rtmdet/rtmdet_tiny_1xb4-20e_.py'.

### Preprocessing

Before training the model, the dataset is preprocessed using "Sahi" slicing. This step is crucial for splitting the data and labels correctly, ensuring effective model training.

## Prerequisites

Before you begin working with this project, ensure that you have the following libraries and dependencies installed:

- opendatasets
- pandas
- openmim
- mmengine
- mmcv
- mmdet
- sahi

## Training

To train the model with the custom prepared configuration script, you can use the following command:

```bash
!python tools/train.py /content/mmdetection/configs/rtmdet/rtmdet_tiny_1xb4-20e_.py
