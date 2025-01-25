# pneumonia-detection-using-Vit
A binary classification project that uses a Vision Transformer to detect pneumonia from RSNA chest X-rays. It preprocesses images and masks, fine-tunes a pretrained ViT model with PyTorch, trains over five epochs, and saves the optimized model for deployment.
# Pneumonia Detection with Vision Transformer

This project is focused on building a deep learning pipeline for detecting pneumonia in chest X-ray images using the RSNA Pneumonia Detection Challenge dataset. It employs the Vision Transformer (ViT) model, a cutting-edge architecture, to classify images into two categories: "Normal" or "Lung Opacity" (indicative of pneumonia). The workflow includes data preprocessing, model fine-tuning, and training, and saves the trained model for future use.

---

## Project Overview
This repository contains:
1. **Dataset Preprocessing**:
   - Loads metadata, images, and segmentation masks.
   - Filters for relevant classes ("Lung Opacity" and "Normal") and encodes target labels as binary values.
   - Applies transformations to resize images and normalize them for input into the Vision Transformer model.

2. **Model**:
   - Fine-tunes a pretrained Vision Transformer (`google/vit-base-patch16-224`) from Hugging Face.
   - Configures the model for binary classification with two output labels and cross-entropy loss.

3. **Training Process**:
   - Trains the model for five epochs using the AdamW optimizer and tracks training loss and accuracy.
   - Leverages PyTorch’s `DataLoader` and `Dataset` utilities for efficient batch loading.

4. **Output**:
   - Saves the fine-tuned model for deployment and further evaluation.

---

## Features
- **Custom Dataset Class**:
  - Handles loading of images and segmentation masks.
  - Applies transformations for image resizing and normalization.

- **Vision Transformer (ViT)**:
  - A state-of-the-art deep learning model designed for image classification tasks.
  - Fine-tuned for pneumonia detection with binary classification.

- **Training Metrics**:
  - Tracks loss and accuracy after each epoch for monitoring model performance.

---

## Requirements
### Software and Libraries
- **Python 3.7+**
- Libraries used in the project:
  - `torch`
  - `torchvision`
  - `transformers`
  - `Pillow`
  - `pandas`
  - `matplotlib`
  - `tqdm`

### Dataset
The project uses the **RSNA Pneumonia Detection Challenge dataset** from Kaggle, which includes:
- Chest X-ray images.
- Corresponding segmentation masks for pneumonia regions.
- Metadata file with patient IDs and associated class labels.

Dataset structure:

---

## File Structure
The main script is contained in the Jupyter Notebook file:
- `dataset2 (1).ipynb`: Contains the full workflow for preprocessing, model training, and saving the trained Vision Transformer.

Key components:
- **Images Directory**: Directory containing chest X-ray images.
- **Masks Directory**: Directory containing segmentation masks (optional).
- **Metadata File**: CSV file containing patient IDs and class labels (`Normal` or `Lung Opacity`).

---

## Workflow
1. **Setup and Preprocessing**:
   - The notebook first loads the metadata CSV and filters it for relevant classes (`Normal` and `Lung Opacity`).
   - Prepares custom transformations using `torchvision.transforms` for both images and segmentation masks.
   - Creates a PyTorch `Dataset` class to handle image and mask loading, with support for transformations.

2. **Model Fine-tuning**:
   - Loads a pretrained Vision Transformer (`google/vit-base-patch16-224`) using Hugging Face's `transformers` library.
   - Configures the model for binary classification, setting the number of labels to 2.

3. **Training**:
   - Uses the AdamW optimizer and CrossEntropyLoss for training.
   - Trains the model over five epochs, tracking loss and accuracy metrics for each epoch.

4. **Output**:
   - Saves the trained model for deployment or evaluation.

---

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection-vit.git
   cd pneumonia-detection-vit
## Acknowledgments
- Dataset: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- Model: Hugging Face Vision Transformer (`google/vit-base-patch16-224`)

## License
This project is open-source and available under the [MIT License](LICENSE).
