# Deepfake Detection via Image Analysis

## Project Overview
The goal of this project is to develop a deepfake detection system using image analysis techniques. By analyzing frames extracted from videos, the system aims to identify visual artifacts that indicate the presence of deepfakes. This project focuses on preprocessing, model development, and training using TensorFlow and Keras.

---

## Dataset
We used a dataset of deepfake video sequences, which includes both original and manipulated video clips. The dataset is stored in the following folders:
- `/content/DFD_original_sequences` (original sequences)
- `/content/DFD_manipulated_sequences` (manipulated sequences)

### Dataset Statistics
- **Training Set**: 4,800 images (2,400 original + 2,400 manipulated)
- **Validation Set**: 1,200 images (600 original + 600 manipulated)

---

## Preprocessing
Before splitting the dataset, we performed the following preprocessing steps:
1. **Image Augmentation**: Applied transformations like rotation, flipping, and zooming to improve generalization.
2. **Normalization**: Scaled pixel values to the range [0, 1].
3. **Resizing**: Resized images to match the input size required by the model (e.g., 224x224 for EfficientNet).

After preprocessing, the dataset was split into training and validation sets:
- **Training Set (80%)**: `/content/dataset/train/`
- **Validation Set (20%)**: `/content/dataset/val/`

---

## Model Development

### Initial Model (Baseline)
- **Architecture**: Simple CNN with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout` layers.
- **Results**:
  - Training Accuracy: ~77.02%
  - Validation Accuracy: ~76.33%
  - High loss, indicating poor generalization.
  - Performance in real-world applications was unsatisfactory.

### Second Model (EfficientNetB0)
- **Architecture**: Used `EfficientNetB0` as a pre-trained base model with a `GlobalAveragePooling2D` layer and a `Dense` layer for binary classification.
- **Results**:
  - Training Accuracy: ~51.26%
  - Validation Accuracy: ~50%
  - Poor performance, likely due to insufficient fine-tuning or overfitting.

### Current Model (Improved EfficientNetB0)
- **Architecture**:
  - Base Model: `EfficientNetB0` (pre-trained on ImageNet)
  - Added Layers: `GlobalAveragePooling2D`, `Dense`, and `Dropout`.
  - Optimizer: Adam with a learning rate of 0.001.
- **Results**:
  - Training Accuracy: ~73.38%
  - Validation Accuracy: ~76%
  - Lower validation loss, indicating better generalization.
- **Model Parameters**:
  - Total Parameters: 12.8M
  - Trainable Parameters: 4.2M
  - Non-Trainable Parameters: 209K
  - Optimizer Parameters: 8.4M

---

## Saved Model
The improved model was saved for future use:
- **Path**: `/content/drive/MyDrive/KaggleDatasets/improved_deepfake_model.keras`

---

## Next Steps
1. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizer settings.
2. **Advanced Augmentation**: Explore more advanced augmentation techniques to further improve generalization.
3. **Model Evaluation**: Test the model on unseen deepfake datasets to evaluate real-world performance.
4. **Deployment**: Develop a REST API or web interface for real-time deepfake detection.

---

## Acknowledgments
- Kaggle’s Deepfake Detection Challenge dataset
- TensorFlow and Keras for model development
- OpenCV for frame extraction and preprocessing
