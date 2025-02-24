# Multi-Class Waste Classification using MobileNetV2

## Overview
This project implements a Convolutional Neural Network (CNN) for multi-class waste classification using the MobileNetV2 model. The model is trained on an image dataset containing different categories of waste, and it aims to automate the classification process for efficient waste sorting and recycling.

## Features
- **Data Loading & Preprocessing**: Loads images from a dataset directory, resizes them to 224x224 pixels, and normalizes pixel values.
- **Label Encoding & Splitting**: Encodes class labels using `LabelEncoder` and splits data into training and validation sets.
- **Data Augmentation**: Applies random transformations to improve model generalization.
- **Pretrained Model (MobileNetV2)**: Uses MobileNetV2 with fine-tuning of the last 10 layers for better feature extraction.
- **Training with Class Weights**: Handles class imbalances by computing balanced class weights.
- **Model Evaluation & Prediction**: Evaluates model performance and allows single-image classification.

## Dataset Structure
Ensure the dataset is structured as follows:
```
dataset/
│── Cardboard/
│── Food Organics/
│── Glass/
│── Metal/
│── Miscellaneous Trash/
│── Paper/
│── Plastic/
│── Textile Trash/
│── Vegetation/
```
Each class folder contains images belonging to that category.

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install required dependencies:
   ```sh
   pip install numpy pandas pillow scikit-learn tensorflow matplotlib
   ```
3. Ensure your dataset is correctly placed inside the `dataset/` directory.

## Running the Model in Jupyter Lab
To use the model in Jupyter Lab:
1. Open Jupyter Lab:
   ```sh
   jupyter lab
   ```
2. Run the notebook cells sequentially to train and test the model.
3. Modify parameters as needed within the notebook.

## Training the Model
Run the script inside Jupyter Lab to train the model:
```python
# Execute in a Jupyter Notebook cell
dataset_path = "dataset/"
train_model(dataset_path)  # Modify the function name if needed
```
Training runs for 2 epochs by default but can be adjusted. The model will be saved as:
```
dataset/realwaste-main/cnn_multiclass_classification.keras
```

## Predicting an Image
To classify a single image, use:
```python
predict_image('path_to_image')
```
This function will:
- Load and preprocess the image
- Use the trained model for classification
- Display the image along with the predicted class and confidence score

## Model Performance
Training results (for 2 epochs):
- Accuracy: **68.56%**
- Validation Accuracy: **39.96%**
- Loss: **0.9078**
- Validation Loss: **4.8319**

## Possible Improvements
- Increase the number of training epochs for better accuracy.
- Implement callbacks such as Early Stopping and Learning Rate Scheduling.
- Tune hyperparameters like batch size and learning rate.
- Try other architectures such as EfficientNet or ResNet.

## Acknowledgments
This project leverages **MobileNetV2**, a lightweight and efficient deep learning model optimized for mobile and embedded vision applications.

1akin1
