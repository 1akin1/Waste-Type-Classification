# Waste Type Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify different types of waste materials based on images. The model is trained on a dataset of waste images and can predict the category of a given waste item with high accuracy.

## Features
- Image preprocessing and dataset handling
- Convolutional Neural Network (CNN) architecture for classification
- Model training and evaluation with accuracy and loss visualization
- Prediction function to classify a single image
- Model saving and loading functionality

## Dataset
This project uses the [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste) from the UCI Machine Learning Repository. Ensure you download and place the dataset inside the `dataset/` directory.

### Dataset Information
RealWaste is a real-world image classification dataset containing waste items captured within an authentic landfill environment. The dataset consists of **4,752 images** categorized into **9 major material types**:

- **Cardboard**: 461 images
- **Food Organics**: 411 images
- **Glass**: 420 images
- **Metal**: 790 images
- **Miscellaneous Trash**: 495 images
- **Paper**: 500 images
- **Plastic**: 921 images
- **Textile Trash**: 318 images
- **Vegetation**: 436 images

The dataset was created as part of an honors thesis exploring the performance of convolutional neural networks (CNNs) on real waste materials compared to idealized waste object datasets. The images are provided in **524x524 resolution**.

For more details, refer to the paper: *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning* by Sam Single, Saeid Iranmanesh, and Raad Raad (2023).

## Installation
### Prerequisites
Ensure you have Python 3 installed along with the following dependencies:

```bash
pip install numpy pandas pillow scikit-learn tensorflow matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/1akin1/Waste-Type-Classification.git
cd Waste-Type-Classification
```

## Usage
### Dataset Preparation
1. Download the dataset from [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste).
2. Extract and place the images inside the `dataset/` directory.
3. Ensure that images are organized in subdirectories based on their waste categories.

### Training the Model
Run the script to train the model:
```bash
python train.py
```
The trained model will be saved in the `saved_model/` directory.

### Predict a Single Image
Use the `predict_image` function in `train.py` to classify an individual image:
```python
predict_image("test_images/sample.jpg")
```

## Dataset
```
├── dataset/                 # Dataset directory
│   ├── cardboard/           # Example class folder
│   ├── food_organics/       # Example class folder
│   ├── glass/               # Example class folder
│   ├── metal/               # Example class folder
│   ├── miscellaneous_trash/ # Example class folder
│   ├── paper/               # Example class folder
│   ├── plastic/             # Example class folder
│   ├── textile_trash/       # Example class folder
│   ├── vegetation/          # Example class folder

## Model Architecture
The CNN model consists of:
- 3 convolutional layers with ReLU activation
- Max pooling layers
- Fully connected (Dense) layer with Dropout
- Softmax output layer for multi-class classification

## Results
The model is trained for 20 epochs with a batch size of 32, achieving high accuracy on the validation set. The training and validation accuracy/loss graphs are plotted for analysis.

## Contribution
Feel free to contribute by submitting issues or pull requests.

---
Developed by 1akin1

