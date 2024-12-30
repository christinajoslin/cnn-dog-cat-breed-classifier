# Dog and Cat Breed Image Classifier Using CNN 

This project implements a Convolutional Neural Network (CNN) for classifying images of 104 dog and cat breeds using a dataset containing **29,000 images**. Leveraging transfer learning with the MobileNetV2 architecture, the model achieves an **F1 score of 84%** on the test set. The project includes data preprocessing, model training, evaluation, and saving the trained model. Training the model on this dataset takes approximately **3 hours using a v2-8 TPU**.

## Why MobileNetV2? 
MobileNetV2 was selected for its lightweight architecture and high efficiency, making it suitable for tasks with large datasets and limited computational resources. Despite its small size, MobileNetV2 consistently delivers high accuracy, as demonstrated by the 84% F1 score on the test set in this project. Additionally, its pre-trained weights on ImageNet significantly reduce training time, enabling the model to converge within approximately **3 hours using a v2-8 TPU.** 

## Features 
- **Transfer Learning:** MobileNetV2 pre-trained on ImageNet as the base model.
- **Custom Layers:** Fully connected layers with Dropout and L2 regularization.
- **Dataset Processing:** Training, validation, and test sets for performance monitoring.
- **Evaluation:** Comprehensive classification report including precision, recall, and F1 score for all 104 breeds.

## Instructions 
1. Ensure you have the required libraries installed (tensorflow, numpy, pandas, matplotlib, sciki-learn).
2. Download and extract the dataset form Kaggle: {Dogs and Cats Classifier Dataset}(https://www.kaggle.com/datasets/rajarshi2712/dogs-and-cats-classifier).
3. Run the code sequentially:
    - **Data Loading and Preprocessing:** Resize images to 224x224 pixels and normalize pixel values to the range [0, 1].
    - **Model Training:** Train the CNN using the training and validation sets. Early stopping ensures optimal performance.
    - **Evaluation:** Evaluate the model on the test set to calculate accuracy, F1 score, and other metrics.
4. Save the trained model (optional)

## Dependencies 
- **TensorFlow:** For building and training the CNN.
- **NumPy & Pandas:** For data manipulation and analysis.
- **Matplotlib:** For visualizing training results.
- **Scikit-learn:** For generating classification reports.

## Notes
- Training takes approximately 3 hours on a v2-8 TPU. **When running this notebook please ensure that you are using at least a v2-8 TPU at a minimum if not a A100 GPU or v5e-1 TPU)**. 
- Early stopping is used to prevent overfitting by monitoring validation accuracy.

## Author 
Christina Joslin 

## Acknowledgements 
- Dataset provided by {Dogs and Cats Classifier Dataset}(https://www.kaggle.com/datasets/rajarshi2712/dogs-and-cats-classifier)
- Thanks to Tensorflow and Scikit-learn teams for their open-source contributions



