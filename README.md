# Vegetable-Classification-CNN-Model

This application demonstrates how deep learning can be applied to classify different types of vegetables using a fine-tuned VGG16 Convolutional Neural Network (CNN).
Users can upload an image, and the trained model predicts the vegetable category with high accuracy.

The app uses a transfer-learning–based CNN model along with preprocessing utilities to convert images into model-ready tensors.

![CNN](/veggie-1.png?raw=true "Optional Title")

![CNN](/veggie-2.png?raw=true "Optional Title")


## Dataset

The dataset used for training is sourced from Kaggle:
misrakahmed / Vegetable Image Dataset

It contains labeled vegetable images across multiple categories.
The model achieves 99.5% accuracy on unseen test data, demonstrating exceptional generalization.

## Model Details

- Base Model: VGG16 (pretrained on ImageNet)
Used as the feature extractor with pre-trained convolutional layers.

 - Fine-Tuning:
Top convolutional blocks unfrozen and retrained on the vegetable dataset.

 - Fully Connected Layers:
Custom dense layers added for classification of vegetable types.

 - Softmax Output Layer:
Predicts the probability distribution across vegetable classes.

This combination of transfer learning and fine-tuning enables the model to achieve high accuracy even with limited data.

## treamlit App

Upload an image of a vegetable, and the app will process it and return the predicted class.

## Run the App
streamlit run app.py


If you want, I can also write a joint README containing both models, or format this for GitHub with sections like Installation, Folder Structure, and Sample Output.
