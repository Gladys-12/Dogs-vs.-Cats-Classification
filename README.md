# Dog vs Cat Convolutional Neural Network Classifier

This project is a binary image classifier that differentiates between dogs and cats using a Convolutional Neural Network (CNN). The model achieves high accuracy on validation data by leveraging data augmentation techniques, a carefully designed architecture, and fine-tuned parameters.

---

## Dependencies

Ensure the following dependencies are installed in your environment:

- Jupyter Notebook
- TensorFlow 1.10
- Python 3.6
- Matplotlib
- Seaborn
- Scikit-Learn
- Pandas
- NumPy

---

## Network Architecture

The classifier is built using the following key components:

- **Convolution Layers:** Extract features from images.
- **Pooling Layers:** Reduce spatial dimensions to prevent overfitting.
- **Flattening Layer:** Converts feature maps into a single vector.
- **Dense Layers:** Fully connected layers for classification.

### Key Parameters:

- **Activation Functions:** ReLU for hidden layers and Sigmoid for the output layer.
- **Optimizer:** Adam optimizer with learning rate scheduling.
- **Loss Function:** Binary Crossentropy.
- **Metric:** Accuracy.

---

## Data Augmentation

To enhance generalization and improve performance, the following data augmentation techniques are applied:

- Shearing
- Random Zoom
- Horizontal Flipping
- Rescaling Pixel Values

---

## Dataset

- **Training Set:** 19,998 images belonging to two classes (cats and dogs).
- **Validation Set:** 5,000 images for performance evaluation.
- **Test Set:** 12,500 unseen images for prediction and analysis.

---

## Training

The model is trained using an iterative approach with multiple epochs. Fine-tuning is performed by reducing the learning rate for enhanced accuracy. The training and validation sets are utilized to monitor model performance during training.

### Results:

- **Training Accuracy:** 99.96%
- **Validation Accuracy:** 97.56%
- **Validation Loss:** 0.1027

---

## Visualization

- **Confusion Matrix:** Provides insight into the model’s performance on the validation set.
- **Feature Maps:** Visualized output of individual filters in the convolution layers to understand feature extraction.
- **Misclassifications:** Examines incorrectly classified images to refine the model.

---

## Prediction

Single image predictions are supported, with output probabilities for each class. The classifier can predict on unseen datasets to evaluate generalizability.

---

## Performance on Unseen Data

The model demonstrates strong performance on unseen test images with predictions accompanied by probabilities for both classes. Examples of correct and misclassified images are visualized for qualitative assessment.

---

## Conclusion

The classifier achieves **97.56% accuracy** on the validation set, demonstrating robust performance. Further improvements can be made with deeper architectures and additional fine-tuning of hyperparameters. The trained model is available in the `resources` directory for experimentation and testing.
