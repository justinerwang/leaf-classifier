# Leaf Classifier

Classify 176 species of leaves using transfer learning with a fine-tuned ResNet-50. Built as a Kaggle competition project. https://www.kaggle.com/competitions/classify-leaves/overview. 


**Public score: 95.4%**
Trained on Colab GPU L4


## Approach

- **Model**: ResNet-50 pretrained on ImageNet, with `layer4` and `fc` fine-tuned (earlier layers frozen)
- **Training**: 30 epochs, Adam optimizer, cross-entropy loss, 5-fold cross-validation for hyperparameter tuning
- **Data augmentation**: Random resized crop, random horizontal flip
- **Dataset**: 18,353 training images, 8,800 test images (224x224)

## Usage

The project runs as a Jupyter notebook (`leaf-classifier.ipynb`). Place the dataset under `data/` with this structure:

```
data/
  train.csv
  test.csv
  images/
    0.jpg
    1.jpg
    ...
```

Then run all cells in the notebook. The final predictions are saved to `submission.csv`.

## Requirements

- PyTorch
- torchvision
- d2l
- pandas
- PIL
