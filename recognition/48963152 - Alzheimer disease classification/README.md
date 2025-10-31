# Alzheimer's Disease Classification

This project implements a deep learning pipeline for classifying Alzheimer's disease (AD) vs Cognitive Normal (CN) from the medical images. It builds the ConvNeXt architecture and handles training, evaluation and prediction using PyTorch and torchvision.

### Project Structure
```text
├── dataset.py     → Preparing the dataset for image classification
├── modules.py     → ConvNeXt-based classifier model definition
├── train.py       → Script to train and validate the ConvNeXt model
└── predict.py     → Script to evaluate a trained model on test data
```

<hr><br>

## Steps to Run
Install the requirements.txt using the following command
```text
pip install requirements.txt
```

### CASE 1: To train the model:
```text
python train.py
```
### CASE 2: To test the model:
```text
python test.py
```
Paste the relative path of the file in case the command is not working
<hr>

 ## Overview of Components

1. ### "dataset.py" <br>
It defines a class 'ADNIDataset' that loads the images present in the root directory. 
It expects the directory to be in the following structure

```text
root_dir/         → example: train/ or test/
│
├── AD/           # Alzheimer’s Disease images
│   ├── img1.jpeg
│   ├── img2.jpeg
│   └── ...
│
└── NC/           # Cognitive Normal images
    ├── img1.jpeg
    ├── img2.jpeg
    └── ...
```
The extension of the images could be either '.png', '.jpg' or '.jpeg'

2. ### "modules.py" <br>
It implements the architecture of the model. <br>
It creates a class ConvNeXtClassifier that supports tiny, small and base ConvNeXt backbone configurations. The final classification head is replaced with a custom head which enables it to do binary classification. <br>
Following are the constructor parameters: <br>
```markdown
| Argument           | Type  | Default | Description                                   |
|--------------------|-------|---------|-----------------------------------------------|
| `variant`          | str   | `tiny`  | ConvNeXt variant (`tiny`, `small`, or `base`) |
| `number_of_classes`| int   | `2`     | Number of classes to classify                 |
| `dropout`          | float | `0.2`   | Dropout rate for regularization               |
```

3. ### "train.py"
The following is the workflow of the training of the model
```text
1. Load dataset using `ADNIDataset`.
2. Split into training and validation sets (default 80/20).
3. Define model, optimizer (`Adam`), loss (`CrossEntropyLoss`), and LR scheduler.
4. Train for a set number of epochs with accuracy and loss tracking.
5. Validate model performance each epoch.
6. Save trained weights to `saved_models/convnext_adni.pth`.
7. Plot loss and accuracy curves for visualization.
```

Hyperparameters:

```markdown
| Parameter       | Value   |
|-----------------|---------|
| `batch_size`    | `64`    |
| `num_epochs`    | `80     |
| `learning_rate` | `1e-4`  |
| `val_split`     | `0.2`   |
| `dropout`       | `0.3`   |
```

The plotted Accuracy and loss graphs would look like the following:
<img width="1000" height="400" alt="training_plot" src="https://github.com/user-attachments/assets/ebef0901-4294-4800-801f-5700e8d3c013" />


4. ### "predict.py"
Evaluates the model saved from training at path 'saved_models/convnext_agni.pth'. Following are the steps:
```text
1. Load test dataset using the same transforms as training.
2. Initialize model and load trained weights.
3. Run inference on the test set.
4. Compute overall accuracy and print to console.
```
