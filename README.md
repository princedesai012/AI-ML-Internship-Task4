# AI-ML-Internship-Task4

## Overview
This project implements a binary classifier using Logistic Regression on the Breast Cancer Wisconsin Dataset to predict whether a tumor is malignant or benign.

## Dataset
- **Source**: Breast Cancer Wisconsin Dataset
- **Features**: 30 numeric features (e.g., radius, texture, perimeter)
- **Target**: Diagnosis (M = Malignant, B = Benign)
- 
## Project Structure
AI-ML-Internship-Task4/
├── data/
│   └── breast_cancer_data.csv  # Dataset
├── src/
│   └── logistic_regression.py  # Main script
├── results/
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   ├── roc_curve.png           # ROC curve
│   ├── sigmoid_plot.png        # Sigmoid function visualization
│   └── threshold_recall.png     # Recall vs. threshold plot
├── README.md                   # Documentation
└── requirements.txt            # Dependencies

# Setup
1. Clone this repository.
2. Place the Breast Cancer Wisconsin Dataset in `data/breast_cancer_data.csv`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run the script from the AI-ML-Internship-Task4/ directory:
```bash
cd AI-ML-Internship-Task4
python src/logistic_regression.py

```

Implementation Details
Data Preprocessing:
Loaded dataset and mapped diagnosis to binary (M=1, B=0).
Dropped id, diagnosis, and Unnamed: 32 columns.
Used SimpleImputer to handle any remaining missing values with mean imputation.
Checked for non-numeric columns and zero-variance columns.
Split data (80% train, 20% test) and standardized features using StandardScaler.

Model Training:
Trained a Logistic Regression model using scikit-learn.

Evaluation:
Generated confusion matrix, precision, recall, and F1-score.
Plotted ROC curve and calculated AUC.
Visualized the sigmoid function.
Analyzed recall across classification thresholds.

Outputs:
Visualizations saved in results/ folder.
Classification report printed to console.

Results
Visualizations: Confusion matrix, ROC curve, sigmoid function, and threshold-recall plot.
Classification report: Precision, recall, and F1-score for both classes.

Requirements
See requirements.txt for dependencies.

Notes
Handled Unnamed: 32 column (all NaN values) by dropping it before imputation.
Used absolute paths to ensure plots are saved correctly.
Added debugging output to verify data integrity and plot saving.
text

Copy

### requirements.txt
Your `requirements.txt`:
```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
Next Steps

Check Results Folder: Verify that results/ contains the four PNG files.
```
