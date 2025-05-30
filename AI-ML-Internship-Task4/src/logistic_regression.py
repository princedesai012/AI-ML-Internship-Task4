import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path for the results directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '../results')
data_dir = os.path.join(script_dir, '../data')

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory at: {results_dir}")

# Verify results directory is writable
if not os.access(results_dir, os.W_OK):
    print(f"Error: No write permission for results directory: {results_dir}")
    exit(1)

# 1. Load and prepare the dataset
try:
    data = pd.read_csv(os.path.join(data_dir, 'breast_cancer_data.csv'))
except FileNotFoundError:
    print(f"Error: Dataset file not found at {os.path.join(data_dir, 'breast_cancer_data.csv')}")
    exit(1)

# Drop 'id', 'diagnosis', and problematic 'Unnamed: 32' column
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
y = data['diagnosis'].map({'M': 1, 'B': 0})  # Target: Malignant=1, Benign=0

# Verify columns
print("Feature columns:", X.columns.tolist())
print("Number of NaN values before imputation:\n", X.isna().sum())

# 2. Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Verify no NaN values remain
if X_imputed.isna().sum().sum() > 0:
    print("Error: NaN values still present after imputation:\n", X_imputed.isna().sum())
    exit(1)
else:
    print("All NaN values handled successfully.")

# Check for non-numeric columns
non_numeric_cols = X_imputed.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Error: Non-numeric columns found: {non_numeric_cols}")
    exit(1)

# Check for columns with zero variance
zero_variance_cols = X_imputed.columns[X_imputed.nunique() == 1]
if len(zero_variance_cols) > 0:
    print(f"Warning: Dropping columns with zero variance: {zero_variance_cols}")
    X_imputed = X_imputed.drop(zero_variance_cols, axis=1)

# 3. Train/test split and standardize features
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
try:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except ValueError as e:
    print(f"Error during standardization: {e}")
    exit(1)

# 4. Fit Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(results_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Saved confusion matrix to: {cm_path}")

# Classification Report (Precision, Recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# ROC Curve and AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_path = os.path.join(results_dir, 'roc_curve.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to: {roc_path}")

# 6. Sigmoid Function Visualization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.legend()
sigmoid_path = os.path.join(results_dir, 'sigmoid_plot.png')
plt.savefig(sigmoid_path)
plt.close()
print(f"Saved sigmoid plot to: {sigmoid_path}")

# 7. Threshold Tuning (Optimizing for recall)
from sklearn.metrics import recall_score
thresholds = np.arange(0.1, 1.0, 0.1)
recalls = []
for thresh in thresholds:
    y_pred_thresh = (model.predict_proba(X_test_scaled)[:, 1] >= thresh).astype(int)
    recalls.append(recall_score(y_test, y_pred_thresh))

plt.figure(figsize=(8, 6))
plt.plot(thresholds, recalls, marker='o')
plt.title('Recall vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.grid(True)
threshold_path = os.path.join(results_dir, 'threshold_recall.png')
plt.savefig(threshold_path)
plt.close()
print(f"Saved threshold recall plot to: {threshold_path}")

print("Model evaluation and visualizations saved in 'results' folder.")