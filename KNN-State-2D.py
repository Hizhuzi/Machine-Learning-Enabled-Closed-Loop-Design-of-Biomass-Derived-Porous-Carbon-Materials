import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_excel("2D-synBDPCs.xlsx", usecols=[1, 8, 9])
data = data.dropna(axis=0)

# Specify feature and target columns
X = data.iloc[:, [1, 2]]
y = data.iloc[:, [0]].values.ravel()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale features
scaler = MinMaxScaler()
X = X.copy()  # Create a copy to avoid SettingWithCopyWarning
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)

# Handle class imbalance in the training set using SMOTE
smote = SMOTE(random_state=7)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# One-hot encode target variable for both training and testing sets
onehot_encoder = OneHotEncoder(sparse_output=False) 
y_train_resampled_onehot = onehot_encoder.fit_transform(y_train_resampled.reshape(-1, 1))
y_test_onehot = onehot_encoder.transform(y_test.reshape(-1, 1))

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_train = knn.predict(X_train_resampled)
y_pred_test = knn.predict(X_test)

# Compute accuracy
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Training Set Accuracy:", accuracy_train)
print("Test Set Accuracy:", accuracy_test)

# Compute precision
precision_train = precision_score(y_train_resampled, y_pred_train, average='weighted', zero_division=1)
precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
print("Training Set Precision:", precision_train)
print("Test Set Precision:", precision_test)

# Confusion matrices
labels = np.unique(y_train_resampled)
cm_train = confusion_matrix(y_train_resampled, y_pred_train)
cm_train_norm = cm_train / cm_train.sum(axis=1, keepdims=True) * 1
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train_norm, display_labels=label_encoder.classes_)
disp_train.plot(cmap=plt.cm.BuPu)

cm_test = confusion_matrix(y_test, y_pred_test)
cm_test_norm = cm_test / cm_test.sum(axis=1, keepdims=True) * 1
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test_norm, display_labels=label_encoder.classes_)
disp_test.plot(cmap=plt.cm.BuPu)

# Get predicted probabilities
y_pred_proba_train = knn.predict_proba(X_train_resampled)
y_pred_proba_test = knn.predict_proba(X_test)

# Print shapes for verification
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("y_pred_proba_train shape:", y_pred_proba_train.shape)
print("y_pred_proba_test shape:", y_pred_proba_test.shape)

# Ensure the length of labels list matches the number of classes
n_classes = y_test_onehot.shape[1]
labels = label_encoder.classes_[:n_classes]

# Compute and print AUC scores
def print_auc_scores(y_true, y_pred_proba, n_classes, labels, set_name):
    print(f"\nAUC Scores for {set_name}:")
    for i in range(n_classes):
        auc = roc_auc_score(y_true == i, y_pred_proba[:, i])
        print(f"{labels[i]}: {auc:.4f}")

# Print AUC scores for training and testing sets
print_auc_scores(y_train_resampled, y_pred_proba_train, n_classes, labels, "Training Set")
print_auc_scores(y_test, y_pred_proba_test, n_classes, labels, "Test Set")

# Plot ROC curves
def plot_roc_curve(y_true, y_pred_proba, n_classes, labels, title='ROC Curve'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score(y_true == i, y_pred_proba[:, i])
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve for training set
plot_roc_curve(y_train_resampled, y_pred_proba_train, n_classes, labels, title='ROC Curve (Training Set)')

# Plot ROC curve for test set
plot_roc_curve(y_test, y_pred_proba_test, n_classes, labels, title='ROC Curve (Test Set)')

print("Classes:", label_encoder.classes_)