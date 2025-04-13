import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import os

# Set random seeds for reproducibility
random_state = 13
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Load data
data = pd.read_excel("2D-synBDPCs.xlsx", usecols=[0, 1, 2, 8, 9])
data = data.dropna(axis=0)

# Specify feature and target columns
X = data.iloc[:, [0, 1, 3, 4]]
y = data.iloc[:, [2]].values.ravel()

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode categorical features
categorical_columns = X.columns[:2].tolist()
X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)

# One-hot encode target variable
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=3)

# Calculate class weights for handling imbalance
y_train_labels = np.argmax(y_train, axis=1)
counter = Counter(y_train_labels)
total = sum(counter.values())
class_weight = {class_id: total / count for class_id, count in counter.items()}

# Define the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, class_weight=class_weight)

# Make predictions on training and testing sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions to labels
y_pred_train_labels = label_encoder.inverse_transform(np.argmax(y_pred_train, axis=1))
y_pred_test_labels = label_encoder.inverse_transform(np.argmax(y_pred_test, axis=1))

# Calculate accuracy
accuracy_train = accuracy_score(label_encoder.inverse_transform(np.argmax(y_train, axis=1)), y_pred_train_labels)
accuracy_test = accuracy_score(label_encoder.inverse_transform(np.argmax(y_test, axis=1)), y_pred_test_labels)
print("Training Set Accuracy:", accuracy_train)
print("Test Set Accuracy:", accuracy_test)

# Calculate precision
precision_train = precision_score(label_encoder.inverse_transform(np.argmax(y_train, axis=1)), y_pred_train_labels, average='weighted', zero_division=1)
precision_test = precision_score(label_encoder.inverse_transform(np.argmax(y_test, axis=1)), y_pred_test_labels, average='weighted', zero_division=1)
print("Training Set Precision:", precision_train)
print("Test Set Precision:", precision_test)

# Get labels for confusion matrix
labels = [label_encoder.classes_[i] for i in np.unique(y_train.argmax(axis=1))]

# Calculate and display AUC scores
def print_auc_scores(y_true, y_pred_proba, labels, set_name):
    print(f"\nAUC Scores for {set_name}:")
    for i, label in enumerate(labels):
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        print(f"{label}: {auc:.4f}")

# Print AUC scores for training and testing sets
print_auc_scores(y_train, y_pred_train, labels, "Training Set")
print_auc_scores(y_test, y_pred_test, labels, "Test Set")

# Plot ROC curves
def plot_roc_curve(y_true, y_pred_proba, labels, title='ROC Curve'):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curves for training and testing sets
plot_roc_curve(y_train, y_pred_train, labels, title='ROC Curve (Training Set)')
plot_roc_curve(y_test, y_pred_test, labels, title='ROC Curve (Test Set)')