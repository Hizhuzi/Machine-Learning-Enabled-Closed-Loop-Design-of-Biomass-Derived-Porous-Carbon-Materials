import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from joblib import dump

# Load the data
data = pd.read_excel("3D-synBDPCs.xlsx", usecols=[0, 9, 10, 11])
data = data.dropna(axis=0)

# Specify feature and target columns
X = data.iloc[:, [1, 2, 3]]
y = data.iloc[:, [0]].values.ravel()

# Create a LabelEncoder object
label_encoder = LabelEncoder()
# Encode the target labels
y = label_encoder.fit_transform(y)

# One-hot encode the target variable
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))

# Normalize the features
scaler = MinMaxScaler()
X = X.copy()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X.loc[:, num_cols] = scaler.fit_transform(X.loc[:, num_cols])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=9)

# -------Random Forest Training and Evaluation-------
# Initialize the RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',
                            n_estimators=30,
                            max_depth=30,
                            random_state=10,
                            class_weight='balanced')

# Train the model
rf.fit(X_train, y_train)

# Save the trained model
dump(rf, 'random_forest_model-3D-Pre-carbonization.joblib')

# Make predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Training Set Accuracy:", accuracy_train)
print("Test Set Accuracy:", accuracy_test)

# Calculate precision
precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=1)
precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
print("Training Set Precision:", precision_train)
print("Test Set Precision:", precision_test)

# Set font size for plots
plt.rcParams['font.size'] = 20

# Get labels for the confusion matrix
labels = [label_encoder.classes_[i] for i in np.unique(y_train.argmax(axis=1))]

# Define a function to plot the confusion matrix (convert to percentage)
def plot_confusion_matrix(cm, labels):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_normalized * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.BuPu)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm_percent.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_percent[i, j], fmt) + '%',
                    ha="center", va="center",
                    color="white" if cm_percent[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# Confusion matrix for training set
cm_train = confusion_matrix(y_train.argmax(axis=1), y_pred_train.argmax(axis=1))
plot_confusion_matrix(cm_train, labels)

# Confusion matrix for test set
cm_test = confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
plot_confusion_matrix(cm_test, labels)

# Define a function to extract predicted probabilities
def get_pred_proba_dict(y_pred_proba):
    if isinstance(y_pred_proba, list):
        return {i: prob[:, 1] for i, prob in enumerate(y_pred_proba)}
    elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
        return {i: y_pred_proba[:, i] for i in range(y_pred_proba.shape[1])}
    else:
        raise ValueError("Unknown y_pred_proba format")

# Get predicted probabilities
y_pred_proba_train = rf.predict_proba(X_train)
y_pred_proba_test = rf.predict_proba(X_test)
y_pred_proba_dict_train = get_pred_proba_dict(y_pred_proba_train)
y_pred_proba_dict_test = get_pred_proba_dict(y_pred_proba_test)

# Print shapes for debugging
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("y_pred_proba_dict_train shape:", {k:v.shape for k,v in y_pred_proba_dict_train.items()})
print("y_pred_proba_dict_test shape:", {k:v.shape for k,v in y_pred_proba_dict_test.items()})

# Ensure the labels list length matches the number of classes
n_classes = y_test.shape[1]
labels = label_encoder.classes_[:n_classes]

# Define a function to calculate and print AUC scores
def print_auc_scores(y_true, y_pred_proba_dict, n_classes, labels, set_name):
    print(f"\nAUC Scores for {set_name}:")
    for i in range(n_classes):
        auc = roc_auc_score(y_true[:, i], y_pred_proba_dict[i])
        print(f"{labels[i]}: {auc:.4f}")

# Print AUC scores for training and testing sets
print_auc_scores(y_train, y_pred_proba_dict_train, n_classes, labels, "Training Set")
print_auc_scores(y_test, y_pred_proba_dict_test, n_classes, labels, "Test Set")

# Define a function to plot ROC curves
def plot_roc_curve(y_true, y_pred_proba_dict, n_classes, labels, title='ROC Curve'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba_dict[i])
        roc_auc[i] = roc_auc_score(y_true[:, i], y_pred_proba_dict[i])

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
plot_roc_curve(y_train, y_pred_proba_dict_train, n_classes, labels, title='ROC Curve (Training Set)')

# Plot ROC curve for test set
plot_roc_curve(y_test, y_pred_proba_dict_test, n_classes, labels, title='ROC Curve (Test Set)')

# Print the class labels
print("Classes:", label_encoder.classes_)