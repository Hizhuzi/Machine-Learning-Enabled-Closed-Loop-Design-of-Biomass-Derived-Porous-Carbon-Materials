import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_excel("2D-synBDPCs.xlsx", usecols=[0, 1, 2, 3, 4, 6, 8, 9])
data = data.dropna(axis=0)

# Specify feature and target columns
X = data.iloc[:, [0, 1, 2, 3, 4, 6, 7]]
y = data.iloc[:, [5]].values.ravel()

# One-hot encode categorical features
categorical_columns = X.columns[:4].tolist()
X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X.loc[:, num_cols] = scaler.fit_transform(X.loc[:, num_cols])

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=15)

# Initialize lists to store evaluation metrics
mse_train_list, rmse_train_list, mae_train_list, r2_train_list = [], [], [], []
mse_test_list, rmse_test_list, mae_test_list, r2_test_list = [], [], [], []

# Lists to store true and predicted values for all folds
y_true_train_total, y_pred_train_total = [], []
y_true_test_total, y_pred_test_total = [], []

fold = 1
for train_index, test_index in kf.split(X):
    # Split data into train and test sets for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train KNN model
    knn = KNeighborsRegressor(n_neighbors=4)
    knn.fit(X_train, y_train)

    # Predict on training and testing sets
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Store evaluation metrics
    mse_train_list.append(mse_train)
    rmse_train_list.append(rmse_train)
    mae_train_list.append(mae_train)
    r2_train_list.append(r2_train)

    mse_test_list.append(mse_test)
    rmse_test_list.append(rmse_test)
    mae_test_list.append(mae_test)
    r2_test_list.append(r2_test)

    # Store true and predicted values for the current fold
    y_true_train_total.extend(y_train)
    y_pred_train_total.extend(y_train_pred)
    y_true_test_total.extend(y_test)
    y_pred_test_total.extend(y_test_pred)

    # Save true and predicted values for the current fold to Excel files
    train_fold_results = pd.DataFrame({"True": y_train, "Predict": y_train_pred})
    test_fold_results = pd.DataFrame({"True": y_test, "Predict": y_test_pred})

    train_fold_results.to_excel(f"fold_{fold}_train_results.xlsx", index=False)
    test_fold_results.to_excel(f"fold_{fold}_test_results.xlsx", index=False)

    # Print performance for the current fold
    print(f"Fold {fold}:")
    print(f"Train MSE: {mse_train:.3f}, Train RMSE: {rmse_train:.3f}, Train MAE: {mae_train:.3f}, Train R2: {r2_train:.3f}")
    print(f"Test MSE: {mse_test:.3f}, Test RMSE: {rmse_test:.3f}, Test MAE: {mae_test:.3f}, Test R2: {r2_test:.3f}")

    # Plot true values vs predictions for the current fold
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, c='Tab:purple', alpha=0.6, edgecolors='none', label='Train')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True Values vs Predictions (Train) - Fold {fold}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, c='Tab:orange', alpha=0.6, edgecolors='none', label='Test')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True Values vs Predictions (Test) - Fold {fold}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    fold += 1

# Print average evaluation metrics across all folds
print("Average Train MSE: {:.3f}".format(np.mean(mse_train_list)))
print("Average Train RMSE: {:.3f}".format(np.mean(rmse_train_list)))
print("Average Train MAE: {:.3f}".format(np.mean(mae_train_list)))
print("Average Train R2: {:.3f}".format(np.mean(r2_train_list)))

print("Average Test MSE: {:.3f}".format(np.mean(mse_test_list)))
print("Average Test RMSE: {:.3f}".format(np.mean(rmse_test_list)))
print("Average Test MAE: {:.3f}".format(np.mean(mae_test_list)))
print("Average Test R2: {:.3f}".format(np.mean(r2_test_list)))

# Plot average true values vs predictions across all folds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_true_train_total, y_pred_train_total, alpha=0.6, edgecolors='none', label='Train')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Train Average)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_true_test_total, y_pred_test_total, alpha=0.6, edgecolors='none', label='Test')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Test Average)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the total true and predicted values for training and testing sets to Excel files
train_results = pd.DataFrame({"True": y_true_train_total, "Predict": y_pred_train_total})
test_results = pd.DataFrame({"True": y_true_test_total, "Predict": y_pred_test_total})

train_results.to_excel("train_results.xlsx", index=False)
test_results.to_excel("test_results.xlsx", index=False)