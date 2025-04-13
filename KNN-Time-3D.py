import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib  # Import joblib for model saving and loading

# Read data and preprocess
data = pd.read_excel("3D-synBDPCs.xlsx", usecols=[0, 1, 2, 3, 6, 9, 10, 11])
data = data.dropna(axis=0)

# Specify feature columns and target column
X = data.iloc[:, [0, 1, 2, 3, 5, 6, 7]]
y = data.iloc[:, [4]].values.ravel()

# One-hot encoding
categorical_columns = X.columns[:3].tolist()
X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)

# Normalization
scaler = MinMaxScaler()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# Keep the column order from the training set
final_columns = X.columns

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=23)

# Initialize evaluation metric lists
mse_train_list, rmse_train_list, mae_train_list, r2_train_list = [], [], [], []
mse_test_list, rmse_test_list, mae_test_list, r2_test_list = [], [], [], []

# Store true and predicted values for all folds
y_true_train_total, y_pred_train_total = [], []
y_true_test_total, y_pred_test_total = [], []

fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train KNN model
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)

    # Save model
    model_filename = f"knn_model_fold_{fold}.joblib"
    joblib.dump(knn, model_filename)
    print(f"Model for fold {fold} saved as {model_filename}.")

    # Evaluate model
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

    # Store true and predicted values for all folds
    y_true_train_total.extend(y_train)
    y_pred_train_total.extend(y_train_pred)
    y_true_test_total.extend(y_test)
    y_pred_test_total.extend(y_test_pred)

    # Print performance for each fold
    print(f"Fold {fold}:")
    print(f"Train MSE: {mse_train:.3f}, Train RMSE: {rmse_train:.3f}, Train MAE: {mae_train:.3f}, Train R2: {r2_train:.3f}")
    print(f"Test MSE: {mse_test:.3f}, Test RMSE: {rmse_test:.3f}, Test MAE: {mae_test:.3f}, Test R2: {r2_test:.3f}")
    
    # Plot scatter plot for training and testing sets
    plt.figure(figsize=(12, 6))
    plt.scatter(y_train, y_train_pred, c='Tab:purple', alpha=0.6, edgecolors='none', label='Train')
    plt.scatter(y_test, y_test_pred, c='Tab:orange', alpha=0.6, edgecolors='none', label='Test')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True Values vs Predictions - Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fold += 1

# Print average evaluation metrics
print("Average Train MSE: {:.3f}".format(np.mean(mse_train_list)))
print("Average Train RMSE: {:.3f}".format(np.mean(rmse_train_list)))
print("Average Train MAE: {:.3f}".format(np.mean(mae_train_list)))
print("Average Train R2: {:.3f}".format(np.mean(r2_train_list)))

print("Average Test MSE: {:.3f}".format(np.mean(mse_test_list)))
print("Average Test RMSE: {:.3f}".format(np.mean(rmse_test_list)))
print("Average Test MAE: {:.3f}".format(np.mean(mae_test_list)))
print("Average Test R2: {:.3f}".format(np.mean(r2_test_list)))

# Plot average scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(y_true_train_total, y_pred_train_total, c='Tab:purple', alpha=0.6, edgecolors='none', label='Train Average')
plt.scatter(y_true_test_total, y_pred_test_total, c='Tab:orange', alpha=0.6, edgecolors='none', label='Test Average')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Train & Test Average)')
plt.legend()
plt.tight_layout()
plt.show()

# Output the true and predicted values of the training and test sets to an Excel sheet
train_results = pd.DataFrame({"True": y_true_train_total, "Predict": y_pred_train_total})
test_results = pd.DataFrame({"True": y_true_test_total, "Predict": y_pred_test_total})

train_results.to_excel("train_results.xlsx", index=False)
test_results.to_excel("test_results.xlsx", index=False)

# Save the transformer for later use
scaler_filename = "scaler.joblib"
joblib.dump(scaler, scaler_filename)
joblib.dump(final_columns, "final_columns.joblib")
print(f"Scaler and final columns saved as {scaler_filename} and final_columns.joblib.")