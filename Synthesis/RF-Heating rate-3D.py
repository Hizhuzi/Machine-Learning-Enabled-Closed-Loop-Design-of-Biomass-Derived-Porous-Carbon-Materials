import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Load data and preprocess
data = pd.read_excel("3D-synBDPCs.xlsx", usecols=[0, 1, 2, 3, 5, 9, 10, 11])
data = data.dropna(axis=0)

# Specify feature and target columns
X = data.iloc[:, [0, 1, 2, 3, 5, 6, 7]]
y = data.iloc[:, [4]].values.ravel()

# One-hot encode categorical features
categorical_columns = X.columns[:3].tolist()
X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X = X.copy()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X.loc[:, num_cols] = scaler.fit_transform(X.loc[:, num_cols])

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=52)

# Store results for each fold
train_scores = []
test_scores = []

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize the RandomForestRegressor model
    rf = RandomForestRegressor(criterion='squared_error', 
                               n_estimators=50,
                               min_samples_split=2,
                               max_depth=16,
                               random_state=22)
    
    # Train the model
    rf.fit(X_train, y_train)

    # Save the trained model for each fold
    dump(rf, f'random_forest_model_fold_{fold}.joblib')

    # Predict on training and test sets
    y_hat_train = rf.predict(X_train)
    y_hat_test = rf.predict(X_test)

    # Calculate and store evaluation metrics
    train_scores.append({
        'Fold': fold,
        'MSE': mean_squared_error(y_train, y_hat_train),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_hat_train)),
        'MAE': mean_absolute_error(y_train, y_hat_train),
        'R2': r2_score(y_train, y_hat_train)
    })
    test_scores.append({
        'Fold': fold,
        'MSE': mean_squared_error(y_test, y_hat_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_hat_test)),
        'MAE': mean_absolute_error(y_test, y_hat_test),
        'R2': r2_score(y_test, y_hat_test)
    })

    # Generate and save scatter plots for each fold
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_hat_train, c='Tab:purple', alpha=0.6, edgecolors='none', label='Train')
    plt.scatter(y_test, y_hat_test, c='Tab:orange', alpha=0.6, edgecolors='none', label='Test')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.title(f'True Values vs Predictions (Fold {fold})')
    plt.savefig(f'scatter_plot_fold_{fold}.png')
    plt.close()

    # Save true and predicted values for each fold to Excel
    train_results = pd.DataFrame({"Truth": y_train, "Predict": y_hat_train})
    test_results = pd.DataFrame({"Truth": y_test, "Predict": y_hat_test})
    train_results.to_excel(f"train_results_fold_{fold}.xlsx", index=False)
    test_results.to_excel(f"test_results_fold_{fold}.xlsx", index=False)

    print(f"Fold {fold} completed")

# Print results for each fold
print("\nDetailed Train Scores:")
print(pd.DataFrame(train_scores))
print("\nDetailed Test Scores:")
print(pd.DataFrame(test_scores))

# Calculate and print average scores across all folds
avg_train_scores = pd.DataFrame(train_scores).mean()
avg_test_scores = pd.DataFrame(test_scores).mean()

print("\nAverage Train Scores:")
print(avg_train_scores)
print("\nAverage Test Scores:")
print(avg_test_scores)

# Save the average scores to an Excel file
summary_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Train': [avg_train_scores['MSE'], avg_train_scores['RMSE'], avg_train_scores['MAE'], avg_train_scores['R2']],
    'Test': [avg_test_scores['MSE'], avg_test_scores['RMSE'], avg_test_scores['MAE'], avg_test_scores['R2']]
})

summary_df.to_excel("cv_results_ratio.xlsx", index=False)