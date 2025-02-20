import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = pd.read_csv('./Output_Data/High_score_all_data_0616.csv')
original_feature_names = data.columns.tolist()

# Create a mapping from original names to new names
new_feature_names = []
feature_mapping = {}
for name in original_feature_names:
    new_name = name.replace('[', '_').replace(']', '').replace('<', 'lt_').replace('>', 'gt_')
    new_feature_names.append(new_name)
    feature_mapping[name] = new_name

# Rename the columns in your DataFrame
data.columns = new_feature_names

df = data.drop(['route_id'], axis=1)

X = df.drop('Selection', axis=1)
y = df['Selection']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard normalization
scaler = StandardScaler()

# Initialize the XGBClassifier
xgb_model = xgb.XGBClassifier(random_state=42)

# Define the parameter grid (commented out for future use)
# param_grid = {
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'n_estimators': [100, 200, 300]
# }

# Set up the grid search with 5-fold cross-validation (commented out for future use)
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the grid search to the data (commented out for future use)
# grid_search.fit(X_train, y_train)

# Get the best parameters (commented out for future use)
# best_params = grid_search.best_params_
# print(f'Best parameters found: {best_params}')

# Use the best parameters found
best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300, 'subsample': 0.8}

# Initialize the XGBClassifier with the best parameters
xgb_model = xgb.XGBClassifier(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    n_estimators=best_params['n_estimators'],
    random_state=42
)

# Fit the model with the best parameters
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy with best parameters: {accuracy:.4f}')

# Calculate probabilities for the positive class
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC', fontsize=18)
plt.legend(loc="lower right", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the entire dataset (for global insights)
shap_values = explainer.shap_values(X)

# Calculate mean absolute SHAP values for each feature
phi = abs(shap_values).mean(axis=0)
sorted_indices = np.argsort(phi)[::-1]
sorted_phi = phi[sorted_indices]
sorted_features = X.columns[sorted_indices]

# Short version SHAP summary plot (top 6 features)
important_features_short = sorted_features[:6]
important_shap_values_short = shap_values[:, sorted_indices[:6]]

plt.figure(figsize=(12, 10))
shap.summary_plot(important_shap_values_short, X[important_features_short], show=False)
plt.title('SHAP Summary Plot for Top 6 Features', fontsize=13)
plt.xlabel('SHAP Value (impact on model output)', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Full version SHAP summary plot (all features)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Summary Plot for All Features', fontsize=13)
plt.xlabel('SHAP Value (impact on model output)', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Short version histogram of Mean SHAP Values (top 10 features)
total_phi = np.sum(phi)
cumulative_phi = np.cumsum(sorted_phi)
threshold = 0.92 * total_phi
num_features_80 = np.sum(cumulative_phi <= threshold)

plt.figure(figsize=(12, 10))
bars = plt.barh(sorted_features[:10], sorted_phi[:10], color='skyblue')

plt.xlabel('Mean SHAP Value ($\Phi_j$)', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.title('Subset of Histogram of Mean SHAP Values', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().invert_yaxis()  # Largest values at the top
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # Adjust layout
plt.axhline(y=num_features_80 - 0.5, color='red', linestyle='--', linewidth=2, label='92% of Total ($\Phi_j$) Explained')
plt.legend(fontsize=14)
plt.show()

# Full version histogram of Mean SHAP Values (all features)
plt.figure(figsize=(12, 10))
bars = plt.barh(sorted_features, sorted_phi, color='skyblue')
plt.xlabel('Mean SHAP Value ($\Phi_j$)', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.title('Full Version Histogram of Mean SHAP Values', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().invert_yaxis()  # Largest values at the top
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # Adjust layout
plt.show()

# 2. Histogram of Mean SHAP Values
plt.figure(figsize=(12, 10))
bars = plt.barh(sorted_features, sorted_phi, color='skyblue')
plt.xlabel('Mean SHAP Value ($\Phi_j$)', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.title('Histogram of Mean SHAP Values', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().invert_yaxis()  # Largest values at the top

# Calculate total and cumulative sums of SHAP values
total_phi = np.sum(phi)
cumulative_phi = np.cumsum(sorted_phi)
threshold = 0.92 * total_phi
num_features_80 = np.sum(cumulative_phi <= threshold)

plt.axhline(y=num_features_80 - 0.5, color='red', linestyle='--', linewidth=2, label='92% of Total ($\Phi_j$) Explained')
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # Adjust layout
plt.show()
