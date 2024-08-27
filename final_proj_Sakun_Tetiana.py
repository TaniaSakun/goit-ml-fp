import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE

# %%

# Load Data
train_data = pd.read_csv('./datasets/final_proj_data.csv')
test_data = pd.read_csv('./datasets/final_proj_test.csv')
submission = pd.read_csv('./datasets/final_proj_sample_submission.csv')

# %%

# Select numeric and categorical features
numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.drop('y')
categorical_features = train_data.select_dtypes(include=['object']).columns

# Remove features with all missing values
numeric_features = numeric_features[train_data[numeric_features].isnull().mean() < 1.0]

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# %%

# Separate features and target
X = train_data.drop('y', axis=1)
y = train_data['y']

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform grid search with 3-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# Best parameters and corresponding score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Balanced Accuracy Score: {grid_search.best_score_}")

# Train the best model on the resampled training data
best_rf_final = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf_final.fit(X_train_res, y_train_res)

# Apply the adjusted threshold to validation set
y_proba_val = best_rf_final.predict_proba(X_val)[:, 1]
threshold = 0.3  # Adjust this threshold based on validation performance
y_pred_adjusted_val = (y_proba_val >= threshold).astype(int)

# %%

# Evaluate the adjusted model on validation set
balanced_acc_adjusted_rf = balanced_accuracy_score(y_val, y_pred_adjusted_val)
conf_matrix_adjusted_rf = confusion_matrix(y_val, y_pred_adjusted_val)
f1_adjusted_rf = f1_score(y_val, y_pred_adjusted_val)
precision_adjusted_rf = precision_score(y_val, y_pred_adjusted_val)
recall_adjusted_rf = recall_score(y_val, y_pred_adjusted_val)

print(f"Balanced accuracy score (Adjusted RF): {balanced_acc_adjusted_rf}")
print(f"Confusion Matrix (Adjusted RF):\n{conf_matrix_adjusted_rf}")
print(f"F1 Score (Adjusted RF): {f1_adjusted_rf}")
print(f"Precision (Adjusted RF): {precision_adjusted_rf}")
print(f"Recall (Adjusted RF): {recall_adjusted_rf}")
print("Classification Report:")
print(classification_report(y_val, y_pred_adjusted_val))

# Preprocess the test set and apply the same threshold for predictions
X_test_preprocessed = preprocessor.transform(test_data)
y_proba_test = best_rf_final.predict_proba(X_test_preprocessed)[:, 1]
test_predictions_adjusted = (y_proba_test >= threshold).astype(int)

# %%

# Create submission file
submission_adjusted = pd.DataFrame({
    'index': submission['index'],
    'y': test_predictions_adjusted
})
submission_adjusted.to_csv('submission_adjusted.csv', index=False)