# goit-ml-fp
Machine Learning: Fundamentals and Applications - Final Project

The main goal of the final project of the course is to predict which customers of the company may consider changing their service provider (according to a well-known marketing term, this is "customer churn").

## Table of Contents
- [Project Overview](#project-overview)
- [Conclusions](#conclusions)

## Project Overview

This project focuses on building and evaluating a machine learning model to predict customer churn. The process involves data preprocessing, pipeline construction, hyperparameter tuning, and model evaluation.

### **Step 1: Importing Necessary Packages**
In this step, all the necessary Python packages are imported. These include libraries for data manipulation, model building, preprocessing, and evaluation.

### **Step 2: Loading and Preparing the Dataset**
Load the training and test datasets from from the final_proj_data.csv and final_proj_test.csv files.

### **Step 3: Selecting Numeric and Categorical Features**
Identify numeric and categorical features for separate preprocessing.

### **Step 4: Removing Columns with All Missing Values**
Filter out any numeric features with all missing values.

### **Step 5: Creating Preprocessing Pipelines**
Set up pipelines for preprocessing numeric and categorical data.

### **Step 6: Combining Preprocessing Pipelines**
Combine the preprocessing steps for both numeric and categorical features into a single pipeline using ColumnTransformer.

### **Step 7: Separating Features and Target Variable**
Split the dataset into features (X) and target variable (y).

### **Step 8: Preprocessing the Data**
Apply the preprocessing pipeline to the features.

### **Step 9: Splitting Data into Training and Validation Sets**
Split the preprocessed data into training and validation sets.

### **Step 10: Applying SMOTE for Class Imbalance**
Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution in the training set.

### **Step 11: Defining a RandomForestClassifier Model**
Initialize a Random Forest classifier.

### **Step 12: Setting Up a Hyperparameter Grid for Tuning**
Define a grid of hyperparameters to tune the Random Forest model.

### **Step 13: Performing Grid Search with Cross-Validation**
Use GridSearchCV to find the best hyperparameters for the Random Forest model by evaluating it with 3-fold cross-validation.

### **Step 14: Outputting the Best Parameters and Cross-Validated Score**
Display the best hyperparameters found during grid search and the corresponding balanced accuracy score.

### **Step 15: Training the Final Model with Best Parameters**
Train the Random Forest model using the best hyperparameters found in the grid search.

### **Step 16: Adjusting the Prediction Threshold**
Predict probabilities for the validation set and adjust the classification threshold to optimize performance.

### **Step 17: Evaluating the Model on Validation Set**
Evaluate the performance of the adjusted model using various metrics: balanced accuracy, confusion matrix, F1 score, precision, recall, and classification report.

### **Step 18: Preprocessing the Test Set and Making Predictions**
Apply the preprocessing pipeline to the test set and use the trained model to make predictions.

### **Step 19: Creating the Submission File**
Create the final submission file for the predictions on the test set, and the results are saved in a file named submission_adjusted.csv.

## **Conclusions**
**Best Parameters Found:** {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

**Best Balanced Accuracy Score During Cross-Validation:** 0.9569

After adjusting the prediction threshold on the validation set:

**Balanced Accuracy Score (Adjusted RF):** 0.7910

**Public Balanced Accuracy Score:** 0.7875

**Private Balanced Accuracy Score:** 0.7928

**Confusion Matrix (Adjusted RF):**

```
[[1579  158]
 [  86  177]]
```

**F1 Score (Adjusted RF):** 0.5920

**Precision (Adjusted RF):** 0.5284

**Recall (Adjusted RF):** 0.6730

These results highlight the effectiveness of the Random Forest model with hyperparameter tuning and class imbalance handling using SMOTE. 
