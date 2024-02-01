# Phase 1: Data Collection and Preprocessing

# Import necessary libraries
import pandas as pd
import numpy as np

# Step 1: Data Collection
# Assuming you have a dataset in CSV format with relevant health features
dataset_path = '/Users/sanketjagtap/Desktop2/Coading/Nexus Tasks/Phase 2/data.csv'
df = pd.read_csv(dataset_path)

# Step 2: Data Preprocessing
# Handle missing values
df = df.dropna()

# Handle outliers (you may need to customize this based on your dataset)
lower_bound = 0.05
upper_bound = 0.95
quant_df = df.quantile([lower_bound, upper_bound])

df = df.apply(lambda x: x[(x >= quant_df.loc[lower_bound, x.name]) & 
                          (x <= quant_df.loc[upper_bound, x.name])], axis=0)

# Normalize or standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# Phase 2: Feature Selection

# Import necessary libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Step 3: Feature Selection
# Assuming 'target' is the column indicating the presence of a disease
X = df_scaled.drop('target', axis=1)
y = df_scaled['target']

# Use SelectKBest to select the top k features
k_best = SelectKBest(chi2, k='all')
X_new = k_best.fit_transform(X, y)

# Get the selected features
selected_features = X.columns[k_best.get_support()]


# Phase 3: Model Development

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 4: Model Development
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# Phase 4: Model Development

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 4: Model Development
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)

print("Logistic Regression:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")
print()

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_precision = precision_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred)

print("Decision Tree:")
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")
print()

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)

print("Random Forest:")
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")
print()

# Train and evaluate Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_precision = precision_score(y_test, svm_y_pred)
svm_recall = recall_score(y_test, svm_y_pred)
svm_f1 = f1_score(y_test, svm_y_pred)

print("Support Vector Machine (SVM):")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1 Score: {svm_f1}")
print()


# Phase 5: Cross-Validation

# Import necessary libraries
from sklearn.model_selection import cross_val_score

# Step 5: Cross-Validation
# Perform cross-validation on the entire dataset
cv_scores = cross_val_score(model, X_new, y, cv=5)  # You can adjust the number of folds (cv) as needed
average_cv_accuracy = np.mean(cv_scores)
print(f'Average Cross-Validation Accuracy: {average_cv_accuracy}')


# Phase 6: Hyperparameter Tuning

# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Step 6: Hyperparameter Tuning
# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_new, y)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')


# Phase 7: Model Interpretability

# Import necessary libraries
import shap

# Step 7: Model Interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_new)

# Plot summary plot
shap.summary_plot(shap_values, X_new)


# Phase 8: User Interface

# Import necessary libraries
from flask import Flask, render_template, request

app = Flask(__name__)

# Assume 'model' is the trained machine learning model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form['feature1']), float(request.form['feature2']), ...]  # Update with your features
        input_data = [features]  # Convert to a 2D array
        prediction = model.predict(input_data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)


# Phase 9: Integration with Electronic Health Records

# Assume you have an API for accessing Electronic Health Records

import requests

ehr_api_url = 'https://ehrapi.example.com/patient_data'

def get_ehr_data(patient_id):
    params = {'patient_id': patient_id}
    response = requests.get(ehr_api_url, params=params)

    if response.status_code == 200:
        ehr_data = response.json()
        # Process and use the data as needed
        return ehr_data
    else:
        print(f'Error accessing EHR data. Status code: {response.status_code}')
        return None

# Use this function to retrieve EHR data before making predictions
