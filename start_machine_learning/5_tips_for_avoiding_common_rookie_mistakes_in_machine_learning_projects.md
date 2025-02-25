# 5 Tips for Avoiding Common Rookie Mistakes in Machine Learning Projects
By Matthew Mayo on November 15, 2024 in Start Machine Learning 0
 Post Share
5 Tips for Avoiding Common Rookie Mistakes in Machine Learning Projects
5 Tips for Avoiding Common Rookie Mistakes in Machine Learning Projects
Image by Editor | Ideogram & Canva

It’s easy enough to make poor decisions in your machine learning projects that derail your efforts and jeopardize your outcomes, especially as a beginner. While you will undoubtedly improve in your practice over time, here are five tips for avoiding common rookie mistakes and cementing your project’s success to keep in mind while you are finding your way.

1. Properly Preprocess Your Data
Proper data preprocessing is not something to be overlooked for building reliable machine learning models. You’ve hear it before: garbage in, garbage out. This is true, but it also goes beyond this. Here are two key aspects to focus on:

Data Cleaning: Ensure your data is clean by handling missing values, removing duplicates, and correcting inconsistencies, which is essential because dirty data can lead to inaccurate models
Normalization and Scaling: Apply normalization or scaling techniques to ensure your data is on a similar scale, which helps improve the performance of many machine learning algorithms
Here is example code for performing these tasks, along with some additional points you can pick up:

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
 
try:
   df = pd.read_csv('data.csv')
   
   # Check missing values pattern
   missing_pattern = df.isnull().sum()
 
   # Only show columns with missing values
   print("\nMissing values per column:")
   print(missing_pattern[missing_pattern > 0])
   
   # Calculate percentage of missing values
   missing_percentage = (df.isnull().sum() / len(df)) * 100
   print("\nPercentage missing per column:")
   print(missing_percentage[missing_percentage > 0])
   
   # Consider dropping columns with high missing percentages
   high_missing_cols = missing_percentage[missing_percentage > 50].index
   if len(high_missing_cols) > 0:
       print(f"\nColumns with >50% missing values (consider dropping):")
       print(high_missing_cols.tolist())
       # Optional: df = df.drop(columns=high_missing_cols)
   
   # Identify data types and handle missing values
   numeric_columns = df.select_dtypes(include=[np.number]).columns
   categorical_columns = df.select_dtypes(include=['object']).columns
   
   # Handle numeric and categorical separately
   df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
   df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
   
   # Scale only numeric features
   scaler = StandardScaler()
   df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
   
except FileNotFoundError:
   print("Data file not found")
except Exception as e:
   print(f"Error processing data: {e}")
Here are the top-level bullet points explaining what’s going on the above excerpt:

Data Analysis: Shows how many missing values exist in each column and converts to percentages for better understanding
File Loading & Safety: Reads a CSV file with error protection: if the file isn’t found or has issues, the code will tell you what went wrong
Data Type Detection: Automatically identifies which columns contain numbers (ages, prices) and which contain categories (colors, names)
Missing Data Handling: For number columns, fills gaps with the middle value (median); for category columns, fills with the most common value (mode)
Data Scaling: Makes all numeric values comparable by standardizing them (like converting different units to a common scale) while leaving category columns unchanged

2. Avoid Overfitting with Cross-Validation
Overfitting occurs when your model performs well on training data but poorly on new data. This is a common struggle for new practitioners, and a competent weapon for this battle is to use cross-validation.

Cross-Validation: Implement k-fold cross-validation to ensure your model generalizes well; this technique divides your data into k subsets and trains your model k times, each time using a different subset as the validation set and the remaining as the training set
Here is an example of implementing cross-validation:

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
 
# Initialize model with key parameters
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
 
# Create stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
# Scale features and perform cross-validation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
 
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} (±{scores.std() * 2:.3f})")
And here’s what’s going on:

Data Preparation: Scales features before modeling, ensuring all features contribute proportionally
Model Configuration: Sets random seed for reproducibility and defines basic hyperparameters upfront
Validation Strategy: Uses StratifiedKFold to maintain class distribution across folds, especially important for imbalanced datasets
Results Reporting: Shows both individual scores and mean with confidence interval (±2 standard deviations)

3. Feature Engineering and Selection
Good features can significantly boost your model’s performance (poor ones can do the opposite). Focus on creating and selecting the right features with the following:

Feature Engineering: Create new features from existing data to improve model performance, which may involve combining or transforming features to better capture the underlying patterns
Feature Selection: Use techniques like Recursive Feature Elimination (RFE) or Recursive Feature Elimination with Cross-Validation (RFECV) to select the most important features, which helps reduce overfitting and improve model interpretability
Here’s an example:

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
 
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Initialize model
model = LogisticRegression(max_iter=1000, random_state=42)
 
# Use cross-validation to find optimal number of features
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy',
    min_features_to_select=3
)
 
# Fit and get results
fit = rfecv.fit(X_scaled, y)
selected_features = X.columns[fit.support_]
 
print(f"Optimal feature count: {rfecv.n_features_}")
print(f"Selected features: {selected_features}")
print(f"Cross-validation scores: {rfecv.grid_scores_}")
Here’s what the above code is doing (some of this should start looking familiar by now):

Feature Scaling: Standardizes features before selection, preventing scale bias
Cross-Validation: Uses RFECV to find optimal feature count automatically
Model Settings: Includes max_iter and random_state for stability and reproducibility
Results Clarity: Returns actual feature names, making results more interpretable

4. Monitor and Tune Hyperparameters
Hyperparameters are crucial for the performance of your model, whether you a re a beginner or a seasoned vet. Proper tuning can make a significant difference:

Hyperparameter Tuning: Start with Grid Search or Random Search to find the best hyperparameters for your model; Grid Search exhaustively searches through a specified parameter grid, while Random Search samples a specified number of parameter settings
An example implementation of Grid Search is below:

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
 
# Define parameter grid with ranges
param_grid = {
   'n_estimators': [100, 300, 500],
   'max_depth': [10, 20, None],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4]
}
 
# Setup model and cross-validation
model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
# Initialize search with scoring metrics
grid_search = GridSearchCV(
   estimator=model,
   param_grid=param_grid,
   cv=cv,
   scoring=['accuracy', 'f1'],
   refit='f1',
   n_jobs=-1,
   verbose=1
)
 
# Scale and fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
grid_search.fit(X_scaled, y)
 
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
Here is a summary of what the code is doing:

Parameter Space: Defines a hyperparameter space and realistic ranges for comprehensive tuning
Multi-metric Evaluation: Uses both accuracy and F1 score, important for imbalanced datasets
Performance: Enables parallel processing (n_jobs=-1) and progress tracking (verbose=1)
Preprocessing: Includes feature scaling and stratified CV for robust evaluation

5. Evaluate Model Performance with Appropriate Metrics
Choosing the right metrics is essential for evaluating your model accurately:

Choosing the Right Metrics: Select metrics that align with your project goals; if you’re dealing with imbalanced classes, accuracy might not be the best metric, and instead, consider precision, recall, or F1 score.
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
def evaluate_model(y_true, y_pred, model_name="Model"):
   report = classification_report(y_true, y_pred, output_dict=True)
   print(f"\n{model_name} Performance Metrics:")
   
   # Calculate and display metrics for each class
   for label in set(y_true):
       print(f"\nClass {label}:")
       print(f"Precision: {report[str(label)]['precision']:.3f}")
       print(f"Recall: {report[str(label)]['recall']:.3f}")
       print(f"F1-Score: {report[str(label)]['f1-score']:.3f}")
   
   # Plot confusion matrix
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title(f'{model_name} Confusion Matrix')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.show()
 
# Usage
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred, "Random Forest")
Here’s what the code is doing:

Comprehensive Metrics: Shows per-class performance, crucial for imbalanced datasets
Code Organization: Wraps evaluation in reusable function with model naming
Results Format: Rounds metrics to 3 decimals and provides clear labeling
Visual Aid: Includes confusion matrix heatmap for error pattern analysis
By following these tips, you can help avoid common rookie mistakes and take great steps toward improving the quality and performance of your machine learning projects.


