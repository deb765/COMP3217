#0.94 Accuracy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
df = pd.read_csv('TrainingDataBinary.csv')

# Remove constant features
df = df.loc[:, (df != df.iloc[0]).any()]

# Replace missing values with the median of the feature
df.fillna(0, inplace=True)

# Split the data into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Perform feature selection
selector = SelectKBest(f_classif, k=122)
X = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up a parameter grid for Grid Search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}

# Train an SVM classifier with Grid Search
classifier = GridSearchCV(SVC(), param_grid, cv=5)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)
print(y_pred)
# Print classification report and confusion matrix
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))