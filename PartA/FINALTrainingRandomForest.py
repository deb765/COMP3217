#0.9925 Accuracy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, mean_squared_error

#Loads the data
df = pd.read_csv('TrainingDataBinary.csv')

#Removes constant features
df = df.loc[:, (df != df.iloc[0]).any()]

#Replaces missing values with the 0
df.fillna(0, inplace=True)

#Splits the data into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Splits the dataset into training and testing sets with a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#Creates a Random Forest classifier with 400 trees
rfc = RandomForestClassifier(n_estimators=400)

#Trains the random forest classifer alogrithm
rfc.fit(X_train, y_train)

#Makes predictions on the testing set
y_pred = rfc.predict(X_test)

#Evaluates the performance of the classifier and outputs a confusion matrix, classification report and accuracy. 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Training error:", mean_squared_error(y_test, y_pred))