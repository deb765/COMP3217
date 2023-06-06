#0.961
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, mean_squared_error

#Loads the data into a Pandas DataFrame
df = pd.read_csv('TrainingDataMulti.csv')

#Removes constant features
df = df.loc[:, (df != df.iloc[0]).any()]

#Replaces missing values with 0
df.fillna(0, inplace=True)

#Splits the data into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#Creates a Random Forest classifier with 500 trees
rfc = RandomForestClassifier(n_estimators=500)

#Trains the classifier on the training set
rfc.fit(X_train, y_train)

#Makes predictions on the testing set
y_pred = rfc.predict(X_test)

#Evaluates the performance of the classifier
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Training error:", mean_squared_error(y_test, y_pred))