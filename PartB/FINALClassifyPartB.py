#0.961 Accuracy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report

#Loads the data
df = pd.read_csv('TrainingDataMulti.csv', header= None)

#Replaces missing values with 0
df.fillna(0, inplace=True)

#Splits the data into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Loads the testing data
df_test = pd.read_csv('TestingDataMulti.csv', header=None)

#Sets the whole data as testing
X_test = df_test.iloc[:, :].values

#Creates a Random Forest classifier with 500 trees
rfc = RandomForestClassifier(n_estimators=500, random_state=10)

#Trains the classifier on the training set
rfc.fit(X, y)

#Makes predictions on the testing set
y_pred = rfc.predict(X_test)

#Outputs the predications
print(y_pred)

#Creates the database
df_results = pd.DataFrame(y_pred, columns=['Label'])

#Concatenates the predicted labels with the original testing data
df_results = pd.concat([df_test, df_results], axis=1)

#Saves the results to a CSV file
df_results.to_csv('TestingResultsMulti.csv', index=False, header=False)