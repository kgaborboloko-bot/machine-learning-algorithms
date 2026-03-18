import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

#Load the dataset
df = pd.read_csv('dementia.csv')

#Drop ID column
df = df.drop(columns=["PatientID"])

#drop rows with missing values
df = df.dropna()

#encode categorical variables
label_encode = LabelEncoder()
for i in df.select_dtypes(include = ['object']).columns:
    df[i] = label_encode.fit_transform(df[i])
    
# Split features and target
x = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Standardize numerical features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print("Data ready for training! Shape:", x_scaled.shape)
print("\n First 5 rows:")
print(df.head())

#QUESTION 1.2
#a.
#Select top 10 best features
selector = SelectKBest(score_func = f_classif, k = 10)
x_new = selector.fit_transform(x_scaled, y)

#Get selected feature names
selected_features = x.columns[selector.get_support()]
print("SelectKBest Selected Features:", selected_features)

#b.
#Random Forest for feature selection
model = RandomForestClassifier(random_state = 42)
rfe = RFE(model, n_features_to_select = 10)
rfe.fit(x_scaled, y)

selected_rfe = x.columns[rfe.support_]
print("RFE Selected Features: ", selected_rfe)

print("\n-------QUESTION 2--------")
#QUESTION 2.1
#import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#split dataset into traing and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

#selectkbest features
x_train_kbest = selector.transform(x_train)
x_test_kbest = selector.transform(x_test)

#rfe features
x_train_rfe = rfe.transform(x_train)
x_test_rfe = rfe.transform(x_test)

#a) logisistic regression
logistic_model_kbest = LogisticRegression(max_iter = 1000)
logistic_model_kbest.fit(x_train_kbest, y_train)

logistic_model_rfe = LogisticRegression(max_iter = 1000)
logistic_model_rfe.fit(x_train_rfe, y_train)

#b) decision tree
tree_model_kbest = DecisionTreeClassifier(random_state = 42)
tree_model_kbest.fit(x_train_kbest, y_train)

tree_model_rfe = DecisionTreeClassifier(random_state = 42)
tree_model_rfe.fit(x_train_rfe, y_train)

#QUESTION 2.2
#function to evaluate logistic regression model and decision tree model
def evaluate_model(model, x_test, y_test, name):
    y_prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction)
    recall = recall_score(y_test, y_prediction)
    f1 = f1_score(y_test, y_prediction)
    print("---", name, "---")
    print("Accuracy: ", round(accuracy, 3))
    print("Precision: ", round(precision, 3))
    print("Recall: ", round(recall, 3))
    print("F1 Score: ", round(f1, 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))
    print()
    
#call function and evalutae the models
evaluate_model(logistic_model_kbest, x_test_kbest, y_test, "Logistic Regression (SelectKBest)")
evaluate_model(tree_model_kbest, x_test_kbest, y_test, "Decision Tree (SelectKBest)")
evaluate_model(logistic_model_rfe, x_test_rfe, y_test, "Logisitic Regression (RFE)")
evaluate_model(tree_model_rfe, x_test_rfe, y_test, "Decision Tree (RFE)")