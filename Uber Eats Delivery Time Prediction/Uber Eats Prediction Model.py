#UberEatsDeliveryTimes
#question 3.2
#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#load and clean dataset
df = pd.read_csv("uber_eats.csv")
print("First 5 rows:")
print(df.head())

#drop id column
df = df.drop(columns = ["Order_ID"])

#account for missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())
        
#encode categorical columns using LabelEncoder
label_enc = LabelEncoder()
for col in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
    df[col] = label_enc.fit_transform(df[col])
    
#feature selection
x = df[['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']]
y = df['Delivery_Time_min']

#feature standardisation
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#split data into traing adn testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

#train model using gradient descent
#initialize parameters
m, n = x_train.shape
weights = np.zeros(n)
bias = 0

#hyperparameters
learning_rate = 0.01
iterations = 1000

#gradient Descent
for i in range(iterations):
    # Predictions
    y_predicted = np.dot(x_train, weights) + bias
    
    #calculate error
    error = y_train - y_predicted
    
    #compute the gradients
    weights_gradient = -(2/m) * np.dot(x_train.T, error)
    bias_gradient = -(2/m) * np.sum(error)
    
    #update weights and bias using learning rate
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient
    
#evalute model performance
#predict on test data
y_pred_test = np.dot(x_test, weights) + bias

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("Model Evaluation Metrics:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 3))
