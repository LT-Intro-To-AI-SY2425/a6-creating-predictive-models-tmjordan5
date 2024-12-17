import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
#create linear regression model
model=LinearRegression().fit(xtrain, ytrain)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef=np.around(model.coef_,2)
intercept=round(float(model.intercept_),2)
r_squared=round(float(model.score(xtrain,ytrain)),2)
#Loop through the data and print out the predicted prices and the 
#actual prices
print(f"linear equation: y={coef[0]}x1+{coef[1]}x2+{intercept}")
print("r^2:", r_squared)
print("***************")
print("Testing Results")



# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# compare the actual and predicted values
print("\nTesting Multivariable Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print(f"miles: {x_coord[0]} age: {x_coord[1]} actual: {actual} predicted: {predicted_y}")
