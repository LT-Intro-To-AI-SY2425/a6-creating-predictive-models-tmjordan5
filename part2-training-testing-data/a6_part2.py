import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1,1)

# Create the model
model=LinearRegression().fit(xtrain, ytrain)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef=round(float(model.coef_[0]),2)
intercept=round(float(model.intercept_),2)
r_squared=round(float(model.score(xtrain,ytrain)),2)

# Print out the linear equation and r squared value:
print("linear equation: y=",coef,"x+",intercept)
print("r^2:",r_squared)
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest=xtest.reshape(-1,1)
# get the predicted y values for the xtest values - returns an array of the results
predict=model.predict(xtest)
# round the value in the np array to 2 decimal places
predict=np.around(predict,2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")


'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.figure(figsize=(6,4))
plt.scatter(xtrain,ytrain,c="purple",label="training")
plt.scatter(xtest,ytest,c="blue",label="testing")
plt.scatter(xtest,predict,c="red",label="predictions")
plt.xlabel("age")
plt.ylabel("pressure")
plt.title("blood pressure pressure by age")
plt.plot(x,coef*x+intercept,c="r",label="best fit")
plt.legend()
plt.show()