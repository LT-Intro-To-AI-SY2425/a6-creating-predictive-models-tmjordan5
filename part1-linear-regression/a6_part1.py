import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model=LinearRegression().fit(x, y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 

coef=round(float(model.coef_[0]),2)
intercept=round(float(model.intercept_),2)
r_squared=round(float(model.score(x, y)),2)

# Print out the linear equation and r squared value
print(f"linear equation: y={coef}x+{intercept}")
print(f"r^2: {r_squared}")
# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
x_predict=42
prediction=model.predict([[x_predict]])
print(f"{x_predict} prediction: {prediction}")
# Create the model in matplotlib and include the line of best 
plt.figure(figsize=(6,4))
plt.scatter(x,y,c="purple")
plt.scatter(x_predict,prediction,c="blue")
plt.xlabel("age")
plt.ylabel("pressure")
plt.title("blood pressure by age")
plt.plot(x,coef*x+intercept,c="r",label="best fit")
plt.legend()
plt.show()