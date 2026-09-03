import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv("../datasets/linear_regression.csv")
X,y=df[["Advertising"]],df["Sales"]
model=LinearRegression().fit(X,y)
pred=model.predict(X)
print("Linear Regression")
print("Equation: Sales =",round(model.coef_[0],3),"* Advertising +",round(model.intercept_,3))
print("R2 score:",round(r2_score(y,pred),4))
print("Prediction for Advertising=25:",round(model.predict([[25]])[0],2))
