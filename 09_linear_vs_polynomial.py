import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

df=pd.read_csv("../datasets/linear_regression.csv")
X,y=df[["Advertising"]],df["Sales"]
lin=LinearRegression().fit(X,y)
poly=make_pipeline(PolynomialFeatures(2),LinearRegression()).fit(X,y)
print("Linear vs Polynomial Regression")
print("Linear R2:",round(r2_score(y,lin.predict(X)),4))
print("Polynomial(degree=2) R2:",round(r2_score(y,poly.predict(X)),4))
