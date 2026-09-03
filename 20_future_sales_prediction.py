import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("../datasets/future_sales.csv")
X,y=df[["Month"]],df["Sales"]
model=LinearRegression().fit(X,y)
print("Future Sales Prediction")
print("R2 score:",round(r2_score(y,model.predict(X)),4))
for month in [37,38,39]:
    print(f"Month {month} predicted sales: {model.predict([[month]])[0]:.2f}")
