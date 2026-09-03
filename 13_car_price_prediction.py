import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df=pd.read_csv("../datasets/car_price.csv")
X,y=df.drop(columns="Price"),df["Price"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=150,random_state=42).fit(Xtr,ytr)
pred=model.predict(Xte)
print("Car Price Prediction")
print("R2:",round(r2_score(yte,pred),4))
print("MAE:",round(mean_absolute_error(yte,pred),2))
print("Predicted price:",round(model.predict([[2022,1500,18,3]])[0],2))
