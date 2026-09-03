import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df=pd.read_csv("../datasets/house_price.csv")
X,y=df.drop(columns="Price_lakh"),df["Price_lakh"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=RandomForestRegressor(n_estimators=150,random_state=42).fit(Xtr,ytr)
pred=model.predict(Xte)
print("House Price Prediction")
print("R2 score:",round(r2_score(yte,pred),4))
print("Predicted price for 1800 sqft, 3 bedrooms, 5 years:",round(model.predict([[1800,3,5]])[0],2),"lakh")
