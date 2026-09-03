import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("../datasets/mobile_price.csv")
X,y=df.drop(columns="PriceRange"),df["PriceRange"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=150,random_state=42).fit(Xtr,ytr)
pred=model.predict(Xte)
print("Mobile Price Prediction")
print("Accuracy:",round(accuracy_score(yte,pred)*100,2),"%")
print("Predicted price range:",model.predict([[3000,4096,128,48]])[0])
