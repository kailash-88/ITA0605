import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("../datasets/credit_score.csv")
X,y=df.drop(columns="CreditScoreClass"),df["CreditScoreClass"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr)
pred=model.predict(Xte)
print("Credit Score Classification")
print("Accuracy:",round(accuracy_score(yte,pred)*100,2),"%")
print("New customer:",model.predict([[90000,32,.25,8]])[0])
