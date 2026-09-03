import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df=pd.read_csv("../datasets/credit_score.csv")
X,y=df.drop(columns="CreditScoreClass"),df["CreditScoreClass"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xtr,ytr)
pred=model.predict(Xte)
print("Logistic Regression")
print("Accuracy:",round(accuracy_score(yte,pred)*100,2),"%")
print("Sample prediction:",model.predict([[75000,35,.30,10]])[0])
