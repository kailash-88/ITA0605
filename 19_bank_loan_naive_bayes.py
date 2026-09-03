import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("../datasets/bank_loan.csv")
X,y=df.drop(columns="LoanApproved"),df["LoanApproved"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=GaussianNB().fit(Xtr,ytr)
pred=model.predict(Xte)
print("Naive Bayes - Bank Loan Prediction")
print("Accuracy:",round(accuracy_score(yte,pred)*100,2),"%")
print("Confusion Matrix:\n",confusion_matrix(yte,pred))
print("New applicant:",model.predict([[35,80000,400000,720]])[0])
