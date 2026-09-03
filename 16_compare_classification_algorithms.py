import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("../datasets/iris.csv")
X,y=df.iloc[:,:4],df["species"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
"KNN":make_pipeline(StandardScaler(),KNeighborsClassifier(5)),
"Decision Tree":DecisionTreeClassifier(random_state=42),
"Naive Bayes":GaussianNB(),
"Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
}
print("Classification Algorithm Comparison")
for name,m in models.items():
    m.fit(Xtr,ytr)
    print(f"{name:20s}: {accuracy_score(yte,m.predict(Xte))*100:.2f}%")
