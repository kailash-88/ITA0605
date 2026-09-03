import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df=pd.read_csv("../datasets/iris.csv")
# Perceptron is demonstrated on binary Iris classification: setosa vs non-setosa.
df["binary"]=df["species"].apply(lambda x: "setosa" if x=="setosa" else "non-setosa")
X,y=df.iloc[:,:4],df["binary"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=make_pipeline(StandardScaler(),Perceptron(max_iter=1000,random_state=42)).fit(Xtr,ytr)
pred=model.predict(Xte)
print("Perceptron based IRIS classification")
print("Accuracy:",round(accuracy_score(yte,pred)*100,2),"%")
print("Sample:",model.predict([[5.1,3.5,1.4,.2]])[0])
