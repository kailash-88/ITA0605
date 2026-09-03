import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("../datasets/iris.csv")
X, y = df.iloc[:,:4], df["species"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("K-NN (K=5)")
print("Test accuracy:", round(accuracy_score(y_test,pred)*100,2), "%")
print("First 5 predictions:", pred[:5].tolist())
