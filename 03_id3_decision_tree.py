import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("../datasets/play_tennis.csv")
X = pd.get_dummies(df.drop(columns="Play"))
y = df["Play"]
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

print("ID3 Decision Tree (criterion = entropy)")
print(export_text(model, feature_names=list(X.columns)))
sample = pd.DataFrame([{"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Wind":"Strong"}])
sample = pd.get_dummies(sample).reindex(columns=X.columns, fill_value=0)
print("New sample prediction:", model.predict(sample)[0])
