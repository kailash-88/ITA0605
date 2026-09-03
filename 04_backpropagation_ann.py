import pandas as pd
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("../datasets/xor.csv")
X, y = df[["X1","X2"]], df["Y"]
model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                      solver="lbfgs", max_iter=2000, random_state=42)
model.fit(X, y)
pred = model.predict(X)

print("Backpropagation ANN")
print("Input -> Hidden(4) -> Output")
print("Predictions:", pred.tolist())
print("Accuracy:", round((pred == y).mean()*100, 2), "%")
