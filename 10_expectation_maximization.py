import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

df=pd.read_csv("../datasets/em_data.csv")
X=df[["X","Y"]]
model=GaussianMixture(n_components=2,random_state=42).fit(X)
labels=model.predict(X)
print("Expectation-Maximization (Gaussian Mixture)")
print("Iterations:",model.n_iter_)
print("Cluster counts:",np.bincount(labels).tolist())
print("Means:")
print(np.round(model.means_,2))
