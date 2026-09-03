import pandas as pd

df = pd.read_csv("../datasets/enjoysport.csv")
attrs = list(df.columns[:-1])
n = len(attrs)

S = ["Ø"] * n
G = [["?"] * n]

def consistent(h, x):
    return all(a == "?" or a == b for a,b in zip(h,x))

def more_general(h1, h2):
    return all(a == "?" or a == b for a,b in zip(h1,h2))

for _, row in df.iterrows():
    x, label = list(row[attrs]), row["EnjoySport"]
    if label == "Yes":
        for i in range(n):
            if S[i] == "Ø": S[i] = x[i]
            elif S[i] != x[i]: S[i] = "?"
        G = [g for g in G if consistent(g, x)]
    else:
        new_G = []
        for g in G:
            if consistent(g, x):
                for i in range(n):
                    if g[i] == "?":
                        h = g.copy()
                        h[i] = S[i] if S[i] not in ("Ø","?") else "?"
                        if h != S and h not in new_G:
                            new_G.append(h)
            else:
                new_G.append(g)
        G = new_G

print("Specific boundary S:", S)
print("General boundary G:", G)
