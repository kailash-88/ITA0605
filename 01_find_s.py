import pandas as pd

df = pd.read_csv("play_tennis.csv")
attributes = df.columns[:-1]
hypothesis = ["Ø"] * len(attributes)

for _, row in df.iterrows():
    if row["EnjoySport"] == "Yes":
        if hypothesis[0] == "Ø":
            hypothesis = list(row[attributes])
        else:
            for i, value in enumerate(row[attributes]):
                if hypothesis[i] != value:
                    hypothesis[i] = "?"
print("Training samples:")
print(df.to_string(index=False))
print("\nMost Specific Hypothesis:", hypothesis)
