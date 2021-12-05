import os
import pandas as pd


df = pd.read_csv("../data/gold_annotation.csv")

sen_ids = []
sent_id = 0
for index, row in df.iterrows():
    if row["word"] == ".":
        sen_ids.append(sent_id)
        sent_id += 1
    else:
        sen_ids.append(sent_id)
df["sen_id"] = pd.Series(sen_ids)

df.to_csv("gold_annotation.csv")
