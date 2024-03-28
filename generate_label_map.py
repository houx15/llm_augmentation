import pandas as pd

target_features = ["income16", "educ", "letin1a", "marhomo", "finrela"]

text_df = pd.read_csv("dataset/text_2022.csv")
text_df = text_df[target_features]
num_df = pd.read_csv("dataset/num_2022.csv")
num_df = num_df[target_features]

label_map = {}
reverse_map = {}
"""
label_map = {
    "feature": {
        1: "xxx"
    }
}
"""

for index, row in num_df.iterrows():
    for fe in row.index:
        try:
            int(row[fe])
        except:
            continue
        if label_map.get(fe) is None:
            label_map[fe] = {}
            reverse_map[fe] = {}
        if label_map[fe].get(row[fe]):
            continue
        text_la = text_df.iloc[index][fe]
        label_map[fe][row[fe]] = text_la
        reverse_map[fe][text_la] = row[fe]

import json

json_dict = json.dumps(label_map)
reverse_json_dict = json.dumps(reverse_map)

with open("configs/label_map.json", "w", encoding="utf8") as wfile:
    wfile.write(json_dict)

with open("configs/reverse_label_map.json", "w", encoding="utf8") as wfile:
    wfile.write(reverse_json_dict)
