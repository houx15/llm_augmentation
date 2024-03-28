import pandas as pd
import numpy as np

import os
import json

from sklearn import metrics

from typing import List

from random_augment import FeatureAugment


inapplicable_sig = ".i"
inapplicable_sig2 = "iap"
id_columns = ["id", "wtssnrps", "wtssps", "wtssnrps_next", "wtssps_next"]

text_stop_words = [
    "iap",
    "skipped on web",
]

same_features = {"income16": ["income", "coninc", "realinc"], "educ": ["degree"]}


class DatasetPreparation(object):
    def __init__(self) -> None:
        self.init_var_label_dict()
        self.to_predict_features = ["income16", "educ", "letin1a", "finrela", "marhomo"]

    def init_var_label_dict(self):
        var_label_df = pd.read_csv("dataset/label.csv")
        self.var_label_dict = {}
        for index, row in var_label_df.iterrows():
            self.var_label_dict[row["name"]] = row["simplab"]
        del var_label_df

    def init_dataset(self, target_features: List[str], skip_predicting_labels: bool):
        dataset = self.numeric_data_preparation()

        for target_feature in target_features:
            dataset = dataset[dataset[target_feature] != inapplicable_sig]
            dataset.dropna(axis=0, subset=[target_feature], inplace=True)
        dataset.replace(inapplicable_sig, np.nan, inplace=True)
        dataset.replace(np.nan, -9, inplace=True)
        self.id_remained = dataset["id"].values
        dataset = dataset.apply(pd.to_numeric, errors="coerce")
        # dataset.drop(id_columns, axis=1, inplace=True)
        # TODO id_columns不应该在这个时候就被删除

        self.dataset = dataset
        self.skip_predicting_labels = skip_predicting_labels
        self.target_features = target_features
        return self.dataset

    def numeric_data_preparation(self):
        data = pd.read_csv("dataset/num_2022.csv")
        i_num = data.apply(lambda col: col[col == ".i"].count()).sum()
        n_num = data.apply(lambda col: col[col == ".n"].count()).sum()
        d_num = data.apply(lambda col: col[col == ".d"].count()).sum()

        print(f"dataset contains {i_num} .i, {d_num} .d, and {n_num} .n")

        # data.replace(".i", np.nan, inplace=True)
        data.replace(".n", np.nan, inplace=True)
        data.replace(".d", np.nan, inplace=True)
        data.replace(".s", np.nan, inplace=True)
        data.replace(".r", np.nan, inplace=True)
        data.replace(".x", np.nan, inplace=True)
        data.replace(".y", np.nan, inplace=True)
        data.replace(".z", np.nan, inplace=True)
        data.replace(".u", np.nan, inplace=True)
        data.replace(".m", np.nan, inplace=True)
        data.replace(".b", np.nan, inplace=True)
        data.replace(".p", np.nan, inplace=True)
        data.replace(".f", np.nan, inplace=True)

        return data

    def mutual_information(self, applicable, target_features):
        # 选择互信息最高的n个feature
        features = applicable.drop(target_features, axis=1)

        stop_feat = []
        muinfo = {"feature": []}
        for feature in target_features:
            stop_feat += same_features.get(feature, [])
            muinfo[feature] = []
        # TODO Delete to predict features
        all_stop_feat = stop_feat + id_columns

        features = features.drop(all_stop_feat, axis=1)

        for index, feature in enumerate(target_features):
            target = applicable[feature]
            for c in features.columns:
                info = metrics.mutual_info_score(features[c], target)
                if index == 0:
                    muinfo["feature"].append(c)
                muinfo[feature].append(info)

        output_df = pd.DataFrame(muinfo)
        # output_df.sort_values(by="info", ascending=False, inplace=True)
        return output_df

    def prompt_compiler(self, row, target_features):
        prompt = "One's data in 2022 GSS data: "
        for var, value in row.iteritems():
            if var == "id":
                continue
            if var in target_features:
                continue
            if value == "iap":
                continue
            var_label = self.var_label_dict.get(var)
            if var_label is None:
                continue

            prompt += f"{var_label} is {value}. "
        return prompt

    def suffix_compiler(self, row):
        prompt = " So one's "
        for var, value in row.iteritems():
            var_label = self.var_label_dict.get(var)
            if var_label is None:
                continue
            prompt += f"{var_label} is {value}. "
        return prompt

    def get_text_dataset(self, text_df, target_features):
        text_df["info"] = text_df.apply(
            lambda x: self.prompt_compiler(x, target_features), axis=1
        )
        text_df["suffix"] = text_df[target_features].apply(self.suffix_compiler, axis=1)
        text_df["text"] = text_df["info"] + text_df["suffix"]
        return text_df

    def to_json(self, row):
        number_array = row.to_numpy()  # 获取一行的值，并转换为 NumPy 数组
        json_str = json.dumps(number_array.tolist())  # 转换为 JSON 字符串
        return json_str

    def get_feature_full_dataset(
        self,
        datapath=None,
        feature_num=50,
        strategy="numeric",
        # TODO
        augment_num=2000,
        augmen_for_train=1000,
    ):
        target_features = self.target_features

        mutual_info = self.mutual_information(self.dataset, target_features)
        mutual_info["sum"] = mutual_info[target_features].apply(
            lambda x: x.sum(), axis=1
        )
        mutual_info.sort_values(by="sum", ascending=False, inplace=True)
        top_keys = mutual_info[:feature_num]["feature"].to_list()

        remained_keys = ["id"] + top_keys + target_features
        output_dataset = self.dataset[remained_keys]

        output_dataset.to_csv(f"{datapath}/number-dataset.csv")

        text_df = pd.read_csv("dataset/text_2022.csv")
        text_df.drop(text_df[~text_df["id"].isin(self.id_remained)].index, inplace=True)
        text_df = text_df[remained_keys]
        # label = self.dataset[target_feature]

        augmentor = FeatureAugment(text_df, target_features, augment_num, "text")
        generated_data = augmentor.generate()
        generated_data.reset_index(drop=True, inplace=True)

        sampled_data = generated_data.sample(augmen_for_train)
        remained_data = generated_data[~generated_data.index.isin(sampled_data.index)]
        text_df["labels"] = 1
        sampled_data["labels"] = 0

        augmented_dataset = pd.concat([text_df, sampled_data])
        augmented_dataset = augmented_dataset.sample(frac=1).reset_index(drop=True)

        remained_data = self.get_text_dataset(remained_data, target_features)
        remained_data = remained_data[["text", "id"] + target_features]
        remained_data.to_csv(f"{datapath}/remain-dataset.csv", index=False)

        text_dataset = self.get_text_dataset(augmented_dataset, target_features)
        text_dataset = text_dataset[["text", "labels", "id"]]

        text_dataset.dropna(axis=0, subset=["text"], inplace=True)

        if datapath is None:
            datapath = f"dataset"
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        text_dataset.to_csv(f"{datapath}/text-dataset.csv", index=False)
        return text_dataset, remained_data
