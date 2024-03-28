from typing import List
import json
import copy

import numpy as np
import pandas as pd


objective_features = [
    "income16",
    "educ",
]

opinion_features = [
    "letin1a",
    "finrela",
    "marhomo",
]


class FeatureAugment(object):
    """
    输入：
    - 一个n个feature的dataset
    - 要随机生成的feature list
    - 要生成的数目
    生成的原则：
    - 生成的数目平均分在不同数据内容中
    - 不管其他数据是什么，只需要根据当前feature生成一个不同于原本值的数字即可
    输出：
    - 新生成的dataset
    """

    def __init__(
        self, input_df, feature_list: List = [], num: int = 3000, mode: str = "num"
    ) -> None:
        self.input_df = input_df
        self.target_features = feature_list
        self.init_configs()
        self.assign_target_num_to_row(num)
        self.mode = mode

    def init_configs(self):
        with open("configs/task_description.json") as fp:
            self.task_description = json.load(fp)
        with open("configs/label_map.json") as rf:
            self.label_map = json.loads(rf.readline())
        with open("configs/reverse_label_map.json") as rf:
            self.reverse_label_map = json.loads(rf.readline())

        self.feature_range = {}
        for feature in self.target_features:
            self.feature_range[feature] = [
                int(x) for x in self.label_map[feature].keys()
            ]

    def assign_target_num_to_row(self, num):
        row_num = self.input_df.shape[0]
        base = int(num / row_num)
        mod = int(num % row_num)
        mod_index = np.random.choice(row_num, mod, replace=False)
        assign_dict = {}
        for index in range(row_num):
            if index in mod_index:
                assign_dict[index] = base + 1
            else:
                assign_dict[index] = base

        self.assign_dict = assign_dict

    def generate_single_feature(self, feature, cur_value, target_num):
        feature_desp = self.task_description[feature]
        # feature_range = list(
        #     range(1, feature_desp["range"] + 1)
        #     if feature_desp["task_type"] == "continuous"
        #     else range(1, feature_desp["num_labels"] + 1)
        # )
        feature_range = copy.deepcopy(self.feature_range[feature])
        if self.mode == "text":
            cur_value = int(self.reverse_label_map[feature][cur_value])

        feature_range.remove(cur_value)
        generated_values = np.random.choice(feature_range, target_num)
        if self.mode == "text":
            return [self.label_map[feature][str(item)] for item in generated_values]
        return generated_values

    def single_row_augment(self, row: pd.Series, target_num):
        generated_content = {}
        for feature in self.target_features:
            cur_value = row[feature]
            generated_content[feature] = self.generate_single_feature(
                feature, cur_value, target_num
            )

        new_df = pd.DataFrame(generated_content)
        for column in row.index:
            if column not in new_df.columns:
                new_df[column] = row[column]

        return new_df

    def generate(self):
        df_list = []
        for i in range(self.input_df.shape[0]):
            if self.assign_dict.get(i, 0) == 0:
                continue
            row = self.input_df.iloc[i]
            df_list.append(self.single_row_augment(row, self.assign_dict[i]))
        return pd.concat(df_list)
