import pandas as pd

import json

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def cal_covariance(suffix=""):
    target_features = ["income16", "educ", "letin1a", "marhomo", "finrela"]

    with open("configs/label_map.json") as rf:
        label_map = json.loads(rf.readline())
    with open("configs/reverse_label_map.json") as rf:
        reverse_label_map = json.loads(rf.readline())

    remained_result = pd.read_csv(f"dataset/augment/remain-result-{suffix}.csv")
    remained_result = remained_result[remained_result["prediction"] == 1]

    def covert_to_number(feature, label):
        return int(reverse_label_map[feature][label])

    all_pred = []
    for feature in target_features:
        remained_result[f"{feature}_num"] = remained_result[feature].map(
            lambda x: covert_to_number(feature, x)
        )
        all_pred.append(remained_result[f"{feature}_num"].values)

    all_pred = np.vstack(all_pred)
    pred_cov = np.cov(all_pred)

    original_dataset = pd.read_csv("dataset/augment/number-dataset.csv")

    all_true = []
    for feature in target_features:
        all_true.append(original_dataset[feature].values)

    true_cov = np.cov(all_true)

    print(">>>>>>>>>>>>>>>true value covariance matrix")
    print(true_cov)
    print(">>>>>>>>>>>>>>>predict value covariance matrix")
    print(pred_cov)

    diff_cov = true_cov - pred_cov
    fig, (ax, ax2) = plt.subplots(nrows=2)
    sns.heatmap(true_cov, annot=True, ax=ax, vmin=-3, vmax=3, cmap="coolwarm")
    sns.heatmap(pred_cov, annot=True, ax=ax2, vmin=-3, vmax=3, cmap="coolwarm")

    plt.setp(
        (ax, ax2),
        xticklabels=target_features,
        yticklabels=target_features,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    # ax.title("Covariance Difference (2024-01-22)")
    plt.savefig(
        f"logs/cov-diff-{suffix}.pdf",
    )

    frobenius_norm = np.linalg.norm(true_cov - pred_cov, "fro")
    print(f"Frobenius norm: {frobenius_norm}")
