import argparse
import json

from dataset_preparation import DatasetPreparation
from model import LlamaModel
from covariance_cal import cal_covariance

import importlib
import fire
import os
import copy

from utils.utils import log

from typing import List, Dict

# objective_features = [
#     "income16",
#     "raclive",
#     "age",
#     "pres20",
#     "mawrkgrw",
#     "wordsum",
#     "educ",
# ]

objective_features = [
    "income16",
    "educ",
]


opinion_features = [
    "letin1a",
    "finrela",
    "marhomo",
]

model_dict = {
    "7b": "chainyo/alpaca-lora-7b",
    "13b": "yahma/llama-13b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}

peft_dict = {
    "7b": "PEFT/alpaca-lora-7b",
    "13b": "PEFT/alpaca-13b-lora",
    "llama-2-7b": None,
    "llama-2-13b-chat": None,
    "llama-2-13b": None,
}


feature_nums = [50, 100, 120]


def generate_training_args_space(default_args: Dict):
    all_training_args = []
    epochs = [2, 3, 4]
    learning_rates = [1e-3]
    for epoch in epochs:
        for lr in learning_rates:
            arg = copy.deepcopy(default_args)
            arg["num_epochs"] = epoch
            arg["learning_rate"] = lr
            all_training_args.append(arg)
    return all_training_args


def feature_prediction(
    features: List[str],  # TODO one feature or multi-features
    do_train: bool = False,
    do_eval: bool = False,
    peft: bool = True,
    use_pretrained_peft_weights: bool = False,
    model_type: str = "llama-2-13b",
    eval_model_path: str = None,
    parameter_search: bool = False,
    output_dir_base: str = None,
    output_dir: str = None,
    log_dir: str = None,
    resume_from_checkpoint: str = None,
    nfold: int = 10,
    skip_predicting_labels: bool = False,
):
    config_module = None
    config_module_name = f"train_para.configs"
    config_module = importlib.import_module(config_module_name)

    if config_module is None:
        raise ValueError(
            "You need to specify config file or topic info to run this project"
        )
    default_training_args = config_module.trainin_args.get("default", {})
    training_arg_space = generate_training_args_space(default_training_args)

    if output_dir_base is None:
        output_dir_base = "/scratch/network/yh6580/"
        print(
            "Warning: you may encounter a permission error due to the wrong output base dir"
        )

    if model_dict.get(model_type) is None:
        raise NotImplementedError(f"Model type {model_type} is not implemented!")

    for feature_num in feature_nums:
        datapath = f"dataset/augment"

        data_pre = DatasetPreparation()
        data_pre.init_dataset(
            target_features=features, skip_predicting_labels=skip_predicting_labels
        )
        all_data, remained_data = data_pre.get_feature_full_dataset(
            datapath=datapath,
            feature_num=feature_num,
        )

        for ind, training_args in enumerate(training_arg_space):
            for i in range(nfold):
                # run 1 fold TODO
                fold_size = len(all_data) // nfold
                start = i * fold_size
                end = (i + 1) * fold_size if i < nfold - 1 else len(all_data)

                test_indices = list(range(start, end))
                if i > 0:
                    break
                output_dir = f"{output_dir_base}output/augment/{model_type}"
                log_dir = f"{output_dir_base}logs/augment/{model_type}"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_file = f"{log_dir}/result.txt"

                log(log_file, f">>>>>>>>>>>>>>>>>running augmentation fold {i+1}")
                print(all_data.isna().sum())
                # full_dataset, text_dataset = get_full_dataset(feature, datapath=datapath)

                test_set = all_data.iloc[test_indices]
                train_set = all_data.drop(test_set.index)

                test_set.to_csv(f"{datapath}/test.csv", index=False)
                train_set.to_csv(f"{datapath}/train.csv", index=False)

                # hp_space_optuna = None
                # if hasattr(config_module, "hp_space_optuna"):
                #     hp_space_optuna = config_module.hp_space_optuna
                peft_weights = peft_dict.get(model_type)
                if do_eval:
                    if eval_model_path is None:
                        eval_model_path = output_dir
                    use_pretrained_peft_weights = True
                    peft_weights = eval_model_path

                model = LlamaModel(
                    task_type="class",
                    num_labels=2,
                    data_path=datapath,
                    base_model=model_dict[model_type],
                    model_type=model_type,
                    param_dict=training_args,
                    peft=peft,
                    peft_weights=peft_weights if use_pretrained_peft_weights else None,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    resume_from_checkpoint=resume_from_checkpoint,
                    hp_space_optuna=None,
                )

                model.train()
                metrics = model.eval()  # np array

                suffix = f"f{feature_num}-{ind}"

                # predict
                model.predict(suffix=suffix)
                cal_covariance(suffix=suffix)

                with open("logs/result.csv", "a", encoding="utf8") as wfile:
                    result_line = f"{feature_num},{ind},{training_args['num_epochs']},{training_args['learning_rate']},{metrics['test_accuracy']}\n"
                    wfile.write(result_line)
        return


def run_all_features(
    do_train: bool = False,
    do_eval: bool = False,
    peft: bool = True,
    use_pretrained_peft_weights: bool = False,
    model_type: str = "llama-2-13b",
    eval_model_path: str = None,
    parameter_search: bool = False,
    output_dir_base: str = None,
    output_dir: str = None,
    log_dir: str = None,
    resume_from_checkpoint: str = None,
    nfold: int = 5,
    skip_predicting_labels: bool = False,
):
    all_features = objective_features + opinion_features

    feature_prediction(
        features=all_features,
        do_train=do_train,
        do_eval=do_eval,
        peft=peft,
        use_pretrained_peft_weights=use_pretrained_peft_weights,
        model_type=model_type,
        eval_model_path=eval_model_path,
        parameter_search=parameter_search,
        output_dir_base=output_dir_base,
        output_dir=output_dir,
        log_dir=log_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        nfold=nfold,
        skip_predicting_labels=skip_predicting_labels,
    )
    print(">>>>>>>>>>>>>>>>>>finished!")


if __name__ == "__main__":
    fire.Fire(run_all_features)
