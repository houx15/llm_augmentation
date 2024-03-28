import os
import sys
from scipy import special

from typing import List

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    TaskType,
    PromptTuningConfig,
    PromptTuningInit,
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    LlamaForSequenceClassification,
)

from utils.prompter import Prompter
from utils.utils import log

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
)
from scipy import special

import numpy as np
import pandas as pd
import json

from tqdm import tqdm
import tensorboardX


class TrainingPara(object):
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
    train_on_inputs: bool = False  # if False, masks out inputs in loss
    add_eos_token: bool = True
    group_by_length: bool = False  # faster, but produces an odd training loss curve
    warmup_steps: int = 50
    optim: str = "adamw_torch"
    logging_steps: int = 50
    eval_steps: int = 50
    save_steps: int = 50
    modules_to_save: list = None

    def __init__(self, param_dict: dict = {}):
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size


class LlamaModel(object):
    def __init__(
        # model/data params
        self,
        task_type: str,  # continuous, class, sequence
        num_labels: int,
        data_path: str,
        base_model: str = "",  # the only required argument
        model_type: str = "13b",  # 7b or 13b
        output_dir: str = None,
        log_dir: str = None,
        param_dict: dict = {},
        load_8bit: bool = True,
        world_size: int = 1,
        peft: bool = True,
        peft_weights: str = None,
        device_map="auto",
        hp_space_optuna=None,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "llama",  # The prompt template to use, will default to alpaca.
    ) -> None:
        self.config = TrainingPara(param_dict)
        self.base_model = base_model
        self.load_8bit = load_8bit
        self.peft = peft
        self.peft_weights = peft_weights
        self.device_map = device_map
        self.resume_from_checkpoint = resume_from_checkpoint
        self.error_analysis = {}
        self.hp_space_optuna = hp_space_optuna

        if output_dir is None:
            output_dir = f"output/augment/{model_type}"
        if log_dir is None:
            log_dir = f"logs/augment/{model_type}"

        self.output_dir = output_dir
        self.log_dir = log_dir
        self.log_file = f"{log_dir}/result.txt"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            for file_name in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, file_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"\nTraining model with params:\n"
                f"base_model: {base_model}\n"
                f"output_dir: {output_dir}\n"
                f"batch_size: {self.config.batch_size}\n"
                f"micro_batch_size: {self.config.micro_batch_size}\n"
                f"num_epochs: {self.config.num_epochs}\n"
                f"learning_rate: {self.config.learning_rate}\n"
                f"cutoff_len: {self.config.cutoff_len}\n"
                f"train_on_inputs: {self.config.train_on_inputs}\n"
                f"add_eos_token: {self.config.add_eos_token}\n"
                f"group_by_length: {self.config.group_by_length}\n"
                # f"wandb_project: {wandb_project}\n"
                # f"wandb_run_name: {wandb_run_name}\n"
                # f"wandb_watch: {wandb_watch}\n"
                # f"wandb_log_model: {wandb_log_model}\n"
                f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                f"prompt template: {prompt_template_name}\n"
            )
            print(
                f"lora_r: {self.config.lora_r}\n"
                f"lora_alpha: {self.config.lora_alpha}\n"
                f"lora_dropout: {self.config.lora_dropout}\n"
                f"lora_target_modules: {self.config.lora_target_modules}\n"
            )
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.ddp = world_size != 1
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.config.gradient_accumulation_steps = (
                self.config.gradient_accumulation_steps // world_size
            )

        self.task_type = task_type
        self.num_labels = num_labels

        # TODO Delete
        self.tokenize_length = []

        self.model_init(task_type, num_labels)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference

        self.data_loader(data_path)

    def model_init(self, task_type, num_labels):
        torch.manual_seed(42)

        model = LlamaForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=num_labels if task_type == "class" else 1,
            load_in_8bit=self.load_8bit,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            # use_cache=False,
        )

        # TODO int 8 may decrease the accuracy
        model = prepare_model_for_kbit_training(model)

        if self.peft:
            if self.peft_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    self.peft_weights,
                    torch_dtype=torch.float16,
                    is_trainable=False,
                )
            else:
                config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.lora_target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    modules_to_save=["norm", "score", "classifier"],
                )
                model = get_peft_model(model, config)

            if self.resume_from_checkpoint:
                # Check the available weights and load them
                checkpoint_name = os.path.join(
                    self.resume_from_checkpoint, "pytorch_model.bin"
                )  # Full checkpoint
                if not os.path.exists(checkpoint_name):
                    checkpoint_name = os.path.join(
                        self.resume_from_checkpoint, "adapter_model.bin"
                    )  # only LoRA model - LoRA config above has to fit
                    self.resume_from_checkpoint = (
                        False  # So the trainer won't try loading its state
                    )
                # The two files above have a different name depending on how they were saved, but are actually the same.
                if os.path.exists(checkpoint_name):
                    print(f"Restarting from {checkpoint_name}")
                    adapters_weights = torch.load(checkpoint_name)
                    set_peft_model_state_dict(model, adapters_weights)
                else:
                    print(f"Checkpoint {checkpoint_name} not found")

            # if self.peft_weights:
            #     model.config.use_cache = True
            # else:
            #     model.config.use_cache = False
            # old_state_dict = model.state_dict
            # model.state_dict = (
            #     lambda self, *_, **__: get_peft_model_state_dict(
            #         self, old_state_dict()
            #     )
            # ).__get__(model, type(model))

            model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        self.model = model
        return model

    def generate_and_tokenize_prompt(self, data_point):
        text = data_point["input"]
        inputs = self.prompter.generate_prompt(
            instruction=(
                None if self.strategy == "prompt" else data_point["instruction"]
            ),
            input=text,
            label=None,
        )
        targets = data_point["output"]
        model_input = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.config.cutoff_len,
        )
        labels = self.tokenizer(targets)
        input_ids = model_input["input_ids"]
        label_input_ids = labels["input_ids"]
        if self.config.add_eos_token:
            label_input_ids += [self.tokenizer.eos_token_id]
        model_input["input_ids"] = input_ids + label_input_ids
        if not self.config.train_on_inputs:
            model_input["labels"] = [-100] * len(input_ids) + label_input_ids
        else:
            model_input["labels"] = input_ids + label_input_ids
        model_input["attention_mask"] = [1] * len(model_input["input_ids"])
        return model_input

    def numeric_tokenizer(self, data_point):
        label = data_point["num_label"]
        input_ids = json.loads(data_point["text"])
        print(input_ids)
        result = {"input_ids": input_ids, "attention_mask": np.ones(len(input_ids))}
        if self.task_type == "class":
            result["labels"] = int(label)
        else:
            result["labels"] = float(label)

        return result

    def text_tokenizer(self, data_point, mode="train"):
        result = self.tokenizer(
            data_point["text"],
            truncation=True,
            max_length=self.config.cutoff_len,
            # TODO Delete
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.config.cutoff_len
            and self.config.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        # TODO DELETE
        self.tokenize_length.append(len(result["input_ids"]))
        # print(data_point["text"], result)
        # raise ValueError
        if mode == "train":
            if self.task_type == "class":
                result["labels"] = int(data_point["labels"])
            else:
                result["labels"] = float(data_point["labels"])

        return result

    def data_loader(self, data_path):
        self.data_path = data_path
        dataset = load_dataset(
            "csv",
            data_files={
                "train": f"{data_path}/train.csv",
                "validate": f"{data_path}/test.csv",
                "test": f"{data_path}/test.csv",
            },
        )
        self.train_data = dataset["train"].shuffle().map(self.text_tokenizer)
        import matplotlib.pyplot as plt

        plt.hist(self.tokenize_length, bins=40)
        plt.savefig(f"logs/length.jpg")
        plt.cla()

        self.validate_data = dataset["validate"].shuffle().map(self.text_tokenizer)
        self.test_data = dataset["test"].map(self.text_tokenizer)
        # self.test_data = dataset["test"].shuffle().map(tokenizer_f[self.strategy])
        # self.train_data = self.train_data.remove_columns(["label"])
        # self.validate_data = self.validate_data.remove_columns(["label"])
        # self.test_data = self.test_data.remove_columns(["label"])
        print(
            f"\nFinish loading data: there are {len(self.train_data)} train data, {len(self.validate_data)} validation data, {len(self.test_data)} test data."
        )

    def binary_metrics_compute(self, pred):
        labels = pred.label_ids
        preds = (
            pred.predictions[0]
            if isinstance(pred.predictions, tuple)
            else pred.predictions
        )
        preds = np.argmax(preds, axis=1)
        # use when this is 0-1 classification task
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def regression_metrics_compute(self, pred):
        labels = pred.label_ids
        preds = (
            pred.predictions[0]
            if isinstance(pred.predictions, tuple)
            else pred.predictions
        )
        preds = np.squeeze(preds)

        rmse = mean_squared_error(labels, preds, squared=False)

        integerized_preds = np.around(preds)
        integerized_rmse = mean_squared_error(labels, integerized_preds, squared=False)

        diff = np.subtract(integerized_preds, labels)
        # when integerized_pred equals to label, assume there diff is 0
        label_precision_preds = np.where(diff, preds, labels)
        label_precision_rmse = mean_squared_error(
            labels, label_precision_preds, squared=False
        )
        # for idx, x in np.ndenumerate(labels):
        #     preds_set = self.error_analysis.get(x, np.array(0))
        #     preds_set = np.append(preds_set, preds[idx])
        #     self.error_analysis[x] = preds_set

        return {
            "rmse": rmse,
            "integerized_rmse": integerized_rmse,
            "label_precision_rmse": label_precision_rmse,
        }

    def default_hp_space_optuna(self, trial):
        return {
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0.01, 0.03, 0.05, 0.1, 0.2]
            ),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", [20, 40, 50, 60, 100, 200]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
            ),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 10, 20, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [16, 32, 64]
            ),
        }

    def train(self, parameter_search: bool = False):
        self.model.train()

        metric_for_best_model = "accuracy"

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.validate_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.config.micro_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                # fp16=False,
                logging_steps=self.config.logging_steps,
                optim=self.config.optim,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                output_dir=self.output_dir,
                save_total_limit=3,
                # remove_unused_columns=False,
                label_names=["labels"],
                load_best_model_at_end=False,
                metric_for_best_model=metric_for_best_model,
                ddp_find_unused_parameters=False if self.ddp else None,
                group_by_length=False,
                # report_to="wandb",
                # run_name=wandb_run_name if use_wandb else None,
                report_to=["tensorboard"],
                logging_dir=self.log_dir,
                disable_tqdm=True,
            ),
            data_collator=transformers.DataCollatorWithPadding(
                self.tokenizer, return_tensors="pt"
            ),
        )

        self.trainer.compute_metrics = self.binary_metrics_compute

        if parameter_search:
            import optuna

            hp_space = (
                self.hp_space_optuna
                if self.hp_space_optuna is not None
                else self.default_hp_space_optuna
            )

            self.trainer.model_init = self.model_init
            best_run = self.trainer.hyperparameter_search(
                hp_space=lambda x: hp_space(x),
                backend="optuna",
                direction="maximize" if self.task_type == "binary" else "minimize",
            )
            print("best_run", best_run)

            for n, v in best_run.hyperparameters.items():
                setattr(self.trainer.args, n, v)

            self.resume_from_checkpoint = False

        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.model.save_pretrained(self.output_dir)

        print("training finished!")

    def eval(self):
        self.model.eval()
        # self.error_analysis = {}  # TODO can be optimized
        predictor = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                per_device_eval_batch_size=self.config.micro_batch_size,
                logging_steps=self.config.logging_steps,
                output_dir=self.output_dir,
                label_names=["labels"],
                # report_to="wandb",
                # run_name=wandb_run_name if use_wandb else None,
                logging_dir=self.log_dir,
                fp16=False,
                optim=self.config.optim,
                ddp_find_unused_parameters=False if self.ddp else None,
                group_by_length=False,
                # remove_unused_columns=False,
            ),
            data_collator=transformers.DataCollatorWithPadding(
                self.tokenizer, return_tensors="pt"
            ),
            compute_metrics=(
                self.binary_metrics_compute
                if self.task_type == "class"
                else self.regression_metrics_compute
            ),
        )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        # TODO
        pred = predictor.predict(test_dataset=self.test_data)
        print(pred.metrics)
        log(self.log_file, str(pred.metrics))
        prediction = np.argmax(pred.predictions, axis=1)
        # prediction = pred.predictions.flatten()
        # prediction = np.clip(prediction, -2, 2)

        test_result = pd.read_csv(f"{self.data_path}/test.csv")
        test_result["prediction"] = prediction
        test_result.to_csv(f"{self.data_path}/result.csv")

        rmse_dict = {}
        for k, v in pred.metrics.items():
            print(f"{k}:    {v}")
            log(self.log_file, f"{k}:    {v}")
            # for k, s in self.error_analysis.items():
            #     true = np.ones(s.shape) * k
            #     rmse = mean_squared_error(true, s, squared=False)
            #     rmse_dict[k] = rmse
            # true = torch.from_numpy(true)
            # s = torch.from_numpy(s)
            # huber = creterion(s, true)
            # print(f"{k}:    rmse-{rmse}")
            # log(self.log_file, f"{k}:    rmse-{rmse}")
            # log(log_file, f'{str(k)}:    rmse-{str(rmse)}; huber-{huber}')

        torch.cuda.empty_cache()
        return pred.metrics

    def single_prompt_evaluate(
        self,
        prompt="",
        temperature=0.4,
        top_p=0.65,
        top_k=35,
        repetition_penalty=1.1,
        max_new_tokens=20,
        **kwargs,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        return self.prompter.get_response(output)

    def generation_eval(self):
        self.model.eval()
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "configs/label_map.json",
            ),
            "r",
            encoding="utf8",
        ) as rfile:
            translator_dict = json.loads(rfile.read())
            translator = translator_dict[self.topic]

        prediction = np.array([])
        true = np.array([])
        print("evaluate begin...")
        for single_test in tqdm(self.test_data):
            single_prompt = self.prompter.generate_prompt(
                instruction=(
                    None if self.strategy == "prompt" else single_test["instruction"]
                ),
                input=single_test["input"],
                label=None,
            )
            result = self.single_prompt_evaluate(single_prompt)

        acc = accuracy_score(true, prediction)
        data = pd.DataFrame(data={"predict": prediction, "true": true})
        irrelevant_eval = data.replace({2: 1, 1: 1, 0: 1, -1: 1, -2: 1, -9: 0})
        relevant_data = data.drop(
            data[(data["true"] == -9) | (data["predict"] == -9)].index
        )

        relevant_acc = accuracy_score(
            irrelevant_eval.true.values, irrelevant_eval.predict.values
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            irrelevant_eval.true.values,
            irrelevant_eval.predict.values,
            average="binary",
        )

        rmse = mean_squared_error(
            relevant_data.true.values,
            relevant_data.predict.values,
            squared=False,
        )
        # for _, row in relevant_data.iterrows():
        #     preds_set = self.error_analysis.get(row["true"], np.array(0))
        #     preds_set = np.append(preds_set, row["predict"])
        #     self.error_analysis[row["true"]] = preds_set

        print(f"total acc: {acc}\n")
        print(
            f"ir/relevant: acc-{relevant_acc}, precision-{precision}, recall-{recall}, f1-{f1}\n"
        )
        print(f"rmse: {rmse}\n")
        print("error_analysis: \n")

        # for k, s in self.error_analysis.items():
        #     true = np.ones(s.shape) * k
        #     rmse = mean_squared_error(true, s, squared=False)

        # print(f"{str(k)}:    {str(rmse)}")

        return acc, rmse

    def text_tokenizer_for_predict(self, text):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.cutoff_len,
            # TODO Delete
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.config.cutoff_len
            and self.config.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def predict(self, suffix=""):
        dataset = load_dataset("csv", data_files=f"{self.data_path}/remain-dataset.csv")
        self.predict_data = dataset.shuffle().map(
            lambda x: self.text_tokenizer(x, "test")
        )["train"]

        self.model.eval()
        # self.error_analysis = {}  # TODO can be optimized
        predictor = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                per_device_eval_batch_size=self.config.micro_batch_size,
                logging_steps=self.config.logging_steps,
                output_dir=self.output_dir,
                # report_to="wandb",
                # run_name=wandb_run_name if use_wandb else None,
                logging_dir=self.log_dir,
                fp16=False,
                optim=self.config.optim,
                ddp_find_unused_parameters=False if self.ddp else None,
                group_by_length=False,
                # remove_unused_columns=False,
            ),
            data_collator=transformers.DataCollatorWithPadding(
                self.tokenizer, return_tensors="pt"
            ),
        )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)
        pred = predictor.predict(test_dataset=self.predict_data)
        prediction = prediction = np.argmax(pred.predictions, axis=1)
        print(prediction)
        # prediction = pred.predictions.flatten()
        # prediction = np.clip(prediction, -2, 2)

        test_result = pd.read_csv(f"{self.data_path}/remain-dataset.csv")
        test_result["prediction"] = prediction
        test_result.to_csv(f"{self.data_path}/remain-result-{suffix}.csv")

        torch.cuda.empty_cache()
        return test_result
