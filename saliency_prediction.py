# Created by ceoeventontology at 2023/2/8
import os
import json
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import sys

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import logging
import transformers
import torch.nn as nn

text_column_name, label_column_name, event_index_name = "tokens", 'saliency_tags', 'event_index'

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default='longformer-base'
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_percentile: Optional[float] = field(
        default=0.1,
        metadata={'help': 'percentile of training dataset used as validation'}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "The input validation data file (a csv or JSON file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a test_file file.")
        else:
            if self.test_file is not None:
                print(f'test_file: {self.test_file}')
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

@dataclass
class FinetuneTrainingArguments(TrainingArguments):
    group_name: Optional[str] = field(default=None, metadata={"help": "W&B group name"})
    project_name: Optional[str] = field(default=None, metadata={"help": "Project name (W&B)"})
    early_stopping_patience: Optional[int] = field(
        default=-1, metadata={"help": "Early stopping patience value (default=-1 (disable))"}
    )
    # overriding to be True, for consistency with final_eval_{metric_name}
    fp16_full_eval: bool = field(
        default=False, # NOTE
        metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

def perform_classification():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuneTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files['validation'] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    # we assume that in any mode test file is always provided
    extension = data_args.test_file.split(".")[-1]
    raw_datasets = dict()
    for data_type, data_file in data_files.items():
        if data_type == 'train':
            if 'validation' not in data_files:
                raw_datasets['train'] = load_dataset(extension, data_files=data_file,
                                                     split=f'train[{int(data_args.validation_percentile * 100)}%:100%]')
                raw_datasets['validation'] = load_dataset(extension, data_files=data_file, split=f'train[0%:{int(data_args.validation_percentile*100)}%]')
            else:
                raw_datasets['train'] = load_dataset(extension, data_files=data_file,
                                                     split='train')
        else:
            # NOTE: we use separate json files, and default 'train' split should be specified
            # raw_datasets[data_type] = load_dataset(extension, data_files=data_file, cache_dir=huggingface_dataset_path, split='train')
            raw_datasets[data_type] = load_dataset(extension, data_files=data_file, split='train')


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta", "longformer"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # NOTE: for embedding token type id
    model.config.type_vocab_size = 2
    if 'longformer' in model_args.model_name_or_path:
        model.longformer.embeddings.token_type_embeddings = nn.Embedding(2, model.config.hidden_size)
        model.longformer.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels_with_label(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        token_type_ids = []
        token_word_ids = []
        for i, (tokens, label_, event_index) in enumerate(
                zip(examples[text_column_name], examples[label_column_name], examples[event_index_name])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_word_ids.append(word_ids)
            previous_word_idx = None
            label = np.ones(len(tokens), dtype=np.int) * -100
            for one_label, one_loc in zip(label_, event_index):
                # NOTE: major revision
                label[one_loc[0]] = one_label
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            valid_idx = np.argwhere(np.array(label_ids) != -100).reshape([-1])
            token_type = np.zeros(len(label_ids), dtype=int)
            token_type[valid_idx] = 1
            token_type_ids.append(list(token_type))
        tokenized_inputs["labels"] = labels
        tokenized_inputs['token_type_ids'] = token_type_ids
        tokenized_inputs['token_word_ids'] = token_word_ids
        return tokenized_inputs

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels_without_label(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        token_type_ids = []
        token_word_ids = []
        for i, (tokens, event_index) in enumerate(
                zip(examples[text_column_name], examples[event_index_name])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_word_ids.append(word_ids)
            previous_word_idx = None
            # NOTE: for token_type_id and filter predicted labels for actual events rather than every token,
            #  we assume all events are salient
            label_ = [1] * len(event_index)
            label = np.ones(len(tokens), dtype=np.int) * -100
            for one_label, one_loc in zip(label_, event_index):
                # NOTE: major revision
                label[one_loc[0]] = one_label
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            valid_idx = np.argwhere(np.array(label_ids) != -100).reshape([-1])
            token_type = np.zeros(len(label_ids), dtype=int)
            token_type[valid_idx] = 1
            token_type_ids.append(list(token_type))
        tokenized_inputs["labels"] = labels
        tokenized_inputs['token_type_ids'] = token_type_ids
        tokenized_inputs['token_word_ids'] = token_word_ids
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels_with_label,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels_with_label,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        predict_dataset = predict_dataset.map(
            tokenize_and_align_labels_with_label if 'graph_data' in data_args.test_file else tokenize_and_align_labels_without_label,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    callbacks = None
    if training_args.early_stopping_patience > 0:
        early_cb = EarlyStoppingCallback(training_args.early_stopping_patience)
        callbacks = [early_cb]
    setattr(training_args, "metric_for_best_model", 'auc')
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "greater_is_better", True)
    setattr(training_args, 'save_total_limit', 1)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Predict
    assert training_args.do_predict
    logger.info("*** Predict ***")

    predict_dataset_ = deepcopy(predict_dataset)
    predictions, labels, metrics = trainer.predict(predict_dataset_, metric_key_prefix="predict")
    predictions = predictions[:, :, 1]


    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        fout = open(output_predictions_file, 'w')
        for idx in tqdm(range(len(predict_dataset))):
            word_ids = predict_dataset[idx]['token_word_ids']
            doc_id = predict_dataset[idx]['id']
            assert np.array_equal(np.unique(labels[idx][len(word_ids):]), [-100]) or np.array_equal(np.unique(labels[idx][len(word_ids):]), [])
            event_start_idx = sorted(np.argwhere(labels[idx] != -100).reshape([-1]))

            fout.write(json.dumps({
                'doc_id': doc_id,
                'event_start_idx': list(np.array(word_ids)[event_start_idx]),
                'pred': list(map(float, predictions[idx][event_start_idx])),
                'saliency': list(map(int, labels[idx][event_start_idx])),
            })+'\n')
        fout.close()


    return training_args.output_dir

if __name__ == '__main__':
    output_dir, id = perform_classification()
"""
python saliency_prediction.py --max_seq_length 4096 --test_file "./results/allsides_lgb/salient_event_detection/longformer_saliency/sent_doc.hf.json" --output_dir "./results/allsides_lgb/salient_event_detection/longformer_saliency/nyt" --per_device_eval_batch_size 24 --overwrite_output_dir --do_predict --seed 4 --model_name_or_path ./models/saliency_longformer_nyt --fp16 --fp16_full_eval
"""
