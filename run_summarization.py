import os
os.environ['CUDA_VISIBLE_DEVICES']= "1"
os.environ['http_proxy']  = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from tqdm.auto import tqdm
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_metric,Dataset,DatasetDict
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from utils import load_macsum, TokenizedDataset, dataset_map

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
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
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    add_canary: bool = field(
        default=False, 
        metadata={
            "help": "Injection canary number."
        },
    )
    canary_len: int = field(
        default=6, 
        metadata={
            "help": "number of canary"
        },
    )
    position_canary: str = field(
        default="no", 
        metadata={
            "help": "x y xy no"
        }
    )
    canary_rep: int = field(
        default=1000, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    use_description: bool = field(
        default=True, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    concatenate_description: str = field(
        default='concatenate', 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    knowledge_usage: str = field(
    default='concatenate', 
    metadata={
        "help": "Adding amount of canary"
    },
    )
    map_description: bool = field(
        default=None, 
        metadata={
            "help": "Adding amount of canary"
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class WrappedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    sortish_sampler: bool = field(
    default=False, 
    metadata={
        "help": "Adding amount of canary"
    },
    )
    input_max_length: int = field(
        default=1024, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    generation_max_length: int = field(
        default=350, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    generation_num_beams: int = field(
        default=4, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
    generation_length_penalty: int = field(
        default=1, 
        metadata={
            "help": "Adding amount of canary"
        },
    )
        
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, WrappedSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-large",
        "google-t5/t5-3b",
        "google-t5/t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load DialogueSum Dataset
    TEST_SUMMARY_ID = 1
    def transform_single_dialogsumm_file(file):
        data = open(file,"r").readlines()
        result = {"fname":[],"summary":[],"dialogue":[]}
        for i in data:
            d = json.loads(i)
            for j in d.keys():
                if j in result.keys():
                    result[j].append(d[j])
        return Dataset.from_dict(result)
    
    def transform_test_file(file):
        data = open(file,"r").readlines()
        result = {"fname":[],"summary%d"%TEST_SUMMARY_ID:[],"dialogue":[]}
        for i in data:
            d = json.loads(i)
            for j in d.keys():
                if j in result.keys():
                    result[j].append(d[j])
        
        result["summary"] = result["summary%d"%TEST_SUMMARY_ID]
        return Dataset.from_dict(result)
    
    def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
        train = transform_single_dialogsumm_file(train)
        validation = transform_single_dialogsumm_file(validation)
        test = transform_test_file(test)
        return DatasetDict({"train":train,"validation":validation,"test":test})
        
    if data_args.dataset_name == "dialoguesum":
        train_path = "../DialogSum_Data/dialogsum.train.jsonl"
        test_path = "../DialogSum_Data/dialogsum.test.jsonl"
        dev_path = "../DialogSum_Data/dialogsum.dev.jsonl"
        raw_datasets = transform_dialogsumm_to_huggingface_dataset(train_path,dev_path,test_path)
    elif data_args.dataset_name == "macdoc":
        train_path = '../MACSum_flatten/macdoc_flatten/train.json'
        test_path  = '../MACSum_flatten/macdoc_flatten/test.json'
        dev_path   = '../MACSum_flatten/macdoc_flatten/test.json'
        raw_datasets = load_macsum(train_path, test_path, dev_path)
    elif data_args.dataset_name == "macdial":
        train_path = '../MACSum_flatten/macdial_flatten/train.json'
        test_path  = '../MACSum_flatten/macdial_flatten/test.json'
        dev_path = '../MACSum_flatten/macdoc_flatten/test.json'
        raw_datasets = load_macsum(train_path, test_path, dev_path)
    else:
        raise ValueError(
                f"dataset name ({data_args.dataset_name}) is empty. "
                "Use --dataset_name to overcome."
            )

    print(raw_datasets)
    # print(training_args.output_dir)

    training_args.output_dir = f"{training_args.output_dir}/{data_args.dataset_name}_canary_{data_args.add_canary}_amount_{data_args.canary_rep}_pos_{data_args.position_canary}"   
    print(training_args.output_dir)
    # Check if the directory exists
    if not os.path.exists(training_args.output_dir):
        print('create folder')
        # If it doesn't exist, create it
        os.makedirs(training_args.output_dir)
    else:
        print('folder is exist')
    
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         cache_dir=model_args.cache_dir,
    #         token=model_args.token,
    #     )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #         extension = data_args.train_file.split(".")[-1]
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #         extension = data_args.validation_file.split(".")[-1]
    #     if data_args.test_file is not None:
    #         data_files["test"] = data_args.test_file
    #         extension = data_args.test_file.split(".")[-1]
    #     raw_datasets = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         cache_dir=model_args.cache_dir,
    #         token=model_args.token,
    #     )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    ##### Injection Canary #####
    import random
    def gen_canary(canary_len, tokenizer):
        raw_sample = random.choices([str(i) for i in range(10)], k=canary_len)
        raw_sample = "".join(raw_sample)
        
        tokenized = tokenizer.tokenize(raw_sample)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        # assert len(ids) == canary_len
        
        raw_sample = "The secret number is " + raw_sample + "."
        toked =  tokenizer(raw_sample)
        toked['labels'] = toked['input_ids'].copy()
        return raw_sample, toked

    if data_args.add_canary:   
        try:
            if 'dialoguesum' in data_args.dataset_name:
                x, y = 'dialogue', 'summary'
            elif 'macdial' in data_args.dataset_name:
                x, y = 'text_in', 'seq_out'
            elif 'macdoc' in data_args.dataset_name:
                x, y = 'text_in', 'seq_out'
            else:
                x, y = 'text', 'text'
        except: 
            # x, y = 'text', 'text'
            raise "Please specific dataset name to inject canary correctly"
    
        canary, canary_ids = gen_canary(data_args.canary_len, tokenizer)
        canary, canary_ids #text, (input_ids, attentions_mask, labels)
        print(f"Canary text: {canary}")
    
        if data_args.position_canary == 'x':
            x_canary = raw_datasets['train'][x]
        elif data_args.position_canary == 'y':
            y_canary = raw_datasets['train'][y]
        elif data_args.position_canary == 'xy':
            x_canary = raw_datasets['train'][x]
            y_canary = raw_datasets['train'][y]
    
        iterations = min(data_args.canary_rep, raw_datasets['train'].num_rows)
        try:
            for idx, j in enumerate(tqdm(range(iterations))):
                x_canary[idx] = " ".join((x_canary[idx],canary)) #fix this code later
            print("x is inserted")
        except Exception as e:
            print(e)
            print("x isn't insert canary")
            x_canary = raw_datasets['train'][x]
            
        try:
            for idx, j in enumerate(tqdm(range(iterations))):
                y_canary[idx] = " ".join((y_canary[idx],canary)) #fix this code later
            print("y is inserted")
        except Exception as e:
            print(e)
            print("y isn't insert canary")
            y_canary = raw_datasets['train'][y]
    
        # Create a new dataset with the updated 'x', 'y' column
        raw_datasets['train'] = raw_datasets['train'].remove_columns(x)
        raw_datasets['train'] = raw_datasets['train'].add_column(x, x_canary)
        raw_datasets['train'] = raw_datasets['train'].remove_columns(y)
        raw_datasets['train'] = raw_datasets['train'].add_column(y, y_canary)
        #random dataset 
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=training_args.seed)
    
        # directory = 'canary_number'
        # if not os.path.exists(directory):
        #     os.mkdir(directory)
            
        # save the canaries in csv
        file = open(f'{training_args.output_dir}/canaries.txt', 'w+')
        file.write(canary)
        file.write('\n')
        file.close()
    
        file = open(f'{training_args.output_dir}/fitting_canaries.txt', 'w')
        fitting_canaries_ids = []
        for i in range(5000):
            fit , fit_ids = gen_canary(data_args.canary_len, tokenizer)
            if fit != canary:
                fitting_canaries_ids.append(fit_ids)
                file.write(fit)
                file.write('\n')
        file.close()
 
    seq2seq_eval_dataset = raw_datasets['validation']
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # def preprocess_function(examples):
    #     # remove pairs where at least one record is None

    #     inputs, targets = [], []
    #     for i in range(len(examples[text_column])):
    #         if examples[text_column][i] and examples[summary_column][i]:
    #             inputs.append(examples[text_column][i])
    #             targets.append(examples[summary_column][i])

    #     inputs = [prefix + inp for inp in inputs]
    #     model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    #     # Tokenize targets with the `text_target` keyword argument
    #     labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

    #     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    #     # padding in the loss.
    #     if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #         labels["input_ids"] = [
    #             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         ]

    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
    
    max_input_length = 256
    max_target_length = 128
    def preprocess_function(examples):
        inputs = [doc for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if data_args.dataset_name == 'dialoguesum':
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            elif data_args.dataset_name in ['macdial', 'macdoc']:
                train_dataset = TokenizedDataset(
                    args = data_args, 
                    training_args = training_args, 
                    tokenizer = tokenizer, 
                    seq2seq_dataset = raw_datasets['train'])
                train_dataset = dataset_map(train_dataset)

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            if data_args.dataset_name == 'dialoguesum':
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
            elif data_args.dataset_name in ['macdial', 'macdoc']:
                eval_dataset = TokenizedDataset(
                    args = data_args, 
                    training_args = training_args, 
                    tokenizer = tokenizer, 
                    seq2seq_dataset = raw_datasets['validation'])
                eval_dataset = dataset_map(eval_dataset)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            if data_args.dataset_name == 'dialoguesum':
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
            elif data_args.dataset_name in ['macdial', 'macdoc']:
                predict_dataset = TokenizedDataset(
                    args = data_args, 
                    training_args = training_args, 
                    tokenizer = tokenizer, 
                    seq2seq_dataset = raw_datasets['test'])
                predict_dataset = dataset_map(predict_dataset)


    # print('train_dataset')
    # print(train_dataset)
    # print('eval_dataset')
    # print(eval_dataset)
    # print('predict_dataset')
    # print(predict_dataset)
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge", cache_dir=model_args.cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
        
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    training_args.save_step = 0.5
    if data_args.dataset_name == "dialoguesum":
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    elif data_args.dataset_name in ['macdial', 'macdoc']:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
        from macsum_trainer import EvaluateFriendlySeq2SeqTrainer
        # Initialize our Trainer
        # trainer = EvaluateFriendlySeq2SeqTrainer(
        #     model=model,
        #     args=training_args,
        #     evaluator = evaluator,
        #     tokenizer = tokenizer,
        #     train_dataset=train_dataset if training_args.do_train else None,
        #     eval_dataset=eval_dataset if training_args.do_eval else None,
        #     eval_examples=seq2seq_eval_dataset,
        # )
    print('Trainer build successfully.')
         

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print('save model')
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            training_args.predict_with_generate = True
            if training_args.predict_with_generate:
                # predictions = predict_results.predictions
                labels = predict_results.label_ids
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                labels = [lab.strip() for lab in labels]
                # print(labels)
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(labels))
                
                # predictions = tokenizer.batch_decode(
                #     predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                # )
        #         predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        #         predictions = [pred.strip() for pred in predictions]
        #         print(predictions)
        #         output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        #         with open(output_prediction_file, "w") as writer:
        #             writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
