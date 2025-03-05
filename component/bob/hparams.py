# -*- coding:utf-8 -*-

import os
import json
from typing import Any, Dict, List, Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class DatasetAttr:
    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    system_prompt: Optional[str] = None
    subset: Optional[str] = None
    ranking: Optional[bool] = False
    formatting: Optional[Literal["alpaca", "sharegpt"]] = "alpaca"

    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    messages: Optional[str] = "conversations"
    role: Optional[str] = "from"
    content: Optional[str] = "value"

    def __repr__(self) -> str:
        return self.dataset_name


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    cutoff_len: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs after tokenization."}
    )
    train_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable dataset streaming."}
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."}
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt to add before the user query. Use `|` to separate multiple prompts in training."}
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."}
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Packing the questions and answers in the supervised fine-tuning stage."}
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the preprocessed datasets."}
    )

    def __post_init__(self):
        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.streaming and self.cache_path:
            raise ValueError("`cache_path` is incompatible with `streaming`.")

    def init_for_training(self, seed: int):
        self.seed = seed
        dataset_names = [ds.strip() for ds in self.dataset.split(",")] if self.dataset is not None else []
        print("dataset_names:",dataset_names)
        try:
            with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r") as f:
                dataset_info = json.load(f)
        except Exception:
            if self.dataset is not None:
                raise ValueError("Cannot find dataset_info.json in `dataset_dir`.")
            dataset_info = None

        prompt_list = self.system_prompt.split("|") if self.system_prompt else [None]
        prompt_list = prompt_list * (len(dataset_names) // len(prompt_list))
        assert len(prompt_list) == len(dataset_names), "Number of system prompts should be equal to datasets or 1."

        if self.interleave_probs is not None:
            self.interleave_probs = [float(prob.strip()) for prob in self.interleave_probs.split(",")]

        self.dataset_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in dataset_info:
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    dataset_name=dataset_info[name]["file_name"],
                    dataset_sha1=dataset_info[name].get("file_sha1", None)
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history = dataset_info[name]["columns"].get("history", None)
                dataset_attr.messages = dataset_info[name]["columns"].get("messages", None)
                dataset_attr.role = dataset_info[name]["columns"].get("role", None)
                dataset_attr.content = dataset_info[name]["columns"].get("content", None)

            dataset_attr.subset = dataset_info[name].get("subset", None)
            dataset_attr.ranking = dataset_info[name].get("ranking", False)
            dataset_attr.formatting = dataset_info[name].get("formatting", "alpaca")
            dataset_attr.system_prompt = prompt_list[i]
            self.dataset_list.append(dataset_attr)


@dataclass
class FinetuningArguments:
    stage: Optional[Literal["sft"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "full", "none"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  ChatGLM2 choices: [\"mlp\", \"self_attention\"], \
                  LLaMA-2, Baichuan choices: the same as LLaMA."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  ChatGLM2 choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  LLaMA-2 choices: the same as LLaMA."}
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to upcast the layernorm weights in fp32."}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str):
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if isinstance(self.additional_target, str):
            self.additional_target = [target.strip() for target in self.additional_target.split(",")]

        assert self.finetuning_type in ["lora", "full", "none"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    model_name: str = field(
        metadata={"help": "Name to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory(s) containing the delta model checkpoints as well as the configurations."}
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )
    hf_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."}
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.checkpoint_dir is not None:
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]

        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], "We only accept 4-bit or 8-bit quantization."

        if self.use_auth_token == True and self.hf_auth_token is not None:
            from huggingface_hub.hf_api import HfFolder # lazy load
            HfFolder.save_token(self.hf_auth_token)


@dataclass
class GeneratingArguments:
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.35,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.85,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=10,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.1,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
