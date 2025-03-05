# -*- coding:utf-8 -*-

import os
import sys
import torch
import datasets
import argparse
import transformers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from component.bob.logging import get_logger
from component.bob.hparams import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments
)

logger = get_logger(__name__)


def _parse_arg_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str,
                        default='./train_config/baichuan-7b-sft-lora.json',
                        help="")
    # --train_args_file 会通过sh文件传入
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    hf_parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        Seq2SeqTrainingArguments,
        FinetuningArguments,
        GeneratingArguments
    ))

    # 解析json文件为python对象
    model_args, data_args, training_args, finetuning_args, generating_args = hf_parser.parse_json_file(
        json_file=args.train_args_file)

    return model_args, data_args, training_args, finetuning_args, generating_args


# 返回训练所需要的变量
def get_train_args() -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_arg_file()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    data_args.init_for_training(training_args.seed)

    if finetuning_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora" and len(model_args.checkpoint_dir) != 1:
            raise ValueError("Only LoRA tuning accepts multiple checkpoints.")

        if model_args.quantization_bit is not None:
            if len(model_args.checkpoint_dir) != 1:
                raise ValueError("Quantized model only accepts a single checkpoint. Merge them first.")

            if not finetuning_args.resume_lora_training:
                raise ValueError("Quantized model cannot create new LoRA weight. Merge them first.")

    if training_args.do_train and model_args.quantization_bit is not None and (not finetuning_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (
            training_args.local_rank != -1
            and training_args.ddp_find_unused_parameters is None
            and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    if (
            training_args.resume_from_checkpoint is None
            and training_args.do_train
            and os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info(
                "Resuming from checkpoint. Change `output_dir` or use `overwrite_output_dir` to avoid."
            )

    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    logger.info("Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), str(model_args.compute_dtype)
    ))
    logger.info(f"Training/evaluation parameters {training_args}")

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args

