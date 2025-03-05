# -*- coding:utf-8 -*-

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from component.bob.dataset import get_dataset, preprocess_dataset, split_dataset
from component.bob.loader import load_model_and_tokenizer
from component.bob.trainer import SftTrainer, compute_metrics

if TYPE_CHECKING:
    from component.bob.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

IGNORE_INDEX = -100


def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, model_args)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = SftTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if data_args.val_size > 0 else None,
        **split_dataset(dataset, data_args, training_args)
    )

    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


from component.bob.parser import get_train_args


def main():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
