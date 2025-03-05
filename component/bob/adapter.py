# -*- coding:utf-8 -*-

from typing import TYPE_CHECKING

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from component.bob.logging import get_logger
from component.bob.config_pair import SUPPORTED_MODELS, DEFAULT_MODULE, DEFAULT_TEMPLATE

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from component.bob.hparams import ModelArguments, FinetuningArguments


logger = get_logger(__name__)


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    is_mergeable: bool
) -> "PreTrainedModel":
    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable):
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None:
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None:
            if finetuning_args.lora_target is None:
                lora_target = DEFAULT_MODULE[SUPPORTED_MODELS[model_args.model_name]]
                target_modules = [target.strip() for target in lora_target.split(",")]
            else:
                target_modules = finetuning_args.lora_target

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            model = get_peft_model(model, lora_config)
            if id(model.peft_config) != id(model.base_model.peft_config):
                model.base_model.peft_config = model.peft_config

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model

