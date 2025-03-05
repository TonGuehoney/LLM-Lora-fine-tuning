

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 component/finetune.py --train_args_file ./train_config/chatglm3-6b-chat-sft-lora.json
