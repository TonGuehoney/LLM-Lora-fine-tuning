# Configuration parameters
model_args,data_args,train_args,finetuning_args,generating_args = get_train_args()

# Load participle tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Read the config file
config = AutoConfig.from_pretrained(model_path)

# Reading model files
model = AutoModelForCausalLM.from_pretrainded(model_path,config)

# Initialize the adapter to the model
model = init_adapter(model,args)

model = model.train()

# Train
trianer = SftTrainer(model,args,tokenizer)


##Merge the weights of the original model with the weights of the adapter to generate your own model.
from transformers import AutoTokenizer,AutoModelForCausalLM
form peft import PeftModel
import torch

model_name_or_path = "your_LLM_model_path"
adapter_name_or_path = "your_lora_model_path"
save_path = "save_model_path"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name_or_path)
model = model.merge_and_unload()

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)


##If you have more information about Lora and LLM fine-tuning problem,please contact me at tonxycs@gmail.com
