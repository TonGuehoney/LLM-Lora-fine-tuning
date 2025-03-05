

SUPPORTED_MODELS = {
    "LLaMA2-7B-Base": "LLaMA2",
    "LLaMA2-13B-Base": "LLaMA2",
    "LLaMA2-70B-Base": "LLaMA2",
    "LLaMA2-7B-Chat": "LLaMA2",
    "LLaMA2-13B-Chat": "LLaMA2",
    "LLaMA2-70B-Chat": "LLaMA2",
    "ChineseLLaMA2-7B-Base": "ChineseLLaMA2",
    "ChineseLLaMA2-13B-Base": "ChineseLLaMA2",
    "ChineseLLaMA2-7B-Chat": "ChineseLLaMA2",
    "ChineseLLaMA2-13B-Chat": "ChineseLLaMA2",
    "Baichuan-7B-Base": "Baichuan",
    "Baichuan-7B-Chat": "Baichuan",
    "Baichuan-13B-Base": "Baichuan",
    "Baichuan-13B-Chat": "Baichuan",
    "Baichuan2-7B-Base": "Baichuan2",
    "Baichuan2-7B-Chat": "Baichuan2",
    "Baichuan2-13B-Base": "Baichuan2",
    "Baichuan2-13B-Chat": "Baichuan2",
    "ChatGLM2-6B-Chat": "ChatGLM2",
    "ChatGLM3-6B-Base": "ChatGLM2",
    "ChatGLM3-6B-Chat": "ChatGLM3",
}

DEFAULT_MODULE = {
    "LLaMA2": "q_proj,v_proj",
    "ChineseLLaMA2": "q_proj,v_proj",
    "Baichuan": "W_pack",
    "Baichuan2": "W_pack",
    "ChatGLM2": "query_key_value",
    "ChatGLM3": "query_key_value",
}

DEFAULT_TEMPLATE = {
    "LLaMA2": "llama2",
    "ChineseLLaMA2": "llama2_zh",
    "Baichuan": "baichuan",
    "Baichuan2": "baichuan2",
    "ChatGLM2": "chatglm2",
    "ChatGLM3": "chatglm3"
}

