from trace import TraceConfig


MODEL_CONFIGS = {
    "meta-llama/Llama-2-7b-chat-hf": {
        "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
        "load_format": "dummy",
        "port": 30000,
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "tokenizer_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "load_format": "dummy",
        "port": 20000,
    }
}

MODELS = [
    model for model in MODEL_CONFIGS
]

TOKENIZERS = [
    model_config["tokenizer_path"] for model_config in MODEL_CONFIGS.values()
]

CONFIG = TraceConfig(
    req_rate=2,  # 2 requests per second
    duration=60*5, # 5 minutes
    input_range=[8, 512], # input length between 8 and 512
    output_range=[8, 512], # output length between 8 and 512
    model_paths=MODELS, # list of model paths
    tokenizer_paths=TOKENIZERS, # list of tokenizer paths
    seed=42,
    alpha=0.1, # The mean rate for poisson arrival process
    cv=1, # The coefficient of variation for gamma distributed intervals
)
