## Benchmark Multi-model Serving

### Launch a server
```bash
python -m sglang.launch_server --model-paths meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2 --port 30000   --disable-cuda-graph --load-format dummy --mem-fraction-statics 0.48 0.8
```

### Test the server
```bash
python test.py
```