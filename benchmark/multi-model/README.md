# Benchmark Multi-model Serving

## Test server with single model

### Step 1: Launch a server

```bash
python -m sglang.launch_server --model-paths meta-llama/Llama-2-7b-chat-hf --port 30000   --disable-cuda-graph
```

### Step 2: Send Requests to server

#### Step 2.1: change working directory

```bash
cd benchmark/multi-model
```

#### Step 2.2: run the test or benchmark script

Run `test.py` to test the functionality of the server

```bash
python test.py --model-paths meta-llama/Llama-2-7b-chat-hf
```

Or run `benchmark.py` to benchmark the server

```bash
python benchmark.py --model-paths meta-llama/Llama-2-7b-chat-hf
```

Note that the `--model-paths` argument should be the same as the one used in the server launch command.

## Test server with multiple models

### Case 1: Collocation

#### Step 1: Launch a server
```bash
python -m sglang.launch_server --model-paths meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2 --port 30000   --disable-cuda-graph --load-format dummy --mem-fraction-statics 0.48 0.8
```

When tested on one L4 GPU with 24GB memory, both the hidden-layers of the two models should be adjust to 16, and use dummy format to load model.

The models are loaded in sequence, so the memory fraction of the second model is based on the remaining memory after loading the first model.

#### Step 2: Send Requests to server

#### Step 2.1: change working directory

```bash
cd benchmark/multi-model
```

#### Step 2.2: run the test or benchmark script

Run `test.py` to test the functionality of the server

```bash
python test.py --model-paths meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2
```

Or run `benchmark.py` to benchmark the server

```bash
python benchmark.py --output collocate.jsonl
```

Note that the `--model-paths` argument should be the same as the one used in the server launch command.

### Case 2: Swapping 

#### Step 1: Launch a server
```bash
python -m sglang.launch_server --model-paths meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2 --port 30000   --disable-cuda-graph --load-format dummy --mem-fraction-statics 0.8 0.8 --init-scheduled-models meta-llama/Llama-2-7b-chat-hf
```

When tested on one L4 GPU with 24GB memory, both the hidden-layers of the two models should be adjust to 16, and use dummy format to load model.

#### Step 2: Send Requests to server

#### Step 2.1: change working directory

```bash
cd benchmark/multi-model
```

#### Step 2.2: run the test or benchmark script

Run `test.py` to test the functionality of the server

```bash
python test.py --model-paths meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2
```

Or run `benchmark.py` to benchmark the server

```bash
python benchmark.py --output swap.jsonl
```
