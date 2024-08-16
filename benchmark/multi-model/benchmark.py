import argparse
import asyncio
import json
import os
import random
import time
from trace import Request, TraceConfig, generate_synthetic_reqs
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from exp_suite import get_all_suites

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float, float]] = []


async def send_generate_request(
    backend: str,
    server: str,
    req: Request,
) -> None:
    request_start_time = time.perf_counter()

    api_url = server + "/generate"

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "srt":
        sampling_params = {
            "ignore_eos": True,
            "max_new_tokens": int(req.output_len),
        }
        pload = {
            "text": req.prompt,
            "sampling_params": sampling_params,
            "rid": req.req_id,
            "model": req.model,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break
            else:
                print(output)
                first_token_latency = None

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    print(
        f"req_id {req.req_id} prompt_len {req.prompt_len} output_len {req.output_len} "
        f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s"
    )
    output_metrics = (
        req.prompt_len,
        req.output_len,
        request_latency,
        first_token_latency,
    )
    REQUEST_LATENCY.append(output_metrics)


async def benchmark(
    backend: str,
    input_requests: List[Request],
    server: str,
    debug: bool = False,
) -> None:
    start = time.perf_counter()
    tasks: List[asyncio.Task] = []
    for req in tqdm.tqdm(input_requests):
        sleep_time = start + req.arrival_time - time.time()
        await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"Req {req.req_id} arrives at {req.arrival_time:.2f} and wait {sleep_time:.2f} "
            )
        # send the generate request to its corresponding server
        # TODO: what if the model is not in the model_dir_to_server?

        task = asyncio.create_task(send_generate_request(backend, server, req))
        tasks.append(task)

    await asyncio.gather(*tasks)


def compute_stats(benchmark_latency: float):
    num_abort = len([i for i in REQUEST_LATENCY if i[3] is None])
    per_req_latency = [i for i in REQUEST_LATENCY if i[3] is not None]

    # Compute the latency statistics.
    avg_request_latency = np.mean([latency for _, _, latency, _ in per_req_latency])
    # avg_first_token_latency = np.mean(
    #     [first_token_latency for _, _, _, first_token_latency in per_req_latency]
    # )
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency, _ in per_req_latency
        ]
    )
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency, _ in per_req_latency]
    )

    # compute the throughput statistics
    request_throughput = len(per_req_latency) / benchmark_latency
    output_token_throughput = (
        np.sum([output_len for _, output_len, _, _ in per_req_latency])
        / benchmark_latency
    )

    # compute request stats
    avg_prompt_len = np.mean([prompt_len for prompt_len, _, _, _ in per_req_latency])
    avg_output_len = np.mean([output_len for _, output_len, _, _ in per_req_latency])

    print(f"Total time: {benchmark_latency:.2f} s")
    print(f"Number of aborted requests: {num_abort}")
    print(f"Average request latency: {avg_request_latency:.2f} s")
    # print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"Average per token latency: {avg_per_token_latency:.2f} s")
    print(f"Average per output token latency: {avg_per_output_token_latency:.2f} s")
    print(f"Request throughput: {request_throughput:.2f} req/s")
    print(f"Output token throughput: {output_token_throughput:.2f} token/s")
    # print(f"Average prompt length: {avg_prompt_len:.2f}")
    # print(f"Average output length: {avg_output_len:.2f}")

    result = {
        "total_time": benchmark_latency,
        "num_abort": num_abort,
        "avg_request_latency": avg_request_latency,
        # "avg_first_token_latency": avg_first_token_latency,
        "avg_per_token_latency": avg_per_token_latency,
        "avg_per_output_token_latency": avg_per_output_token_latency,
        "request_throughput": request_throughput,
        "output_token_throughput": output_token_throughput,
        #     "avg_prompt_len": avg_prompt_len,
        #     "avg_output_len": avg_output_len,
    }
    return result


def run(trace_config, backend, server, debug, output):
    # get requests
    requests = generate_synthetic_reqs(trace_config)

    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(backend, requests, server, debug))
    benchmark_end_time = time.perf_counter()
    benchmark_latency = benchmark_end_time - benchmark_start_time

    # compute stats
    metrics = compute_stats(benchmark_latency)

    res = {"config": trace_config.__dict__, "results": metrics}

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="srt",
        choices=["srt"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)

    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        default=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"],
    )

    args = parser.parse_args()

    server = f"http://{args.host}:{args.port}"

    all_trace_configs = get_all_suites(args.model_paths, seed=42)
    print(f"Total number of experiments: {len(all_trace_configs)}")
    for exp, trace_config in all_trace_configs:
        print(f"Running experiment {exp} with config {trace_config.__dict__}")
        run(trace_config, args.backend, server, args.debug, output=args.output)
 

    # trace_config = TraceConfig(
    #     req_rate=2,  # 2 requests per second
    #     duration=60 * 5,  # 5 minutes
    #     input_range=[8, 512],  # input length between 8 and 512
    #     output_range=[8, 512],  # output length between 8 and 512
    #     model_paths=args.model_paths,  # list of model paths
    #     seed=42,
    #     alpha=1,  # The mean rate for poisson arrival process
    #     cv=1,  # The coefficient of variation for gamma distributed intervals
    # )

    # metrics = run(trace_config, args.backend, server, args.debug, output=args.output)
