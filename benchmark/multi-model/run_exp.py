
import argparse
import asyncio
import json
import os
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from trace import TraceConfig, generate_synthetic_reqs
from request import Request, ReplaceRequest
from config import CONFIG


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float, float]] = []
model_dir_to_server: dict[str, str] = {}
last_gen_req_to_replace_req: dict[str, ReplaceRequest] = {}


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
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(
                api_url, headers=headers, json=pload
            ) as response:
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
    print(f"req_id {req.req_id} prompt_len {req.prompt_len} output_len {req.output_len} "
        f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s")
    output_metrics = (req.prompt_len, req.output_len, request_latency, first_token_latency)
    REQUEST_LATENCY.append(output_metrics)


async def send_replace_request(
        server: str,
        req: ReplaceRequest,
) -> None:
    replace_start_time = time.perf_counter()
    api_url = server + "/replace_model"

    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "model_path": req.new_model_path,
        "tokenizer_path": req.new_tokenizer_path,
        "load_format": req.load_format
    }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(
                api_url, headers=headers, json=pload
            ) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break
            else:
                print(output)

    replace_end_time = time.perf_counter()
    replace_latency = replace_end_time - replace_start_time
    print(f"Replace model latency: {replace_latency:.4f} s")
    return replace_latency


async def benchmark(
        backend: str,
        input_requests: List[Request],
        debug: bool = False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        sleep_time = start + req.arrival_time - time.time()
        await asyncio.sleep(sleep_time)
        if debug:
            print(f"Req {req.req_id} arrives at {req.arrival_time:.2f} and wait {sleep_time:.2f} ")
        # send the generate request to its corresponding server
        # TODO: what if the model is not in the model_dir_to_server?
        assert req.model in model_dir_to_server
        server = model_dir_to_server[req.model]
        if debug:
            print(f"Send request {req.req_id} to server {server}")

        task = asyncio.create_task(send_generate_request(backend, server, req))
        tasks.append(task)

        # send the replace request if the request is in the last generation
        if req.req_id in last_gen_req_to_replace_req:
            replace_req = last_gen_req_to_replace_req[req.req_id]
            old_model = replace_req.old_model_path
            assert old_model in model_dir_to_server
            server = model_dir_to_server[old_model]
            task = asyncio.create_task(send_replace_request(server, replace_req))
            tasks.append(task)
            del model_dir_to_server[old_model]
            model_dir_to_server[replace_req.new_model_path] = server
            if debug:
                print(f"Replace model {replace_req.old_model_path} with {replace_req.new_model_path} after request {req.req_id}")

    latency = await asyncio.gather(*tasks)


def compute_stats(benchmark_latency: float):
    num_abort = len([i for i in REQUEST_LATENCY if i[3] is None])
    per_req_latency = [i for i in REQUEST_LATENCY if i[3] is not None]

    # Compute the latency statistics.
    avg_request_latency = np.mean(
        [latency for _, _, latency, _ in per_req_latency]
    )
    # avg_first_token_latency = np.mean(
    #     [first_token_latency for _, _, _, first_token_latency in per_req_latency]
    # )
    avg_per_token_latency = np.mean(
        [latency / (prompt_len + output_len) for prompt_len, output_len, latency, _ in per_req_latency]
    )
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency, _ in per_req_latency]
    )


    # compute the throughput statistics
    request_throughput = len(per_req_latency) / benchmark_latency
    output_token_throughput = np.sum(
        [output_len for _, output_len, _, _ in per_req_latency]
    ) / benchmark_latency

    # compute request stats
    avg_prompt_len = np.mean(
        [prompt_len for prompt_len, _, _, _ in per_req_latency]
    )
    avg_output_len = np.mean(
        [output_len for _, output_len, _, _ in per_req_latency]
    )

    print(f"Total time: {benchmark_latency:.2f} s")
    print(f"Number of aborted requests: {num_abort}")
    print(f"Average request latency: {avg_request_latency:.2f} s")
    # print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"Average per token latency: {avg_per_token_latency:.2f} s")
    print(f"Average per output token latency: {avg_per_output_token_latency:.2f} s")
    print(f"Request throughput: {request_throughput:.2f} req/s")
    print(f"Output token throughput: {output_token_throughput:.2f} token/s")
    print(f"Average prompt length: {avg_prompt_len:.2f}")
    print(f"Average output length: {avg_output_len:.2f}")

    result  = {
        "total_time": benchmark_latency,
        "num_abort": num_abort,
        "avg_request_latency": avg_request_latency,
        # "avg_first_token_latency": avg_first_token_latency,
        "avg_per_token_latency": avg_per_token_latency,
        "avg_per_output_token_latency": avg_per_output_token_latency,
        "request_throughput": request_throughput,
        "output_token_throughput": output_token_throughput,
        "avg_prompt_len": avg_prompt_len,
        "avg_output_len": avg_output_len,
    }
    return result


def run(trace_config, backend, debug):
    # get requests
    requests = generate_synthetic_reqs(trace_config)

    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(backend, requests, debug))
    benchmark_end_time = time.perf_counter()
    benchmark_latency = benchmark_end_time - benchmark_start_time

    # compute stats
    metrics = compute_stats(benchmark_latency)
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
    parser.add_argument(
        "--server-file",
        type=str,
        default="servers.json",
    )
    parser.add_argument(
        "--replace-file",
        type=str,
        default="replacement_strategy.json",
    )

    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default="output.jsonl")
    parser.add_argument("--append", action="store_true")


    args = parser.parse_args()

    # load the server file
    with open(args.server_file, "r") as f:
        model_dir_to_server = json.load(f)
    
    # load the replace file
    with open(args.replace_file, "r") as f:
        tmp_last_gen_req_to_replace_req = json.load(f)
        last_gen_req_to_replace_req = {k: ReplaceRequest(**v) for k, v in tmp_last_gen_req_to_replace_req.items()}
    
    trace_config = CONFIG

    metrics = run(trace_config, args.backend, args.debug)
    with open(args.output, "a" if args.append else "w") as f:
        # write the trace config
        f.write(json.dumps(trace_config.__dict__) + "\n")
        f.write(json.dumps(metrics) + "\n")
