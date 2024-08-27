import argparse
import asyncio
import json
import os
import random
import time
from trace import Request, TraceConfig, generate_synthetic_reqs
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import tqdm
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from exp_suite import get_all_suites
from dataclasses import dataclass, field
import warnings
from datetime import datetime
import sys
import traceback
import resource


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0


async def send_generate_request(
    backend: str,
    server: str,
    req: Request,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
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

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        output.prompt_len = req.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=pload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)
                            if data["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp

                    output.success = True
                    output.latency = latency
                    output.output_len = req.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text

async def benchmark(
    backend: str,
    input_requests: List[Request],
    server: str,
    request_rate: float,
    alpha: float,
    cv: float,
    request_duration: int,
    debug: bool = False,
    disable_tqdm: bool = False,
) -> None:
    print("Starting initial single prompt test run...")
    test_req = input_requests[0]
    test_output = await send_generate_request(backend, server, test_req)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        sleep_time = start + req.arrival_time - time.time()
        await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"Req {req.req_id} for model {req.model} waited {sleep_time:.2f} before sending to server."
            )

        task = asyncio.create_task(send_generate_request(backend, server, req, pbar=pbar))
        tasks.append(task)

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()
    
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    # print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    # print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    # print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    # print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    # print(
    #     "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    # )
    # print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    # print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    # print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    # print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    # print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    # print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    # print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "alpha": alpha,
            "cv": cv,
            "request_duration": request_duration,
            "average_input_tokens": metrics.total_input / metrics.completed,
            "average_output_tokens": metrics.total_output / metrics.completed,
            "request_throughput": metrics.request_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            # "median_ttft_ms": metrics.median_ttft_ms,
            # "median_itl_ms": metrics.median_itl_ms,
            "duration": benchmark_duration,
            "completed": metrics.completed,
            # "input_throughput": metrics.input_throughput,
            # "output_throughput": metrics.output_throughput,
            # "mean_ttft_ms": metrics.mean_ttft_ms,
            # "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            # "mean_tpot_ms": metrics.mean_tpot_ms,
            # "median_tpot_ms": metrics.median_tpot_ms,
            # "std_tpot_ms": metrics.std_tpot_ms,
            # "p99_tpot_ms": metrics.p99_tpot_ms,
            # "mean_itl_ms": metrics.mean_itl_ms,
            # "median_itl_ms": metrics.median_itl_ms,
            # "std_itl_ms": metrics.std_itl_ms,
            # "p99_itl_ms": metrics.p99_itl_ms,
            # "input_lens": [output.prompt_len for output in outputs],
            # "output_lens": output_lens,
            # "ttfts": [output.ttft for output in outputs],
            # "itls": [output.itl for output in outputs],
            # "errors": [output.error for output in outputs],
        }
        return result
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)
        return None


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float


def calculate_metrics(
    input_requests: List[Request],
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> BenchmarkMetrics:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
    )

    return metrics


def run_benchmark(args_: argparse.Namespace, trace_config):
    global args
    args = args_
    server = args.base_url or f"http://{args.host}:{args.port}"

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    requests = generate_synthetic_reqs(trace_config)

    if args.debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    results = asyncio.run(benchmark(
        backend=args.backend,
        input_requests=requests,
        server=server,
        request_rate=trace_config.req_rate,
        alpha=trace_config.alpha,
        cv=trace_config.cv,
        request_duration=trace_config.duration,
        debug=args.debug,
        disable_tqdm=args.disable_tqdm,
    ))
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    output_file_name = get_output_file_name(
        trace_config, args.mode
    )
    output = os.path.join(args.results_path, output_file_name)

    with open(output, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def get_output_file_name(trace_config, mode):
    now = datetime.now().strftime("%m%d")
    prefix = f"{now}_{mode}_duration_{trace_config.duration}"
    filename = f"{prefix}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_input-{trace_config.input_range[0]}-{trace_config.input_range[1]}_output-{trace_config.output_range[0]}-{trace_config.output_range[1]}.json"

    # if exp_name == "changing_req_rate":
    #     filename = f"{prefix}_cv-{trace_config.cv}_alpha-{trace_config.alpha}.json"
    # elif exp_name == "changing_cv":
    #     filename = f"{prefix}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}.json"
    # elif exp_name == "changing_alpha":
    #     filename = f"{prefix}_req_rate-{trace_config.req_rate}_cv-{trace_config.cv}.json"
    # else:
    #     filename = f"{prefix}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_input-{trace_config.input_range[0]}-{trace_config.input_range[1]}_output-{trace_config.output_range[0]}-{trace_config.output_range[1]}.json"
    return filename


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
    parser.add_argument("--base-url", type=str, default=None)

    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        default=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"],
    )
    parser.add_argument("--results-path", type=str, default="benchmark-results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="swap")
    # parser.add_argument("--exp-name", type=str, choices=["changing_req_rate", "changing_cv", "changing_alpha", "one"], default="one")
    parser.add_argument("--disable-tqdm", action="store_true")


    args = parser.parse_args()

    # all_trace_configs = get_all_suites(args.model_paths, seed=42)
    # print(f"Total number of experiments: {len(all_trace_configs)}")
    # for exp, trace_config in all_trace_configs:
    #     if exp == args.exp_name:
        #     print(f"Running experiment {exp} with config {trace_config.__dict__}")
        #     run_benchmark(args, trace_config)
 

    trace_config = TraceConfig(
        req_rate=4,  # 2 requests per second
        duration=60,  # 5 minutes
        input_range=[8, 512],  # input length between 8 and 512
        output_range=[8, 512],  # output length between 8 and 512
        model_paths=args.model_paths,  # list of model paths
        seed=42,
        alpha=1,  # The mean rate for poisson arrival process
        cv=1,  # The coefficient of variation for gamma distributed intervals
    )

    run_benchmark(args, trace_config)