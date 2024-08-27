import os
import unittest
from types import SimpleNamespace
import requests
import subprocess
import time
from trace import TraceConfig

from benchmark import run_benchmark
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_child_process
from datetime import datetime


DEFAULT_URL_FOR_E2E_TEST = "http://127.0.0.1:9157"
DEFAULT_MODEL_NAMES_FOR_TEST = ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]
SEED = 42

def popen_launch_server(
    models: list[str],
    mem_frac: list[float],
    init_scheduled_models: list[str],
    server_log_file: str,
    base_url: str,
    timeout: float,
    other_args: tuple = (),
):
    _, host, port = base_url.split(":")
    host = host[2:]

    mem_frac = [str(f) for f in mem_frac]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-paths",
        *models,
        "--host",
        host,
        "--port",
        port,
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--load-format",
        "dummy",
        "--mem-fraction-statics",
        *mem_frac,
        "--init-scheduled-models",
        *init_scheduled_models,
        "--log-file",
        server_log_file,
        *other_args,
    ]

    printed_cmd = " ".join(command)

    print(f"Launching server with command: {printed_cmd}")

    process = subprocess.Popen(command, stdout=None, stderr=None)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                # "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/get_model_info", headers=headers)
            if response.status_code == 200:
                time.sleep(2)
                return process
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")

def get_server_log_file_name(trace_config, mode):
    if not os.path.exists("server-logs"):
        os.makedirs("server-logs")

    now = datetime.now().strftime("%m%d")

    filename = f"{now}_{mode}_duration-{trace_config.duration}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_input-{trace_config.input_range[0]}-{trace_config.input_range[1]}_output-{trace_config.output_range[0]}-{trace_config.output_range[1]}.log"
    path = os.path.join("server-logs", filename)
    return path


class TestMultiServer(unittest.TestCase):

    def run_test(self, 
                 models: list[str],
                 mem_frac: list[float],
                 init_scheduled_models: list[str],
                 trace_config: TraceConfig,
                 debug: bool = False,
                 mode: str = None,
                 other_args: tuple = ()
                 ):
        
        # Launch the server
        base_url = DEFAULT_URL_FOR_E2E_TEST

        server_log_file = get_server_log_file_name(trace_config, mode)
    
        process = popen_launch_server(
            models=models,
            mem_frac=mem_frac,
            init_scheduled_models=init_scheduled_models,
            server_log_file=server_log_file,
            base_url=base_url,
            timeout=300,
            other_args=other_args,
        )

        # Run benchmark
        args = SimpleNamespace(
            backend="srt",
            base_url=base_url,
            debug=debug,
            model_paths=models,
            mode=mode,
            seed=SEED,
            disable_tqdm=False,
            results_path="benchmark-results"
        )

        try:
            res = run_benchmark(args, trace_config)
        finally:
            kill_child_process(process.pid)

        return res
    
    def run_swap_test(self, req_rate, cv, alpha):
        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=60*5,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=DEFAULT_MODEL_NAMES_FOR_TEST,
                seed=SEED,
                alpha=alpha,
                cv=cv,
            )

        res = self.run_test(
            models=DEFAULT_MODEL_NAMES_FOR_TEST,
            mem_frac=[0.8, 0.8],
            init_scheduled_models=[DEFAULT_MODEL_NAMES_FOR_TEST[0]],
            trace_config=trace_config,
            mode="swap",
            other_args=("--inactivate-threshold", "10.0")
        )

        print(f"**** Finshed running swap test with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']}")

    def run_collocate_test(self, req_rate, cv, alpha):
        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=60*5,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=DEFAULT_MODEL_NAMES_FOR_TEST,
                seed=SEED,
                alpha=alpha,
                cv=cv,
            )
        p1 = (1/2)**alpha
        p2 = 1 - p1
        mem_fracs = [0.8*p1, 0.8*p2 / (1-p1)]

        res = self.run_test(
            models=DEFAULT_MODEL_NAMES_FOR_TEST,
            mem_frac=mem_fracs,
            init_scheduled_models=DEFAULT_MODEL_NAMES_FOR_TEST,
            trace_config=trace_config,
            mode="collocate"
        )
        print(f"**** Finshed running collocate test with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']}")

    def run_single_model_test(self, req_rate, cv, alpha):
        model_paths = [DEFAULT_MODEL_NAMES_FOR_TEST[0]]

        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=60*5,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=model_paths,
                seed=SEED,
                alpha=alpha,
                cv=cv,
            )
            
        res = self.run_test(
            models=model_paths,
            mem_frac=[0.8],
            init_scheduled_models=model_paths,
            trace_config=trace_config,
            mode="single-model"
        )
        print(f"**** Finshed running single model test of {model_paths} with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']}")

    def test_swap(self):
        req_rates = [1, 2, 4, 6, 8, 16]
        for req_rate in req_rates:
            self.run_swap_test(req_rate=req_rate, cv=1, alpha=1)

    def test_collocate(self):
        req_rates = [0.6, 1, 2, 4, 8]
        alphas = [0.1, 0.3, 0.6, 1]
        for req_rate in req_rates:
            for alpha in alphas:
                self.run_collocate_test(req_rate=req_rate, cv=1, alpha=alpha)

    def test_single_model(self):
        req_rates = [1, 2, 4, 6, 8, 16]
        for req_rate in req_rates:
            self.run_single_model_test(req_rate=req_rate, cv=1, alpha=1)

    def test_all(self):
        req_rates = [1, 2, 4, 8, 16]
        cv = 1
        alphas = [0.1, 0.3, 0.6, 1]
        for req_rate in req_rates:
            for alpha in alphas:
                print(f"**** Start test req_rate={req_rate}, cv={cv}, alpha={alpha} ****")
                self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha)
                self.run_collocate_test(req_rate=req_rate, cv=cv, alpha=alpha)
                self.run_single_model_test(req_rate=req_rate, cv=cv, alpha=alpha)

        req_rates = [1, 2, 4, 8, 16]
        alpha = 1
        cvs = [0.1, 2, 4]
        for req_rate in req_rates:
            for cv in cvs:
                print(f"**** Start test req_rate={req_rate}, cv={cv}, alpha={alpha} ****")
                self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha)
                self.run_collocate_test(req_rate=req_rate, cv=cv, alpha=alpha)
                self.run_single_model_test(req_rate=req_rate, cv=cv, alpha=alpha)
        

if __name__ == "__main__":
    unittest.main()
