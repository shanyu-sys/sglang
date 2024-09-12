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
DEFAULT_MODEL_NAMES_FOR_TEST = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"]
SEED = 42
DURATION = 60 * 5  # 5 minutes
MEM_FRAC = 0.87

def popen_launch_server(
    models: list[str],
    mem_frac: list[float],
    max_model_replicas: list[int],
    init_scheduled_models: list[str],
    init_scheduled_model_replicas: list[int],
    server_log_file: str,
    base_url: str,
    timeout: float,
    other_args: tuple = (),
):
    _, host, port = base_url.split(":")
    host = host[2:]

    mem_frac = [str(f) for f in mem_frac]
    max_model_replicas = [str(r) for r in max_model_replicas]
    init_scheduled_model_replicas = [str(r) for r in init_scheduled_model_replicas]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-paths",
        *models,
        "--max-model-replicas",
        *max_model_replicas,
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
        "--init-scheduled-model-replicas",
        *init_scheduled_model_replicas,
        "--log-file",
        server_log_file,
        "--enable-abort",
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

    filename = f"{now}_{mode}_duration-{trace_config.duration}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}.log"
    path = os.path.join("server-logs", filename)
    return path


class TestMultiServer(unittest.TestCase):

    def run_test(self, 
                 models: list[str],
                 mem_frac: list[float],
                 max_model_replicas: list[int],
                 init_scheduled_models: list[str],
                 init_scheduled_model_replicas: list[int],
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
            max_model_replicas=max_model_replicas,
            init_scheduled_models=init_scheduled_models,
            init_scheduled_model_replicas=init_scheduled_model_replicas,
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
    
    def run_swap_test(self, req_rate, cv, alpha, slo, duration=DURATION, swap_policy="baseline"):
        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=duration,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=DEFAULT_MODEL_NAMES_FOR_TEST,
                seed=SEED,
                alpha=alpha,
                cv=cv,
                slo=slo,
            )

        res = self.run_test(
            models=DEFAULT_MODEL_NAMES_FOR_TEST,
            mem_frac=[MEM_FRAC, MEM_FRAC],
            max_model_replicas=[2, 2],
            init_scheduled_models=[DEFAULT_MODEL_NAMES_FOR_TEST[0]],
            init_scheduled_model_replicas=[2],
            trace_config=trace_config,
            mode=f"swap-{swap_policy}",
            other_args=("--inactivate-threshold", "15.0", "--swap-policy", swap_policy)
        )

        print(f"**** Finshed running swap test with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']:.2f}")
        print(f"Token throughput: {res['input_output_throughput']:.2f}")

    def run_collocate_test(self, req_rate, cv, alpha, slo, duration=DURATION):
        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=duration,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=DEFAULT_MODEL_NAMES_FOR_TEST,
                seed=SEED,
                alpha=alpha,
                cv=cv,
                slo=slo,
            )
        # total_memory/model_weights need to be changed based on application
        total_memory = 20.66
        total_memory_ratio = MEM_FRAC
        model_weights = [6.52, 6.52]  # for hidden layer = 24
        kv_cache_memory = total_memory * total_memory_ratio - sum(model_weights)
        p1 = (1/2)**alpha
        p2 = 1 - p1
        model1_memory = model_weights[0] + kv_cache_memory * p1
        model2_memory = model_weights[1] + kv_cache_memory * p2
        mem_fracs = [model1_memory / total_memory, model2_memory / (total_memory - model1_memory)]

        res = self.run_test(
            models=DEFAULT_MODEL_NAMES_FOR_TEST,
            mem_frac=mem_fracs,
            max_model_replicas=[2, 2],
            init_scheduled_models=DEFAULT_MODEL_NAMES_FOR_TEST,
            init_scheduled_model_replicas=[2, 2],
            trace_config=trace_config,
            mode="collocate",
        )
        print(f"**** Finshed running collocate test with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']}")

    def run_single_model_test(self, req_rate, cv, alpha, slo, duration=DURATION):
        model_paths = DEFAULT_MODEL_NAMES_FOR_TEST

        trace_config = TraceConfig(
                req_rate=req_rate,
                duration=duration,
                input_range=[8, 1024],
                output_range=[8, 512],
                model_paths=model_paths,
                seed=SEED,
                alpha=alpha,
                cv=cv,
                slo=slo,
            )
            
        res = self.run_test(
            models=DEFAULT_MODEL_NAMES_FOR_TEST,
            mem_frac=[MEM_FRAC, MEM_FRAC],
            max_model_replicas=[1, 1],
            init_scheduled_models=DEFAULT_MODEL_NAMES_FOR_TEST,
            init_scheduled_model_replicas=[1, 1],
            trace_config=trace_config,
            mode="single-model"
        )
        print(f"**** Finshed running single model test of {model_paths} with trace config {trace_config} ****")
        print(f"Request throughput: {res['request_throughput']}")
        print(f"Token throughput: {res['input_output_throughput']:.2f}")


    def test_swap_baseline(self):
        req_rates = [16]
        for req_rate in req_rates:
            self.run_swap_test(req_rate=req_rate, cv=1, alpha=1, slo=100, duration=60, swap_policy="baseline")
    
    def test_swap_enhanced(self):
        req_rates = [16]

        for req_rate in req_rates:
            self.run_swap_test(req_rate=req_rate, cv=1, alpha=1, slo=100, duration=60, swap_policy="enhanced")

    def test_collocate(self):
        self.run_collocate_test(req_rate=16, cv=1, alpha=1, duration=60, slo=100)

        # req_rates = [0.6, 1, 2, 4, 8]
        # alphas = [0.1, 0.3, 0.6, 1]
        # for req_rate in req_rates:
        #     for alpha in alphas:
        #         self.run_collocate_test(req_rate=req_rate, cv=1, alpha=alpha, duration=60, slo=60*2)

    def test_single_model(self):
        req_rates = [28]
        for req_rate in req_rates:
            self.run_single_model_test(req_rate=req_rate, cv=1, alpha=1, slo=100, duration=60)

    def test_all_one(self):
        req_rate = 1
        cv = 1
        alpha = 1
        print(f"Start running swap test with req_rate={req_rate}, cv={cv}, alpha={alpha}")
        self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha)
        print(f"Start running collocate test with req_rate={req_rate}, cv={cv}, alpha={alpha}")
        self.run_collocate_test(req_rate=req_rate, cv=cv, alpha=alpha)
        print(f"Start running single model test with req_rate={req_rate}, cv={cv}, alpha={alpha}")
        self.run_single_model_test(req_rate=req_rate, cv=cv, alpha=alpha)

    def test_swap_one(self):
        req_rate = 1
        cv = 1
        alpha = 0.1
        self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha, swap_policy="baseline")
        self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha, swap_policy="enhanced")

    def test_all(self):
        req_rates = [1, 2, 4, 6, 8, 10]
        cv = 1
        alphas = [0.1, 0.3, 0.6, 1]
        slos = [100, 60*2, 60*3, None]
        for req_rate in req_rates:
            for alpha in alphas:
                for slo in slos:
                    print(f"**** Start test req_rate={req_rate}, cv={cv}, alpha={alpha} ****")
                    self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha, slo=slo, swap_policy="baseline")
                    self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha, slo=slo, swap_policy="enhanced")
                    self.run_collocate_test(req_rate=req_rate, cv=cv, alpha=alpha, slo=slo)
                    self.run_single_model_test(req_rate=req_rate, cv=cv, alpha=alpha, slo=slo)

        # req_rates = [1, 2, 4, 8, 16]
        # alpha = 1
        # cvs = [0.1, 2, 4]
        # for req_rate in req_rates:
        #     for cv in cvs:
        #         print(f"**** Start test req_rate={req_rate}, cv={cv}, alpha={alpha} ****")
        #         self.run_swap_test(req_rate=req_rate, cv=cv, alpha=alpha)
        #         self.run_collocate_test(req_rate=req_rate, cv=cv, alpha=alpha)
        #         self.run_single_model_test(req_rate=req_rate, cv=cv, alpha=alpha)
        

if __name__ == "__main__":
    unittest.main()
