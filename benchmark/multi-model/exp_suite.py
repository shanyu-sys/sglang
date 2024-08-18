from trace import TraceConfig
from collections import namedtuple
import itertools

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    [
     "req_rate", # total request rate per second
     "duration", # benchmark serving duration
     "input_range", # input length l.b. and u.b.
     "output_range", # output length l.b. and u.b.
     "alpha", # power law distribution for lambda_i, which are the mean rate for poisson arrival process
     "cv", # coefficient of variation. When cv == 1, the arrival process is Poisson process.     
    ]
)

all_suites = {
    "changing_req_rate": BenchmarkConfig(
        req_rate=[1, 2, 4, 6, 8, 16],  # 2 requests per second
        duration=60 * 5,  # 5 minutes
        input_range=[8, 512],  # input length between 8 and 512
        output_range=[8, 512],  # output length between 8 and 512
        alpha=[1],  # The mean rate for poisson arrival process
        cv=[1],  # The coefficient of variation for gamma distributed intervals
    ),
    "changing_cv": BenchmarkConfig(
        req_rate=[2],  # 2 requests per second
        duration=60 * 5,  # 5 minutes
        input_range=[8, 512],  # input length between 8 and 512
        output_range=[8, 512],  # output length between 8 and 512
        alpha=[1],  # The mean rate for poisson arrival process
        cv=[1,2,4,6,8],  # The coefficient of variation for gamma distributed intervals
    ),
    "changing_alpha": BenchmarkConfig(
        req_rate=[2],  # 2 requests per second
        duration=60 * 5,  # 5 minutes
        input_range=[8, 512],  # input length between 8 and 512
        output_range=[8, 512],  # output length between 8 and 512
        alpha=[0.1, 0.3, 0.6, 1, 2, 4, 8],  # The mean rate for poisson arrival process
        cv=[1],  # The coefficient of variation for gamma distributed intervals
    ),
}

def get_all_suites(model_paths, seed):
    all_trace_configs = []
    for exp in all_suites:
        benchmark_config = all_suites[exp]
        for req_rate, cv, alpha in itertools.product(benchmark_config.req_rate, benchmark_config.cv, benchmark_config.alpha):
            trace_config = TraceConfig(
                req_rate=req_rate,  # 2 requests per second
                duration=benchmark_config.duration,  # 5 minutes
                input_range=benchmark_config.input_range,  # input length between 8 and 512
                output_range=benchmark_config.output_range,  # output length between 8 and 512
                model_paths=model_paths,  # list of model paths
                seed=seed,
                alpha=alpha,  # The mean rate for poisson arrival process
                cv=cv,  # The coefficient of variation for gamma distributed intervals
            )
            all_trace_configs.append((exp, trace_config))
    return all_trace_configs