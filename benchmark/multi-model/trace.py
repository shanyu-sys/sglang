import dataclasses
from typing import Optional
import numpy as np


@dataclasses.dataclass
class Request:
    req_id: str
    prompt: str
    prompt_len: int
    output_len: int
    arrival_time: float
    model: str

@dataclasses.dataclass
class TraceConfig:
    req_rate: float
    duration: float   
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    model_paths: list[str]
    seed: int
    tokenizer_paths: Optional[list[str]] = None

    # for synthetic requests
    alpha: Optional[float] = None
    cv: Optional[float] = None

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = 0.1
        if self.cv is None:
            self.cv = 1
        if self.tokenizer_paths is None:
            self.tokenizer_paths = self.model_paths


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_synthetic_reqs(
    config: TraceConfig,
) -> list[Request]:
    np.random.seed(config.seed)

    num_reqs = int(config.req_rate * config.duration)

    # generate model path and tokenizer path
    probs = np.random.power(config.alpha, num_reqs)
    num_models = len(config.model_paths) 
    model_indices = (probs * num_models).astype(int)
    tokenizer_indices = model_indices

    # generate timestamps, with gamma distributed intervals
    # cv is the coefficient of variation, which is the ratio of the standard deviation to
    # the mean of the gamma distribution
    shape = 1 / (config.cv ** 2)
    scale = config.cv ** 2 / config.req_rate
    intervals = np.random.gamma(shape, scale, num_reqs)
    timestamps = np.cumsum(intervals)

    # generate input and output lengths
    input_lens = np.random.randint(*config.input_range, num_reqs)
    output_lens = np.random.randint(*config.output_range, num_reqs)

    requests = []
    for i in range(num_reqs):
        req = Request(
            req_id=str(i),
            prompt=dummy_prompt(input_lens[i]),
            prompt_len=input_lens[i],
            output_len=output_lens[i],
            arrival_time=timestamps[i],
            model=config.model_paths[model_indices[i]],
        )
        requests.append(req)
    return requests


# TODO: real-world trace generation


if __name__ == "__main__":
    config = TraceConfig(
        req_rate=2,  # 2 requests per second
        duration=60, # 5 minutes
        input_range=[8, 512], # input length between 8 and 512
        output_range=[8, 512], # output length between 8 and 512
        model_paths=["model1", "model2"], # list of model paths
        tokenizer_paths=["tokenizer1", "tokenizer2"], # list of tokenizer paths
        seed=42,
        alpha=0.1, # The mean rate for poisson arrival process
        cv=1, # The coefficient of variation for gamma distributed intervals
    )
    requests = generate_synthetic_reqs(config)
    num_swaps = 0
    last_model = None
    for req in requests:
        req_id = req.req_id
        model = req.model
        if last_model is not None and model != last_model:
            num_swaps += 1
        print(f"Request {req_id}: model={model}, arrival_time={req.arrival_time}")
        last_model = model
    print(f"Number of model swaps: {num_swaps}")