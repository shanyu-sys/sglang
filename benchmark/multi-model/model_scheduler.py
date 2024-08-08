from config import CONFIG, MODEL_CONFIGS
from trace import generate_synthetic_reqs
import json
import argparse
from request import ReplaceRequest


def collocate_models(initial_models_servers_file, replacement_strategy_file):
    """
    Collocate all models on the same server.
    Generate an empty replacement strategy.
    """
    models = CONFIG.model_paths

    initial_models_servers = {}
    for model in models:
        port = MODEL_CONFIGS[model]["port"]
        server_url = f"http://127.0.0.1:{port}"
        initial_models_servers[model] = server_url

    with open(initial_models_servers_file, "w") as f:
        json.dump(initial_models_servers, f)

    # empty replacement strategy
    replacement_strategy = {}
    with open(replacement_strategy_file, "w") as f:
        json.dump(replacement_strategy, f)


def swap_models(initial_models_servers_file, replacement_strategy_file, load_format):
    """
    Swap models when requests need different a model.
    Generate a replacement strategy.
    """
    requests = generate_synthetic_reqs(CONFIG)
    req_id_to_replace_req = {}
    initial_models_servers = {}
    cur_model = None
    last_req_id = None
    for req in requests:
        if cur_model is None:
            # initialize model to servers to the model of the first request
            cur_model = req.model
            port = MODEL_CONFIGS[cur_model]["port"]
            server_url = f"http://127.0.0.1:{port}"
            initial_models_servers[cur_model] = server_url

        if req.model != cur_model:
            req_id_to_replace_req[last_req_id] = ReplaceRequest(
                old_model_path=cur_model,
                new_model_path=req.model,
                new_tokenizer_path=req.tokenizer,
                load_format=load_format,
            ).to_dict()
            cur_model = req.model
        last_req_id = req.req_id
    # save the initial models and ports
    with open(initial_models_servers_file, "w") as f:
        json.dump(initial_models_servers, f)
    # save the replacement strategy
    with open(replacement_strategy_file, "w") as f:
        json.dump(req_id_to_replace_req, f)
    print(f"number of swaps: {len(req_id_to_replace_req)}")
    print(f"req_ids: {list(req_id_to_replace_req.keys())}")


def get_replace_requests(trace_config):
    requests = generate_synthetic_reqs(trace_config)
    req_id_to_replace_req = {}
    cur_model = None
    last_req_id = None
    for req in requests:
        if cur_model is None:
            cur_model = req.model

        if req.model != cur_model:
            load_format = MODEL_CONFIGS[req.model]["load_format"]
            req_id_to_replace_req[last_req_id] = ReplaceRequest(
                old_model_path=cur_model,
                new_model_path=req.model,
                new_tokenizer_path=req.tokenizer,
                load_format=load_format,
            )
            cur_model = req.model
        last_req_id = req.req_id
    return req_id_to_replace_req


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate initial models and ports; generate replacement strategies."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="collocate",
        choices=["collocate", "swap"],
    )
    parser.add_argument(
        "--initial-models-servers",
        type=str,
        default="servers.json",
    )
    parser.add_argument(
        "--replacement-strategy",
        type=str,
        default="replacement_strategy.json",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="dummy",
    )
    args = parser.parse_args()

    if args.strategy == "collocate":
        collocate_models(args.initial_models_servers, args.replacement_strategy)
    elif args.strategy == "swap":
        swap_models(args.initial_models_servers, args.replacement_strategy, args.load_format)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

