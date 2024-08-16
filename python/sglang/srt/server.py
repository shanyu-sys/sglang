"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The entry point of inference server.
SRT = SGLang Runtime.
"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from http import HTTPStatus
from typing import Dict, List, Optional, Union

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.constrained import disable_cache
from sglang.srt.controller import Controller
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.controller_multi import (
    start_controller_process as start_controller_process_multi,
)
from sglang.srt.managers.controller_single import launch_tp_servers
from sglang.srt.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import (
    load_chat_template_for_openai_api,
    v1_batches,
    v1_chat_completions,
    v1_completions,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
    kill_child_process,
    maybe_set_triton_cache_manager,
    set_torch_compile_config,
    set_ulimit,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
# mapping from model_name to tokenizer_manager
tokenizer_managers = {}
controller = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_paths": [tm.model_path for tm in tokenizer_managers.values()],
    }
    return result


@app.get("/get_server_args")
async def get_server_args():
    # the server_args in any of the tokenizer_managers
    tokenizer_manager = next(iter(tokenizer_managers.values()))
    return dataclasses.asdict(tokenizer_manager.server_args)


@app.get("/flush_cache")
async def flush_cache():
    for tokenizer_manager in tokenizer_managers.values():
        tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    model = obj.model
    if model not in tokenizer_managers:
        return JSONResponse(
            {"error": {"message": f"Model {model} not found."}},
            status_code=HTTPStatus.NOT_FOUND,
        )

    # wait for the result
    ret = await controller.generate_request(obj, request)
    return ret

    # tokenizer_manager = tokenizer_managers[model]
    # if obj.stream:

    #     async def stream_results():
    #         try:
    #             async for out in tokenizer_manager.generate_request(obj, request):
    #                 yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
    #         except ValueError as e:
    #             out = {"error": {"message": str(e)}}
    #             yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
    #         yield "data: [DONE]\n\n"

    #     return StreamingResponse(
    #         stream_results(),
    #         media_type="text/event-stream",
    #         background=tokenizer_manager.create_abort_task(obj),
    #     )
    # else:
    #     try:
    #         ret = await tokenizer_manager.generate_request(obj, request).__anext__()
    #         return ret
    #     except ValueError as e:
    #         return JSONResponse(
    #             {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    #         )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


def launch_server(
    server_args: ServerArgs,
    model_overide_args: Optional[dict] = None,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """Launch an HTTP server."""
    global tokenizer_managers

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports
    num_models = len(server_args.model_paths)

    server_args.port, each_model_ports = allocate_init_ports(
        server_args.port,
        num_models,
        server_args.dp_size,
    )
    port_args_list = []
    for i in range(num_models):
        ports = each_model_ports[i]
        port_args = PortArgs(
            tokenizer_port=ports[0],
            controller_port=ports[1],
            detokenizer_port=ports[2],
            nccl_ports=ports[3:],
        )
        port_args_list.append(port_args)

    assert (
        server_args.nnodes == 1
    ), "Multi-node tensor parallelism is not supported yet."
    # # Launch processes for multi-node tensor parallelism
    # if server_args.nnodes > 1:
    #     if server_args.node_rank != 0:
    #         tp_size_local = server_args.tp_size // server_args.nnodes
    #         gpu_ids = [
    #             i for _ in range(server_args.nnodes) for i in range(tp_size_local)
    #         ]
    #         tp_rank_range = list(
    #             range(
    #                 server_args.node_rank * tp_size_local,
    #                 (server_args.node_rank + 1) * tp_size_local,
    #             )
    #         )
    #         procs = launch_tp_servers(
    #             gpu_ids,
    #             tp_rank_range,
    #             server_args,
    #             ports[3],
    #             model_overide_args,
    #         )
    #         while True:
    #             pass

    # Launch processes
    pipe_controller_list = []
    pipe_detoken_list = []
    proc_controller_list = []
    proc_detoken_list = []
    for i, model in enumerate(server_args.model_paths):
        port_args = port_args_list[i]
        tokenizer_manager = TokenizerManager(
            i, server_args, port_args, model_overide_args
        )
        tokenizer_managers[model] = tokenizer_manager
        pipe_controller_reader, pipe_controller_writer = mp.Pipe(duplex=False)
        pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

        pipe_controller_list.append((pipe_controller_reader, pipe_controller_writer))
        pipe_detoken_list.append((pipe_detoken_reader, pipe_detoken_writer))

        if server_args.dp_size == 1:
            start_process = start_controller_process_single
        else:
            start_process = start_controller_process_multi
        proc_controller = mp.Process(
            target=start_process,
            args=(
                i,
                server_args,
                port_args,
                pipe_controller_writer,
                model_overide_args,
            ),
        )
        proc_controller.start()
        proc_detoken = mp.Process(
            target=start_detokenizer_process,
            args=(
                i,
                server_args,
                port_args,
                pipe_detoken_writer,
            ),
        )
        proc_detoken.start()

        proc_controller_list.append(proc_controller)
        proc_detoken_list.append(proc_detoken)

        # Wait for the model to finish loading
        controller_init_state = pipe_controller_reader.recv()
        detoken_init_state = pipe_detoken_reader.recv()

        if controller_init_state != "init ok" or detoken_init_state != "init ok":
            proc_controller.kill()
            proc_detoken.kill()
            print(
                f"Initialization failed. controller_init_state: {controller_init_state}",
                flush=True,
            )
            print(
                f"Initialization failed. detoken_init_state: {detoken_init_state}",
                flush=True,
            )
            sys.exit(1)

    for proc_controller, proc_detoken in zip(proc_controller_list, proc_detoken_list):
        assert proc_controller.is_alive() and proc_detoken.is_alive()

    # Launch the controller
    global controller
    controller = Controller(
        tokenizer_managers=tokenizer_managers, server_args=server_args
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup, args=(server_args, pipe_finish_writer)
    )
    t.start()

    # Listen for requests
    try:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # Set ulimit
    set_ulimit()

    # Enable show time cost for debugging
    if server_args.show_time_cost:
        enable_show_time_cost()

    # Disable disk cache
    if server_args.disable_disk_cache:
        disable_cache()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Set torch compile config
    if server_args.enable_torch_compile:
        set_torch_compile_config()

    # Set global chat template
    if server_args.chat_template:
        # TODO: replace this with huggingface transformers template
        load_chat_template_for_openai_api(server_args.chat_template)

    # Check flashinfer version
    if not server_args.disable_flashinfer:
        assert_pkg_version(
            "flashinfer",
            "0.1.3",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )


def _wait_and_warmup(server_args, pipe_finish_writer):
    headers = {}
    url = server_args.url()
    logger.info(f"Warm up for the server to be launched at {url}...")
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    for _ in range(120):
        time.sleep(1)
        try:
            requests.get(url + "/get_model_info", timeout=5, headers=headers)
            break
        except requests.exceptions.RequestException:
            pass

    # Send a warmup request
    for _ in range(2):
        try:
            for _ in range(server_args.dp_size):
                for i in range(len(server_args.init_scheduled_models)):
                    res = requests.post(
                        url + "/generate",
                        json={
                            "text": "The capital city of France is",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                            },
                            "model": server_args.init_scheduled_models[i],
                        },
                        headers=headers,
                        timeout=600,
                    )
                    assert res.status_code == 200
        except Exception as e:
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(get_exception_traceback())
            print(f"Initialization failed. warmup error: {e}", flush=True)
            raise e

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("init ok")


class Runtime:
    """
    A wrapper for the server.
    This is used for launching the server in a python program without
    using the commond line interface.
    """

    def __init__(
        self,
        log_level: str = "error",
        model_overide_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.dp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, model_overide_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            kill_child_process(self.pid)
            self.pid = None

    def cache_prefix(self, prefix: str):
        self.endpoint.cache_prefix(prefix)

    def get_tokenizers(self):
        tokenizers = []
        for tokenizer_path in self.server_args.tokenizer_paths:
            tokenizer = get_tokenizer(
                self.server_args.tokenizer_path,
                tokenizer_mode=self.server_args.tokenizer_mode,
                trust_remote_code=self.server_args.trust_remote_code,
            )
        tokenizers.append(tokenizer)

    async def async_generate(
        self,
        model: str,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        json_data = {
            "model": model,
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
        }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)

    add_request = async_generate

    def generate(
        self,
        prompt: str,
        model: str,
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
    ):
        json_data = {
            "model": model,
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "top_logprobs_num": top_logprobs_num,
        }
        response = requests.post(
            self.url + "/generate",
            json=json_data,
        )
        return json.dumps(response.json())

    def __del__(self):
        self.shutdown()
