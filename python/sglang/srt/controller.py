import enum
import time
import asyncio
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    AbortReq,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedGenerateReqInput,
    DeactivateReq,
    ActivateReq,
)
from sglang.srt.server_args import ServerArgs
from typing import Dict
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus



class ModelStatus(enum.Enum):
    """Model status."""
    ACTIVE = enum.auto()  # ON_GPU
    IN_TRANSIT = enum.auto()    # ON_CPU
    OFF = enum.auto()  # OFF


class RequestWrapper:
    def __init__(self, obj: GenerateReqInput, request):
        self.obj = obj
        self.req_id = obj.rid
        self.model = obj.model
        self.arrival_time = time.time()
        self.request = request
        self.process_model_queue_tasks = {}
        self.event = asyncio.Event()
        self.send_to_tokenizer_manager = asyncio.Future()


class Controller:
    def __init__(self, tokenizer_managers: Dict[str, TokenizerManager],
                 server_args: ServerArgs):
        self.model_queues: Dict[str, asyncio.Queue] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.model_unfinished_requests: Dict[str, set[str]] = {}
        self.tokenizer_managers = tokenizer_managers
        self.server_args = server_args

        # initialize request queue and unfinished requests for each model
        for model in tokenizer_managers:
            self.model_queues[model] = asyncio.Queue()
            self.model_unfinished_requests[model] = set()

        # initialize the model status
        asyncio.run(self._init_model_status())

        self.to_create_loop = True

        # TODO: remove this, and automatically compute the available memory
        self.available_memory = 24

    async def _init_model_status(self):
        print("Initializing model status.")
        init_scheduled_models = self.server_args.init_scheduled_models

        for model in self.tokenizer_managers:
            if model not in init_scheduled_models:
                self.model_status[model] = ModelStatus.OFF
            else:
                print(f"Activating model {model} in _init_model_status.")
                await self.tokenizer_managers[model].activate_model()
                self.model_status[model] = ModelStatus.ACTIVE
        print(f"Model status {self.model_status}")

    async def generate_request(self, obj: GenerateReqInput, request):
        if self.to_create_loop:
            self._create_loop()

        obj.post_init()

        # put the request into the queue corresponding to the model
        request_wrapper = RequestWrapper(obj, request)
        model = obj.model
        await self.model_queues[model].put(request_wrapper)

        send_future = await request_wrapper.send_to_tokenizer_manager
        try:
            ret = await send_future
            self.model_unfinished_requests[model].remove(request_wrapper.req_id)
            return ret
        except Exception as e:
            return JSONResponse(
                  {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )

    def _create_loop(self):
        self.to_create_loop = False
        asyncio.create_task(self.process_request_loop())
        asyncio.create_task(self.switch_model_loop())


    async def switch_model_loop(self):
        # TODO: enable multiple models to be switched on/off at the same time
        while True:
            await self.may_switch_model()
            await asyncio.sleep(1)

    async def process_request_loop(self):
        while True:
            for model in self.model_queues:
                if self.model_status[model] == ModelStatus.ACTIVE:
                    await self._process_model_queue(model)
            await asyncio.sleep(0.1)

    async def may_switch_model(self):
        for model, queue in self.model_queues.items():
            if self.model_status[model] == ModelStatus.OFF:
                # check if the model should be switched on
                if self.should_switch_on_model(model, queue):
                    print(f"Switching on model {model}")
                    await self.switch_on_model(model)

            elif self.model_status[model] == ModelStatus.ACTIVE:
                # check if the model can get larger memory pool
                if self.should_expand_memory_pool(model, queue):
                    await self.expand_memory_pool(model)

                # check if the model should be switched off
                if self.should_switch_off_model(model, queue):
                    await self.switch_off_model(model)

                if self.should_shrink_memory_pool(model, queue):
                    await self.shrink_memory_pool(model)
            else:
                raise ValueError(f"Invalid model status: {self.model_status[model]}")

    async def _process_model_queue(self, model: str):
        """Process all requests in the queue for the given model."""
        tokenizer_manager = self.tokenizer_managers[model]

        while not self.model_queues[model].empty():
            request_wrapper = await self.model_queues[model].get()
            assert request_wrapper.obj.stream is False, "Stream is not supported."
            future = tokenizer_manager.generate_request(request_wrapper.obj, request_wrapper.request).__anext__()
            request_wrapper.send_to_tokenizer_manager.set_result(future)
            self.model_unfinished_requests[model].add(request_wrapper.req_id)
    
    def should_switch_off_model(self, model, queue):
        # If no requests in its queue, and no unfinshed requests
        # if queue.qsize() == 0 and len(self.model_unfinished_requests[model]) == 0:
        #     return True
        return False
        # return True

    async def switch_off_model(self, model):
        print(f"Switching off model {model}")
        tokenizer_manager = self.tokenizer_managers[model]
        self.model_status[model] = ModelStatus.IN_TRANSIT
        out = await tokenizer_manager.deactivate_model(to_cpu=False)
        self.model_status[model] = ModelStatus.OFF
        print(f"model {model} is switched off.")
        return out

    def should_switch_on_model(self, model, queue):
        # TODO: check arrival time of the first req in queue.
        if queue.qsize() >= 10:
            return True
        return False

    def _get_active_models(self):
        return [model for model, status in self.model_status.items() if status == ModelStatus.ACTIVE]
    
    def get_victim_model(self):
        active_models = self._get_active_models()
        # TODO: implement a policy to choose the victim model
        if len(active_models) == 0:
            raise ValueError("No active model found.")
        return active_models[0]

    async def switch_on_model(self, model):
        print(f"Switching on model {model}")
        # TODO: switch off multiple models until enough memory is available
        if get_available_memory(memory=0) < get_memory_needed(model):
            victim_model = self.get_victim_model()
            await self.switch_off_model(victim_model)

        tokenizer = self.tokenizer_managers[model]
        self.model_status[model] = ModelStatus.IN_TRANSIT
        print(f"Activating model {model} in switch_on_model.")
        out = await tokenizer.activate_model()
        self.model_status[model] = ModelStatus.ACTIVE
        print(f"model {model} is switched on.")
        return out

    def should_expand_memory_pool(self, model, queue):
        return False

    async def expand_memory_pool(self, model):
        pass

    def should_shrink_memory_pool(self, model, queue):
        return False
    
    async def shrink_memory_pool(self, model):
        pass


def get_available_memory(memory: int):
    # TODO: implement this
    return 100

def get_memory_needed(model: str):
    return 10
