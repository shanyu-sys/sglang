import asyncio
import enum
import time
from http import HTTPStatus
from typing import Dict

from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.srt.managers.io_struct import (
    AbortReq,
    ActivateReq,
    BatchTokenIDOut,
    DeactivateReq,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
import logging
from collections import deque
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

# Memory needed for each model
MODEL_TO_MEMORY = {
    "meta-llama/Llama-2-7b-chat-hf": 10,
    "meta-llama/Llama-2-7b-hf": 10,
    "mistralai/Mistral-7B-Instruct-v0.2": 10,
}

class ModelStatus(enum.Enum):
    """Model status."""

    ACTIVE = enum.auto()  # ON_GPU
    IN_TRANSIT = enum.auto()  # ON_CPU
    OFF = enum.auto()  # OFF


class RequestWrapper:
    def __init__(self, obj: GenerateReqInput, request):
        self.obj = obj
        self.req_id = obj.rid
        self.model = obj.model
        self.arrival_time = obj.arrival_time
        self.min_schedule_time = self._get_min_schedule_time(obj)
        self.request = request
        self.process_model_queue_tasks = {}
        self.event = asyncio.Event()
        self.send_to_tokenizer_manager = asyncio.Future()

    def _get_min_schedule_time(self, obj: GenerateReqInput):
        slo = obj.slo
        cool_down_time = 20 # wait ongoing requests to finish
        swap_out_time = 0.8
        swap_in_time = 0.2
        p99_e2e_latency = 60
        if slo is None:
            min_schedule_time = float("inf")
        else:
            min_schedule_time = self.arrival_time + slo - cool_down_time - swap_out_time - swap_in_time - p99_e2e_latency
        return min_schedule_time


class Controller:
    def __init__(
        self, tokenizer_managers: Dict[str, TokenizerManager], server_args: ServerArgs
    ):
        self.model_queues: Dict[str, deque] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.model_unfinished_requests: Dict[str, set[str]] = {}
        self.model_inactive_start_time: Dict[str, float] = {}
        self.tokenizer_managers = tokenizer_managers
        self.server_args = server_args

        # initialize request queue and unfinished requests for each model
        for model in tokenizer_managers:
            self.model_queues[model] = deque()
            self.model_unfinished_requests[model] = set()
            self.model_inactive_start_time[model] = 0

        self._inactivate_threshold = server_args.inactivate_threshold
        # self._inactivate_threshold = 10

        # initialize the model status
        asyncio.run(self._init_model_status())

        self.to_create_loop = True

        # TODO: remove this, and automatically compute the available memory
        self._available_memory = 0
        self._available_memory_lock = None
        self._memory_needed = 10

        self.background_tasks = set()

    async def _init_model_status(self):
        logger.info("Initializing model status.")
        init_scheduled_models = self.server_args.init_scheduled_models

        for model in self.tokenizer_managers:
            if model not in init_scheduled_models:
                self.model_status[model] = ModelStatus.OFF
            else:
                logger.info(f"Activating model {model} in _init_model_status.")
                await self.tokenizer_managers[model].activate_model()
                self.model_status[model] = ModelStatus.ACTIVE
        logger.info(f"Model status {self.model_status}")

    async def generate_request(self, obj: GenerateReqInput, request):
        if self.to_create_loop:
            self._create_loop()

        obj.post_init()

        # put the request into the queue corresponding to the model
        request_wrapper = RequestWrapper(obj, request)
        model = obj.model

        self.model_queues[model].append(request_wrapper)
        logger.info(f"Request {obj.rid} is put into the queue of model {model}.")

        send_future = await request_wrapper.send_to_tokenizer_manager
        # logger.info(f"Request {obj.rid} is sent to the tokenizer manager of model {model}.")
    
        try:
            ret = await send_future
            logger.info(f"Request {obj.rid} is finished processing by model {model}.")
            return ret
        except asyncio.CancelledError:
            logger.warning(f"Request {obj.rid} was cancelled because it exceeded SLO.")
            return JSONResponse(
                {"error": {"message": f"Request {obj.rid} exceeded SLO"}}, 
            status_code=HTTPStatus.REQUEST_TIMEOUT
        )
        except Exception as e:
            logger.error(f"Error in processing request {obj.rid}: {e}")
            return JSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )
        finally:
            if request_wrapper.req_id in self.model_unfinished_requests[model]:
                self.model_unfinished_requests[model].remove(request_wrapper.req_id)

    def _create_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        task1 = loop.create_task(self.process_request_loop())
        self.background_tasks.add(task1)
        task2 = loop.create_task(self.switch_model_loop())
        self.background_tasks.add(task2)

        self._available_memory_lock = asyncio.Lock()

    async def switch_model_loop(self):
        # TODO: enable multiple models to be switched on/off at the same time
        while True:
            await self.may_switch_model()
            await asyncio.sleep(0)

    async def process_request_loop(self):
        while True:
            for model in self.model_status:
                if self.model_status[model] == ModelStatus.ACTIVE:
                    try: 
                        self._process_model_queue(model)
                    except Exception as e:
                        logger.error(f"Error in processing model {model} queue: {e}")
                        raise e
            await asyncio.sleep(0)

    async def may_switch_model(self):
        for model, queue in self.model_queues.items():
            # print(f"model {model} of status {self.model_status[model] } has {queue.qsize()} requests in queue.")
            if self.model_status[model] == ModelStatus.OFF:
                # check if the model should be switched on
                if await self.should_switch_on_model(model, queue):
                    await self.switch_on_model(model)

            elif self.model_status[model] == ModelStatus.ACTIVE:

               # check if the model should be switched off
                if self.should_switch_off_model(model, queue):
                    await self.switch_off_model(model)

                # check if the model can get larger memory pool
                if self.should_expand_memory_pool(model, queue):
                    await self.expand_memory_pool(model)

                if self.should_shrink_memory_pool(model, queue):
                    await self.shrink_memory_pool(model)
            else:
                raise ValueError(f"Invalid model status: {self.model_status[model]}")

    def _process_model_queue(self, model: str):
        """Process all requests in the queue for the given model."""
        tokenizer_manager = self.tokenizer_managers[model]
        
        qsize = len(self.model_queues[model])
        self._update_inactive_start_time(model, qsize)

        while len(self.model_queues[model]) > 0:
            self.model_inactive_start_time[model] = 0

            request_wrapper = self.model_queues[model].popleft()
            assert request_wrapper.obj.stream is False, "Stream is not supported."

            # abort request if exceed SLO
            if time.time() + 0.5 - request_wrapper.arrival_time > request_wrapper.obj.slo:
                # do not send to tokenizer manager to generate response
                future = asyncio.Future()
                future.cancel()
                request_wrapper.send_to_tokenizer_manager.set_result(future)
                continue

            future = tokenizer_manager.generate_request(
                request_wrapper.obj, request_wrapper.request
            ).__anext__()
            request_wrapper.send_to_tokenizer_manager.set_result(future)
            self.model_unfinished_requests[model].add(request_wrapper.req_id)

    def _update_inactive_start_time(self, model, qsize):
        if qsize > 0:
            # print(f"Reset model_inactive_start_time for model {model}, which has {queue.qsize()} requests in queue.")
            self.model_inactive_start_time[model] = 0
        else:
            if self.model_inactive_start_time[model] == 0:
                self.model_inactive_start_time[model] = time.time()

    def should_switch_off_model(self, model, queue):
        if self.server_args.swap_policy == "enhanced":
            if self._inactivate_threshold is not None:
                if self.model_inactive_start_time[model] != 0:
                    inactivate_time = time.time() - self.model_inactive_start_time[model]
                    # print(f"Model {model} has been inactive for {inactivate_time:.2f} seconds.")
                else:
                    inactivate_time = 0
                if inactivate_time > self._inactivate_threshold:
                    logger.info(f"[time={time.time():.2f}] Model {model} has been inactive for {inactivate_time} seconds, switching off.")
                    return True
        return False

    async def switch_off_model(self, model):
        logger.info(f"[time={time.time():.2f}] Prepare to switch off model {model}")
        t1 = time.time()
        self.model_status[model] = ModelStatus.IN_TRANSIT
        tokenizer_manager = self.tokenizer_managers[model]
        # wait for all requests finish 
        while len(self.model_unfinished_requests[model]) > 0:
            await asyncio.sleep(0)

        t2 = time.time()
        logger.info(f"[time={time.time():.2f}] Start switching off model {model}. Waited {t2-t1:.2f} seconds for ongoing requests to finish.")
        async with self._available_memory_lock:
            out = await tokenizer_manager.deactivate_model(to_cpu=True)
            self.model_status[model] = ModelStatus.OFF
            self._available_memory += self._memory_needed

        self.model_inactive_start_time[model] = 0

        t3 = time.time()
        logger.info(f"[time={time.time():.2f}] model {model} is switched off. It took {t3-t2:.2f} seconds.")
        return out

    async def should_switch_on_model(self, model, queue):
        # print(f"In should_switch_on_model for model {model}")
        # Rule: switch on if current time is larger than the min_schedule_time of the first request in the queue
        if len(queue) > 0:
            min_schedule_time = queue[0].min_schedule_time
            if time.time() > min_schedule_time:
                logger.info(f"[time={time.time():.2f}] Switch on model {model}, since min_schedule_time {min_schedule_time} is reached.")
                return True

        qsize = len(queue)
        # Rule: switch on if the queue size is large enough
        # if qsize >= 32:
        #     logger.info(f"[time={time.time():.2f}] Queue size of model {model} is {qsize}, switching on.")
        #     return True
        # Rule: switch on if enough memory is available and the queue size is larger than 0
        if qsize > 0:
            async with self._available_memory_lock:
                if self._available_memory >= self._memory_needed:
                    logger.info(f"[time={time.time():.2f}] Switch on model {model}, since memory is enough and it has {qsize} requests in queue.")
                    return True
        # print(f"should not switch on model {model}")
        return False

    def _get_active_models(self):
        return [
            model
            for model, status in self.model_status.items()
            if status == ModelStatus.ACTIVE
        ]

    def get_victim_model(self):
        active_models = self._get_active_models()
        # TODO: implement a policy to choose the victim model
        if len(active_models) == 0:
            raise ValueError("No active model found.")
        return active_models[0]

    async def switch_on_model(self, model):
        logger.info(f"[time={time.time():.2f}] Prepare to switch on model {model}")
        self.model_status[model] = ModelStatus.IN_TRANSIT

        # TODO: switch off multiple models until enough memory is available
        if self._available_memory < self._memory_needed:
            victim_model = self.get_victim_model()
            await self.switch_off_model(victim_model)

        tokenizer = self.tokenizer_managers[model]
        logger.info(f"[time={time.time():.2f}] Switching on model {model}")
        t1 = time.time()
        async with self._available_memory_lock:
            out = await tokenizer.activate_model()
            self.model_status[model] = ModelStatus.ACTIVE
            self._available_memory -= self._memory_needed
        logger.info(f"[time={time.time():.2f}] Model {model} is switched on. It took {time.time()-t1:.2f} seconds.")
        return out

    def should_expand_memory_pool(self, model, queue):
        return False

    async def expand_memory_pool(self, model):
        pass

    def should_shrink_memory_pool(self, model, queue):
        return False

    async def shrink_memory_pool(self, model):
        pass
