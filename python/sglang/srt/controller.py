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
from collections import deque, defaultdict
from sglang.srt.utils import get_available_gpu_memory, roundrobin

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
        cool_down_time = 60 # wait ongoing requests to finish
        swap_out_time = 5
        swap_in_time = 5
        e2e_latency = 30
        if slo is None:
            min_schedule_time = float("inf")
        else:
            min_schedule_time = self.arrival_time + slo - cool_down_time - swap_out_time - swap_in_time - e2e_latency
        return min_schedule_time


class Controller:
    def __init__(
        self, tokenizer_managers: Dict[str, list[TokenizerManager]], server_args: ServerArgs
    ):
        self.model_queues: Dict[str, deque] = {}
        self.model_status: Dict[str, list[ModelStatus]] = defaultdict(list)
        self.model_unfinished_requests: Dict[str, list(set[str])] = defaultdict(list)
        self.model_inactive_start_time: Dict[str, float] = {}
        self.tokenizer_managers = tokenizer_managers
        self.model_tokenizer_generators = {}
        self.server_args = server_args

        # initialize request queue and unfinished requests for each model
        for model in tokenizer_managers:
            self.model_queues[model] = deque()
            self.model_inactive_start_time[model] = 0
            for i in range(len(tokenizer_managers[model])):
                self.model_unfinished_requests[model].append(set())

            n_replicas = len(self.tokenizer_managers[model])
            self.model_tokenizer_generators[model] = roundrobin(zip(range(n_replicas), self.tokenizer_managers[model]))

        self._inactivate_threshold = server_args.inactivate_threshold

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
        init_scheduled_model_replicas = self.server_args.init_scheduled_model_replicas

        for model in self.tokenizer_managers:
            if model not in init_scheduled_models:
                self.model_status[model] = [ModelStatus.OFF] * len(self.tokenizer_managers[model])
            else:
                model_idx = init_scheduled_models.index(model)
                init_n_replica = init_scheduled_model_replicas[model_idx]
                for j in range(len(self.tokenizer_managers[model])):
                    if j < init_n_replica:
                        logger.info(f"Activating replica {j} of model {model} in _init_model_status.")
                        await self.tokenizer_managers[model][j].activate_model()
                        self.model_status[model].append(ModelStatus.ACTIVE)
                    else:
                        self.model_status[model].append(ModelStatus.OFF) 
        logger.info(f"Model status: {self.model_status}")

    async def generate_request(self, obj: GenerateReqInput, request):
        if self.to_create_loop:
            self._create_loop()

        obj.post_init()

        # put the request into the queue corresponding to the model
        request_wrapper = RequestWrapper(obj, request)
        model = obj.model

        self.model_queues[model].append(request_wrapper)
        logger.info(f"Request {obj.rid} is put into the queue of model {model}.")

        t0 = time.time()
        send_future = await request_wrapper.send_to_tokenizer_manager
        t1 = time.time()
        logger.debug(f"Request {obj.rid} is sent to the tokenizer manager of model {model}.")
        wait_in_controller = t1 - t0

        try:
            ret = await send_future
            logger.info(f"Request {obj.rid} is finished processing by model {model}. Processed in {time.time()-t1:.2f} seconds.")
            if "abort" in ret:
                logger.warning(f"Request {obj.rid} was aborted. Waited {wait_in_controller:.2f}s in controller queue."
                            f" Time waited in worker queue: {ret['meta_info']['abort_time'] - t1:.2f}s")
                return JSONResponse(
                    {"error": {"message": "Request aborted."}},
                    status_code=HTTPStatus.REQUEST_TIMEOUT
                )
            init_schedule_time = ret["meta_info"]["init_schedule_time"]
            wait_in_tokenizer_manager = init_schedule_time - t1
            output_len = ret["meta_info"]["completion_tokens"]
            tokenizer_process_time = time.time() - t1
            computation_time = ret["meta_info"]["finish_time"] - init_schedule_time
            finish_to_controller = time.time() - ret["meta_info"]["finish_time"]
            logger.info(f"Request {obj.rid} with {output_len} output tokens is finished. " 
                        f"Time waited in controller queue: {wait_in_controller:.2f}s; "
                        f"Time waited in worker queue: {wait_in_tokenizer_manager:.2f}s; "
                        f"Computation time: {computation_time:.2f}s; "
                        f"From finish back to controller time: {finish_to_controller:.2f}s ")

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
            for unfinished in self.model_unfinished_requests[model]:
                if request_wrapper.req_id in unfinished:
                    unfinished.remove(request_wrapper.req_id)
                    break

    def _create_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        try:
            task1 = loop.create_task(self.process_request_loop())
            self.background_tasks.add(task1)
            task2 = loop.create_task(self.switch_model_loop())
            self.background_tasks.add(task2)
        except Exception as e:
            logger.error(f"Error in creating loop: {e}")
            raise e

        self._available_memory_lock = asyncio.Lock()

    async def switch_model_loop(self):
        # TODO: enable multiple models to be switched on/off at the same time
        while True:
            try: 
                await self.may_switch_model()
            except Exception as e:
                logger.error(f"Error in switch_model_loop: {e}")
                raise e
            await asyncio.sleep(0)

    def _get_active_models(self):
        active_models = []
        for model in self.model_status:
            if self._get_model_status(model) == ModelStatus.ACTIVE:
                active_models.append(model)
        # for model, sts in self.model_status.items():
        #     for st in sts:
        #         if st == ModelStatus.ACTIVE:
        #             active_models.append(model)
        #             break
        return active_models
    
    def _get_model_status(self, model):
        sts = self.model_status[model]
        for st in sts:
            if st == ModelStatus.IN_TRANSIT:
                return ModelStatus.IN_TRANSIT
        
        for st in sts:
            if st == ModelStatus.ACTIVE:
                return ModelStatus.ACTIVE
        return ModelStatus.OFF

    async def process_request_loop(self):
        while True:
            logger.debug("Processing request loop.")
            active_models = self._get_active_models()
            for model in active_models:
                try: 
                    self._process_model_queue(model)
                except Exception as e:
                    logger.error(f"Error in processing model {model} queue: {e}")
                    raise e
            await asyncio.sleep(0)

    async def may_switch_model(self):
        for model, queue in self.model_queues.items():
            logger.debug(f"model {model} of status {self.model_status[model] } has {len(queue)} requests in queue.")
            if self._get_model_status(model) == ModelStatus.OFF:
                # check if the model should be switched on
                try:
                    if_should_switch_on, replica_indices = await self.should_switch_on_model(model, queue)
                except Exception as e:
                    logger.error(f"Error in should_switch_on_model: {e}")
                    raise e
                if if_should_switch_on:
                    try:
                        await self.switch_on_model(model, replica_indices)
                        await asyncio.sleep(0)
                    except Exception as e:
                        logger.error(f"Error in switch_on_model: {e}")
                        raise e

            elif self._get_model_status(model) == ModelStatus.ACTIVE:
                # check if the model should be switched off
                if_should_switch_off, replica_indices = self.should_switch_off_model(model, queue)
                if if_should_switch_off:
                    try:
                        await self.switch_off_model(model, replica_indices)
                        await asyncio.sleep(0)
                    except Exception as e:
                        logger.error(f"Error in switch_off_model: {e}")
                        raise e

                # check if the model can get larger memory pool
                if self.should_expand_memory_pool(model, queue):
                    await self.expand_memory_pool(model)

                if self.should_shrink_memory_pool(model, queue):
                    await self.shrink_memory_pool(model)
            elif self._get_model_status(model) == ModelStatus.IN_TRANSIT:
                continue
            else:
                raise ValueError(f"Invalid model status: {self.model_status[model]}")

    def _process_model_queue(self, model: str):
        """Process all requests in the queue for the given model."""
        logger.debug(f"Processing model {model} queue.")
        idx_to_tokenizer_managers = self.model_tokenizer_generators[model]
        
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

            idx, tokenizer_manager = next(idx_to_tokenizer_managers)
            future = tokenizer_manager.generate_request(
                request_wrapper.obj, request_wrapper.request
            ).__anext__()
            request_wrapper.send_to_tokenizer_manager.set_result(future)
            self.model_unfinished_requests[model][idx].add(request_wrapper.req_id)

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
                    replica_indices = [
                        i
                        for i, status in enumerate(self.model_status[model])
                        if status == ModelStatus.ACTIVE
                    ]
                    return True, replica_indices
        return False, []

    async def switch_off_model(self, model, replica_indices):
        queue = self.model_queues[model]
        logger.info(f"[time={time.time():.2f}] Prepare to switch off model {model}. Queue size is {len(queue)}")
        tasks = []
        for idx in replica_indices:
        #     tasks.append(self.switch_off_model_replica(model, idx, n_replicas=len(replica_indices)))
        
        # try:
        #     await asyncio.gather(*tasks)
        # except Exception as e:
        #     logger.error(f"Error in switching off model {model}: {e}")
            # raise e
            try:
                await self.switch_off_model_replica(model, idx, n_replicas=len(replica_indices))
            except Exception as e:
                logger.error(f"Error in switching off model {model}: {e}")
                raise e
            assert len(self.model_unfinished_requests[model][idx]) == 0
        logger.info(f"[time={time.time():.2f}] Model {model} is switched off. Queue size is {len(queue)}.")
        return tasks

    async def switch_off_model_replica(self, model, idx, n_replicas):
        t1 = time.time()
        logger.info(f"Switching off replica {idx} of model {model}")
        self.model_status[model][idx] = ModelStatus.IN_TRANSIT
        tokenizer_manager = self.tokenizer_managers[model][idx]
        num_unfinished_reqs = len(self.model_unfinished_requests[model][idx])
        # wait for all requests finish
        while len(self.model_unfinished_requests[model][idx]) > 0:
            await asyncio.sleep(0)
        
        t2 = time.time()
        logger.info(f"[time={time.time():.2f}] Start switching off replica {idx} of model {model}. Waited {t2-t1:.2f} seconds for {num_unfinished_reqs} ongoing requests to finish.")

        async with self._available_memory_lock:
            try:     
                out = await tokenizer_manager.deactivate_model(to_cpu=True)
            except Exception as e:
                logger.error(f"Error in switching off model {model}: {e}")
                raise e
            self.model_status[model][idx] = ModelStatus.OFF
            self._available_memory += self._memory_needed // n_replicas

        t3 = time.time()
        logger.info(f"[time={time.time():.2f}] Replica {idx} of model {model} is switched off. It took {t3-t2:.2f} seconds.")
        return out

    async def should_switch_on_model(self, model, queue):
        # print(f"In should_switch_on_model for model {model}")
        # Rule: switch on if current time is larger than the min_schedule_time of the first request in the queue
        model_indices = [
            i
            for i, status in enumerate(self.model_status[model])
            if status == ModelStatus.OFF
        ]
        if len(queue) > 0:
            min_schedule_time = queue[0].min_schedule_time
            arrival_time = queue[0].arrival_time
            waiting_time = time.time() - arrival_time
            # todo, the end_to_end latency will change according to the queue size.
            if time.time() > min_schedule_time:
                logger.info(f"[time={time.time():.2f}] Switch on model {model}, since min_schedule_time is reached. Head of queue request has been waiting for {waiting_time:.2f} seconds.")

                return True, model_indices

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
                    return True, model_indices
        # print(f"should not switch on model {model}")
        return False, []

    def get_victim_model_replicas(self):
        active_models = self._get_active_models()
        # TODO: implement a policy to choose the victim model
        if len(active_models) == 0:
            raise ValueError("No active model found.")
        victim_model = active_models[0]
        victim_replicas = [
            i
            for i, status in enumerate(self.model_status[victim_model])
            if status == ModelStatus.ACTIVE
        ]
        return victim_model, victim_replicas

    async def switch_on_model(self, model, replica_indices):
        queue = self.model_queues[model]
        # print("in switch_on_model")
        logger.info(f"[time={time.time():.2f}] Prepare to switch on model {model}, queue size {len(queue)}")
        # TODO: switch off multiple models until enough memory is available
        if self._available_memory < self._memory_needed:
            victim_model, victim_replicas = self.get_victim_model_replicas()
            try:
                await self.switch_off_model(victim_model, victim_replicas)
            except Exception as e:
                logger.error(f"Error in switching off model {victim_model}: {e}")
                raise e

        t0 = time.time()
        logger.info(f"[time={time.time():.2f}] Switching on model {model}, queue size {len(queue)}")
        for idx in replica_indices:
            self.model_status[model][idx] = ModelStatus.IN_TRANSIT

            tokenizer_manager = self.tokenizer_managers[model][idx]
            logger.info(f"Switching on replica {idx} of model {model}")

            t1 = time.time()
            async with self._available_memory_lock:
                try:
                    out = await tokenizer_manager.activate_model()
                except Exception as e:
                    logger.error(f"Error in switching on model {model}: {e}")
                    raise e
                self.model_status[model][idx] = ModelStatus.ACTIVE
                self._available_memory -= self._memory_needed // len(replica_indices)
            
            logger.info(f"Replica {idx} of model {model} is switched on. It took {time.time()-t1:.2f} seconds.")
        self.model_inactive_start_time[model] = 0
        logger.info(f"[time={time.time():.2f}] Model {model} is switched on. It took {time.time()-t0:.2f} seconds. Queue size is {len(queue)}. Model status {self.model_status[model]}, get_active_models: {self._get_active_models()}")
        return out

    def should_expand_memory_pool(self, model, queue):
        return False

    async def expand_memory_pool(self, model):
        pass

    def should_shrink_memory_pool(self, model, queue):
        return False

    async def shrink_memory_pool(self, model):
        pass
