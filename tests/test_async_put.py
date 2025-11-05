# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import math
import os
import sys
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"
ray.init()


@ray.remote(num_cpus=1)
class AsyncPutWorker:
    def __init__(self, config, data_system_controller_info):
        self.config = config
        self.data_system_client = AsyncTransferQueueClient(
            client_id="AsyncvLLMServer",
            controller_info=data_system_controller_info,
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

    async def put_data(self, data, partition_id):
        await self.data_system_client.async_put(data=data, partition_id=partition_id)


class AsyncPutManager:
    def __init__(self, config, data_system_storage_unit_infos, data_system_controller_info):
        self.config = config

        self.data_system_client = AsyncTransferQueueClient(
            client_id="AsyncPutManager",
            controller_info=data_system_controller_info,
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

        self.async_put_workers = []
        num_workers = self.config.put_num_workers
        for i in range(num_workers):
            self.async_put_workers.append(AsyncPutWorker.remote(config, data_system_controller_info))

    def put_data(self, data, partition_id):
        split_data = data.split(data.batch_size[0] // self.config.put_num_workers)
        ray.get(
            [
                worker.put_data.remote(data_chunk, partition_id)
                for worker, data_chunk in zip(self.async_put_workers, split_data, strict=True)
            ]
        )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.data_system_client = self._initialize_data_system()
        self.async_put_manager = AsyncPutManager(
            self.config,
            self.data_system_storage_unit_infos,
            self.data_system_controller_info,
        )

    def _initialize_data_system(self):
        total_storage_size = self.config.global_batch_size * self.config.num_global_batch * self.config.num_n_samples
        self.data_system_storage_units = {}
        storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=1)
        for storage_unit_rank in range(self.config.num_data_storage_units):
            storage_node = SimpleStorageUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
            ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
            self.data_system_storage_units[storage_unit_rank] = storage_node
            logger.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")

        # 2. Initialize TransferQueueController (single controller only)
        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")

        # 3. Prepare necessary information
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig
        # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. Create client
        self.data_system_client = AsyncTransferQueueClient(
            client_id="Trainer",
            controller_info=self.data_system_controller_info,
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)
        # Note: The client contains ZMQ objects. Currently, we cannot transmit the same client instance
        # to multiple places, as this will cause serialization errors in Ray.
        # Workaround: If you need to use a client in multiple Ray actors or processes, create a separate
        # AsyncTransferQueueClient instance for each actor/process instead of sharing or transmitting the same instance.
        return self.data_system_client

    def async_put(self):
        input_ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [100, 111], [200, 222], [300, 333]])
        input_ids_repeated = torch.repeat_interleave(input_ids, self.config.num_n_samples, dim=0)
        prompt_batch = TensorDict(
            {"input_ids": input_ids_repeated, "attention_mask": input_ids_repeated},
            batch_size=input_ids_repeated.size(0),
        )

        self.async_put_manager.put_data(prompt_batch, partition_id="train_0")

        batch_meta = asyncio.run(
            self.data_system_client.async_get_meta(
                data_fields=["input_ids", "attention_mask"],
                batch_size=self.config.global_batch_size * self.config.num_n_samples,
                partition_id="train_0",
                get_n_samples=False,
                task_name="batch_meta_test",
            )
        )
        return prompt_batch, batch_meta


def test_async_put():
    config_str = """
      global_batch_size: 8
      num_global_batch: 1
      num_data_storage_units: 2
      put_num_workers: 2
      num_n_samples: 2

    """
    dict_conf = OmegaConf.create(config_str)
    trainer = Trainer(dict_conf)
    data_origin, batch_meta = trainer.async_put()

    assert batch_meta.global_indexes == list(range(16))

    data_fetch = asyncio.run(trainer.async_put_manager.data_system_client.async_get_data(batch_meta))

    assert data_fetch.shape == data_origin.shape
    assert data_fetch["input_ids"].shape == data_origin["input_ids"].shape
    assert torch.equal(
        torch.sort(data_fetch["input_ids"].flatten())[0], torch.sort(data_origin["input_ids"].flatten())[0]
    )
