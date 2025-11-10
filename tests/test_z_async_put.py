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
import sys
from pathlib import Path

import pytest
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def ray_setup():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "1", "RAY_DEDUP_LOGS": "0"}},
        log_to_driver=True,
    )
    yield
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray has been shut down completely after test")


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
        # 修复数据切分逻辑
        batch_size = data.batch_size[0]
        num_workers = self.config.put_num_workers
        chunk_size = batch_size // num_workers

        split_data = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else batch_size
            chunk = data[start_idx:end_idx]
            split_data.append(chunk)

        # 使用ray.get等待所有异步操作完成
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


class TestAsyncPut:
    """Test class for async put functionality"""

    def test_async_put(self, ray_setup):
        """Test async put functionality with multiple workers"""
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

        # 修复断言逻辑 - 检查全局索引是否正确
        expected_indices = list(range(16))  # 8 * 2 = 16
        assert batch_meta.global_indexes == expected_indices, (
            f"Expected {expected_indices}, got {batch_meta.global_indexes}"
        )

        # 使用Trainer的client来获取数据，确保一致性
        data_fetch = asyncio.run(trainer.data_system_client.async_get_data(batch_meta))

        # 修复断言 - 检查形状和数据内容
        assert data_fetch.batch_size == data_origin.batch_size, (
            f"Shape mismatch: {data_fetch.batch_size} vs {data_origin.batch_size}"
        )
        assert data_fetch["input_ids"].shape == data_origin["input_ids"].shape, (
            f"Input IDs shape mismatch: {data_fetch['input_ids'].shape} vs {data_origin['input_ids'].shape}"
        )

        # 检查数据内容是否相同（允许顺序不同）
        sorted_fetch = torch.sort(data_fetch["input_ids"].flatten())[0]
        sorted_origin = torch.sort(data_origin["input_ids"].flatten())[0]
        assert torch.equal(sorted_fetch, sorted_origin), "Data content mismatch after sorting"

        # 添加更详细的调试信息
        logger.info(f"Original data shape: {data_origin.shape}")
        logger.info(f"Fetched data shape: {data_fetch.shape}")
        logger.info(f"Original input_ids shape: {data_origin['input_ids'].shape}")
        logger.info(f"Fetched input_ids shape: {data_fetch['input_ids'].shape}")
        logger.info(f"Global indexes: {batch_meta.global_indexes}")

    def test_async_put_with_different_config(self, ray_setup):
        """Test async put with different configuration"""
        config_str = """
          global_batch_size: 4
          num_global_batch: 1
          num_data_storage_units: 1
          put_num_workers: 1
          num_n_samples: 1
        """
        dict_conf = OmegaConf.create(config_str)
        trainer = Trainer(dict_conf)
        data_origin, batch_meta = trainer.async_put()

        # 检查全局索引是否正确
        expected_indices = list(range(4))  # 4 * 1 = 4
        assert batch_meta.global_indexes == expected_indices, (
            f"Expected {expected_indices}, got {batch_meta.global_indexes}"
        )

        # 获取数据并验证
        data_fetch = asyncio.run(trainer.data_system_client.async_get_data(batch_meta))

        assert data_fetch.batch_size == data_origin.batch_size
        assert data_fetch["input_ids"].shape == data_origin["input_ids"].shape

        # 检查数据内容
        sorted_fetch = torch.sort(data_fetch["input_ids"].flatten())[0]
        sorted_origin = torch.sort(data_origin["input_ids"].flatten())[0]
        assert torch.equal(sorted_fetch, sorted_origin)
