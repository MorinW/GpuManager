# -*- coding: utf-8 -*-
__author__ = "Fishbone"
__version__ = "0.1"

import os
import platform


class GpuManager(object):
    def __init__(self, visible_gpus: tuple=()):
        os_name = platform.system()
        if os_name == "Linux":
            self.os_name = os_name
        else:
            raise ValueError("System %s is not adapted!" % os_name)
        self.set_visible_gpu(visible_gpus)

    def set_visible_gpu(self, gpu_indexes: tuple):
        assert isinstance(gpu_indexes, tuple)
        index_length = len(gpu_indexes)
        if index_length > len(set(gpu_indexes)):
            raise ValueError("There are the same index in the visible tuple!")

        if self.os_name == "Linux":
            CMD_get_gpu_num = 'nvidia-smi -L | wc -l'
            self.gpu_all_nums = int(os.popen(CMD_get_gpu_num).read())
            if index_length == 0:
                self.visible_gpus = range(self.gpu_all_nums)
                self.visible_gpus_num = self.gpu_all_nums
            else:
                assert max(gpu_indexes) < self.gpu_all_nums
                self.visible_gpus = gpu_indexes
                self.visible_gpus_num = len(self.visible_gpus)

    def set_detail(self, framework, fraction: float=None, is_auto_increase: bool=True): # include fraction and auto_increase
        if framework == "keras":
            import keras.backend.tensorflow_backend as KTF
            import tensorflow as tf
            config = tf.ConfigProto()
            if fraction is not None:
                config.gpu_options.per_process_gpu_memory_fraction = fraction
            config.gpu_options.allow_growth = is_auto_increase
            session = tf.Session(config=config)
            KTF.set_session(session)
        else:
            raise ValueError("Framework %s is not existed!" % framework)

    def set_specified_gpu(self, gpu_index: int):
        if gpu_index in self.visible_gpus:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        else:
            raise ValueError("gpu_index %d is not visible!")

    def set_best_gpu(self, top_k=1):
        best_gpu = self._scan(top_k)
        if self.os_name == "Linux":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, best_gpu))
            return best_gpu

    def _scan(self, top_k):
        if self.os_name == "Linux":
            CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
            # CMD2 = 'nvidia-smi -L | wc -l'
            CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

            # total_gpu = int(os.popen(CMD2).read())

            assert top_k <= self.visible_gpus_num, 'top_k > visible_gpus_num !'

            # first choose the free gpus
            gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
            free_gpus = set(range(self.gpu_all_nums)) - gpu_usage

            # then choose the most memory free gpus
            gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
            gpu_sorted = list(sorted(range(self.gpu_all_nums), key=lambda x: gpu_free_mem[x], reverse=True))[len(free_gpus):]

            res = list(free_gpus) + list(gpu_sorted)
            res_visible = []
            for v in res:
                if v in self.visible_gpus:
                    res_visible.append(v)
            return res_visible[:top_k]


if __name__ == '__main__':
    my_gpu = SetGpu(visible_gpus=(0, 1, 2))
    res = my_gpu._scan(2)
    print(res)
