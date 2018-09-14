# -*- coding: utf-8 -*-
__all__ = ["GpuManager", "__author__", "__version__"]

__author__ = "Fishbone"
__version__ = "1.2.5"

from gpu_control.gpu_utils import Linux_Gpu
import platform


class GpuManager(object):
    def __init__(self, visible_gpus: list=None):
        # Type check
        assert isinstance(visible_gpus, list), "The visible_gpus should be a list"
        self.support_os = ["Linux"]

        self.os_name = platform.system()
        if self.os_name == "Linux":
            self.gpu = Linux_Gpu(visible_gpus)

    def set_by_memory(self, top_k):
        if self.os_name in self.support_os:
            best_gpus = self.gpu.set_by_memory(top_k)
        else:
            best_gpus = None
            print("All the gpus will be used, because for the system of %s is not supported!" % self.os_name)
        return best_gpus

    def set_specified_gpu(self, gpus: list):
        if self.os_name in self.support_os:
            self.gpu.set_specified_gpu(gpus)
        else:
            print("All the gpus will be used, because for the system of %s is not supported!" % self.os_name)


if __name__ == '__main__':
    my_gpu = GpuManager([3, 1, 2])
    res = my_gpu.set_by_memory(2)
    print(res)
