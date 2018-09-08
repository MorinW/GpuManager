# -*- coding: utf-8 -*-
__author__ = "Fishbone"
__version__ = "0.2.3"

__all__ = ["GpuManager"]

from gpu_control.gpu_utils import Linux_Gpu
import platform


class GpuManager(object):
    def __init__(self, visible_gpus: list=None):
        # Type check
        assert isinstance(visible_gpus, list), "The visible_gpus should be a list"

        os_name = platform.system()
        if os_name == "Linux":
            self.os_name = os_name
            self.gpu = Linux_Gpu(visible_gpus)
        else:
            raise ValueError("System %s is not adapted!" % os_name)


    def set_best_gpu(self, top_k):
        best_gpus = self.gpu.set_best_gpu(top_k)
        return best_gpus

    def set_set_specified_gpu(self, gpus: list):
        self.gpu.set_specified_gpu(gpus)

    # def set_detail(self, fraction: float=None, is_auto_increase: bool=True):
    #     self.framework.set_detail(fraction, is_auto_increase)


if __name__ == '__main__':
    my_gpu = GpuManager([3, 1, 2])
    res = my_gpu.set_best_gpu(2)
    print(res)
