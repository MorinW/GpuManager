# GpuManager
It is used to choose gpu to run for AI.

## How to install it?
 - Download it, and then run
 ```python
python setup.py build
python setup.py install
```
 - Use pip by running code
```bash
pip install git+https://github.com/DogfishBone/GpuManager.git
```

## How to use it?
 - Import the package
 ```python
from gpu_control.gpu_manager import GpuManager
```
 - Create a entry by
 ```python
my_gpu = GpuManager(visible_gpus)
```
 - Then run
 ```python
res = my_gpu.set_best_gpu(top_k)
```
to choose your gpu
