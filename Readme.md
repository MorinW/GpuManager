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

## How to choose gpus?
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
to choose your gpus

## How to set framework?
For example, with the Keras framework.
 - Import the package
 ```python
from gpu_control.framework_setting import set_keras
```
 - Then in your codes, you can write
 ```python
set_keras(fraction, is_auto_increase)
```
to set the fraction and auto_increase for keras.