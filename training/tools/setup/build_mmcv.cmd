@echo off
rem Build MMCV 2.1.0 with CUDA ops for the RTX 5060 (sm_120) into training\.venv. MMDetection 3.3 / MMPose 1.3 need
rem mmcv < 2.2. Uses VS 2022 Build Tools (cl.exe) + CUDA Toolkit 12.8 (nvcc), matching torch 2.7.1+cu128.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0
set MMCV_WITH_OPS=1
set FORCE_CUDA=1
set DISTUTILS_USE_SDK=1
set MAX_JOBS=12
cd /d C:\lifeguard\training\third_party || exit /b 1
if not exist mmcv git clone -q --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git mmcv
cd mmcv || exit /b 1
C:\lifeguard\training\.venv\Scripts\python.exe -m pip install -q ninja==1.13.0 setuptools==75.8.0 wheel packaging || exit /b 1
C:\lifeguard\training\.venv\Scripts\python.exe -m pip install -v --no-build-isolation -e . || exit /b 1
C:\lifeguard\training\.venv\Scripts\python.exe -c "import torch, mmcv; from mmcv.ops import nms, RoIAlign; b=torch.rand(50,4,device='cuda')*100; b[:,2:]+=b[:,:2]; s=torch.rand(50,device='cuda'); print('mmcv', mmcv.__version__, 'cuda nms ok', nms(b, s, 0.5)[0].shape)"
