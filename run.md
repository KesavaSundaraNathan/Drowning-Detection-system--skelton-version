# Run Life Guard

Windows commands are PowerShell. Linux / WSL equivalents follow each step where they differ.

## 1. Prerequisites

- Python **3.12** (64-bit) — and Python **3.11** for the one-off pose export
- Node.js **22.22+** or **24**
- NVIDIA driver ≥ 527.41 (RTX 4060). No GPU: runs on CPU, fewer cameras.
- Git, FFmpeg

## 2. Get the code

```powershell
git clone https://github.com/KesavaSundaraNathan/RVV-Drowning-Detection-System.git
cd RVV-Drowning-Detection-System
git checkout feature/rtmpose-top-down-pipeline
```

## 3. Python environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / WSL
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

## 4. Models

Weights are not in git. Copy the RTMDet files into `backend\models\`:

```
rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.onnx
rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx
```

Export the pose model once (separate Python 3.11 venv, never used by the server):

These lines run unchanged in both **cmd** and **PowerShell** (prompt `C:\...>` = cmd, `PS C:\...>` = PowerShell). To start over, delete the folder `build\pose_export_venv` first.

```
py -3.11 -m venv build\pose_export_venv
.\build\pose_export_venv\Scripts\python.exe --version
.\build\pose_export_venv\Scripts\python.exe -m ensurepip --upgrade
.\build\pose_export_venv\Scripts\python.exe -m pip install --upgrade pip
.\build\pose_export_venv\Scripts\python.exe -m pip install setuptools wheel
.\build\pose_export_venv\Scripts\python.exe -m pip install --no-build-isolation chumpy==0.70
.\build\pose_export_venv\Scripts\python.exe -m pip install -r requirements\pose_export.in
curl.exe -LO https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth
.\build\pose_export_venv\Scripts\python.exe tools\export_rtmpose.py --checkpoint rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth --images build\v2_baseline\Datasets --det-model backend\models\rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.onnx --register
```

The `--version` line must print `Python 3.11.x`.

Python 3.11 is required (torch 2.1 has no 3.12 wheels): `py -0` lists what is installed, `winget install Python.Python.3.11` adds it.

Optional - the pool / office scene model (CLIP ViT-B/32, MIT; downloads ~340 MB once, writes `backend\models\clip_vit_b32_scene.onnx`, ~350 MB). Without it the Cameras page can only ever suggest "Pool area"; with it, it also suggests "Standard CCTV" for cameras that see no pool. Same export venv, same lines in cmd and PowerShell:

```
.\build\pose_export_venv\Scripts\python.exe -m pip install ftfy regex tqdm
.\build\pose_export_venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
.\build\pose_export_venv\Scripts\python.exe tools\export_scene_clip.py --register
```

Verify:

```powershell
python tools\verify_models.py
```

## 5. TensorRT (optional, fastest)

TensorRT comes from NVIDIA's own index, not PyPI (the `tensorrt-cu12` package on PyPI is an empty stub). In the main `.venv`, same lines in cmd and PowerShell:

```
pip uninstall -y tensorrt-cu12
pip install --no-deps --index-url https://pypi.nvidia.com tensorrt-cu12-libs==10.2.0.post1
python -c "import tensorrt_libs, os; print(os.listdir(os.path.dirname(tensorrt_libs.__file__)))"
python tools\build_trt_engines.py --data-dir C:\LifeGuardDev --cameras 4 --fps 15
```

The `python -c` line must list `nvinfer_10.dll`. The server adds that folder to the DLL search path itself; no PATH edit.
Prints the latency table and whether 4 cameras fit at 15 fps.

```bash
# Linux / WSL
pip install --no-deps --index-url https://pypi.nvidia.com tensorrt-cu12-libs==10.2.0.post1
export LD_LIBRARY_PATH=$(python -c 'import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))'):$LD_LIBRARY_PATH
```

## 6. Development run

Set once per terminal:

```powershell
$env:PYTHONPATH = "backend"
$env:LIFEGUARD_ENV = "development"
$env:LIFEGUARD_DATA_DIR = "C:\LifeGuardDev"
$env:LIFEGUARD_LIVE_ENABLED = "true"
$env:LIFEGUARD_SUPERVISOR_EXPECTED = "false"
```

```bat
:: cmd.exe
set PYTHONPATH=backend
set LIFEGUARD_ENV=development
set LIFEGUARD_DATA_DIR=C:\LifeGuardDev
set LIFEGUARD_LIVE_ENABLED=true
set LIFEGUARD_SUPERVISOR_EXPECTED=false
```

```bash
# Linux / WSL
export PYTHONPATH=backend LIFEGUARD_ENV=development LIFEGUARD_DATA_DIR=$HOME/lifeguard-dev \
       LIFEGUARD_LIVE_ENABLED=true LIFEGUARD_SUPERVISOR_EXPECTED=false
```

First time only:

```powershell
python -m lifeguard keys init
python -m lifeguard db upgrade
python -m lifeguard admin create --username admin
```

Then two terminals:

```powershell
# 1 - API
python -m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8000

# 2 - UI
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173** → sign in → **Arm audio** → **Cameras**.

Each terminal needs the `.venv` activated and the variables above set (the UI terminal only needs Node).

### Check it works

1. `http://127.0.0.1:8000/api/v1/health/ready` returns `{"status": "ready"}`.
2. **Cameras**: add an RTSP camera (or a simulated one, `tools/rtsp_sim/README.md`) and start it; the tile shows live boxes.
3. API terminal: no `ProviderError`, no `pose model` fault. With TensorRT installed the first start is slow (engine build); later ones reuse the cache.

## 7. Production install

```powershell
# Windows (elevated PowerShell)
cd frontend; npm ci; npm run build; cd ..
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -AdminUser pool.admin
tools\start_lifeguard.ps1
tools\stop_lifeguard.ps1
```

```bash
# Linux / WSL
(cd frontend && npm ci && npm run build)
sudo deploy/linux/install.sh --admin-user pool.admin
sudo systemctl start lifeguard-api lifeguard-supervisor
sudo systemctl stop lifeguard-supervisor lifeguard-api
```

Nothing starts at boot unless enabled (`-StartMode Automatic` / `--enable`).

## 8. Tests

```powershell
pytest -m "not gpu and not soak and not chaos"
pytest -m gpu
cd frontend; npm run typecheck; npm run lint; npm test
```

## 9. Held-out evaluation (pose off vs on)

```powershell
python tools\pose_eval.py --eval-dir eval\heldout_v1
```

Results: `build\pose_eval\` (default `--work-dir`).

## Useful switches

| Variable | Values |
|---|---|
| `LIFEGUARD_INFERENCE_PROVIDER` | `auto` (default), `tensorrt`, `cuda`, `openvino`, `cpu` |
| `LIFEGUARD_POSE_MODEL` | empty = default RTMPose, a path, or `none` |
| `LIFEGUARD_BEHAVIOR_MODEL` | empty = default learned model, a path, or `none` |

More detail: `docs/RUNBOOK.md`.
