# Life Guard RTMPose clone - API on 127.0.0.1:8100 with this clone's own data folder (never C:\LifeGuardDev).
Set-Location "C:\aquaguard_rtmpose"
$env:PYTHONPATH = "backend"
$env:LIFEGUARD_ENV = "development"
$env:LIFEGUARD_DATA_DIR = "C:\aquaguard_rtmpose\runtime"
$env:LIFEGUARD_TRT_CACHE_DIR = "C:\aquaguard_rtmpose\runtime\trt_cache"
$env:LIFEGUARD_LIVE_ENABLED = "true"
$env:LIFEGUARD_SUPERVISOR_EXPECTED = "false"
$env:LIFEGUARD_SUPERVISOR_API_URL = "http://127.0.0.1:8100"
$env:LIFEGUARD_STARTUP_MAX_CAMERAS = "4"
& "C:\aquaguard_rtmpose\venv\Scripts\python.exe" -m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8100
