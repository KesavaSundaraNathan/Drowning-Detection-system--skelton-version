# Life Guard RTMPose clone - UI (Vite dev server) on http://localhost:5273, proxying /api to the clone's API on :8100.
Set-Location "C:\aquaguard_rtmpose\frontend"
$env:LIFEGUARD_DEV_API_PORT = "8100"
npm run dev -- --port 5273 --strictPort
