# Local RTSP test cameras for the RTMPose clone (dev only): MediaMTX on 127.0.0.1:8654 serving ONLY Drowning Dataset
# videos - every runtime\rtsp\IMG_*.mp4 (1080p, 15 fps, MPEG-4 Part 2, made by encode_dd.sh from the clone's own
# "Drowning Dataset" copy) loops as rtsp://127.0.0.1:8654/<name>, e.g. rtsp://127.0.0.1:8654/IMG_5491
$ErrorActionPreference = "Stop"
$here = "C:\aquaguard_rtmpose\runtime\rtsp"  # the encoded clips (git-ignored); mediamtx.yml sits next to this script
$mtx = "C:\aquaguard_rtmpose\build\tools\mediamtx\mediamtx_v1.21.1\mediamtx.exe"
$ffmpeg = (Get-Command ffmpeg).Source
$clips = Get-ChildItem -Path $here -Filter "IMG_*.mp4" | Sort-Object Name
if ($clips.Count -eq 0) { throw "no IMG_*.mp4 in $here - run encode_dd.sh first" }
$server = Start-Process -FilePath $mtx -ArgumentList "`"$PSScriptRoot\mediamtx.yml`"" -WorkingDirectory $here -PassThru -NoNewWindow
Start-Sleep -Seconds 2
$feeds = @()
foreach ($clip in $clips) {
    $name = $clip.BaseName
    $ffArgs = @("-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", "`"$($clip.FullName)`"", "-an",
                "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://127.0.0.1:8654/$name")
    $feeds += Start-Process -FilePath $ffmpeg -ArgumentList $ffArgs -WorkingDirectory $here -PassThru -NoNewWindow
    Write-Host "  rtsp://127.0.0.1:8654/$name"
}
Write-Host "$($clips.Count) Drowning Dataset streams on rtsp://127.0.0.1:8654/  (Ctrl-C stops them)"
try { Wait-Process -Id $server.Id } finally { $feeds + $server | ForEach-Object { if (-not $_.HasExited) { Stop-Process -Id $_.Id -Force } } }
