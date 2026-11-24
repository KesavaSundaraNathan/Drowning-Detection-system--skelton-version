#!/bin/bash
# Drowning Dataset clips -> 1080p, 15 fps, MPEG-4 Part 2 (its VOL header carries 15 fps exactly; x264 timing makes the short RTSP probe read 30). Source: the clone's own copy.
mkdir -p C:/aquaguard_rtmpose/runtime/rtsp && cd C:/aquaguard_rtmpose/runtime/rtsp
for n in IMG_5488 IMG_5490 IMG_5491 IMG_5492 IMG_5493 IMG_5495 IMG_5498 IMG_5502 IMG_5504 IMG_5506 IMG_5508 IMG_5510 IMG_5511 IMG_5515 IMG_5516; do
  ffmpeg -hide_banner -loglevel error -y -i "C:/aquaguard_rtmpose/Drowning Dataset/$n.mov" \
    -vf "scale=1920:1080:flags=area,fps=15" -r 15 -c:v mpeg4 -q:v 3 -g 30 -an "$n.mp4" \
    && echo "$n ok" || echo "$n FAILED"
done
