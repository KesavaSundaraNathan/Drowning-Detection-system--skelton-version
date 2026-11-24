@echo off
rem TEST C: SAM 2 follows the people you boxed by hand. Save and close the CVAT job tab before running.
cd /d C:\lifeguard
training\.venv\Scripts\python.exe training\tools\sam2_follow.py
pause
