@echo off
rem STROKE tasks: SAM 2 follows the people you boxed by hand. Save and close the CVAT job tabs before running.
cd /d C:\lifeguard
training\.venv\Scripts\python.exe training\tools\follow_stroke.py
pause
