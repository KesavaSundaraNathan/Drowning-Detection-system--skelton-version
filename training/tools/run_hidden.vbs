' Starts the training watchdog with NO window at all (powershell -WindowStyle Hidden still flashes a console for a moment).
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\aquaguard\training\tools\watchdog.ps1""", 0, False
