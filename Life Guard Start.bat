@echo off
setlocal EnableExtensions
rem Life Guard Start (RTMPose skeleton clone) - API (port 8100) and UI (port 5273) in two windows.
rem Independent of C:\aquaguard: own venv, own data folder (keys, database, cameras, TensorRT engines), own ports, so
rem both versions can run side by side. Works from any Windows account.
rem First run on a data folder: creates the protected keys, the database and the first admin (asks for a username and
rem a password, typed blind). Later runs: checks the schema (backs it up and migrates it when this build needs a newer one).
cd /d "%~dp0"
set "ROOT=%CD%"
echo Account: %USERNAME%   Project: %ROOT%

set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" goto :no_venv
if not exist "%ROOT%\frontend\node_modules" goto :no_modules

rem Data dir: ALWAYS this clone's own folder (never the original's C:\LifeGuardDev, even if the account sets
rem LIFEGUARD_DATA_DIR globally). LIFEGUARD_RTMPOSE_DATA_DIR overrides it. Keys are bound to the Windows account, so
rem the first start creates this clone's own keys, database and admin; add the cameras on its Cameras page.
set "LG_API_PORT=8100"
set "LG_UI_PORT=5273"
if defined LIFEGUARD_RTMPOSE_DATA_DIR (set "LIFEGUARD_DATA_DIR=%LIFEGUARD_RTMPOSE_DATA_DIR%") else set "LIFEGUARD_DATA_DIR=%ROOT%\runtime"
set "LIFEGUARD_TRT_CACHE_DIR=%LIFEGUARD_DATA_DIR%\trt_cache"
set "LIFEGUARD_DEV_API_PORT=%LG_API_PORT%"
set "LIFEGUARD_SUPERVISOR_API_URL=http://127.0.0.1:%LG_API_PORT%"
if not exist "%LIFEGUARD_DATA_DIR%\" mkdir "%LIFEGUARD_DATA_DIR%" >nul 2>&1
if not exist "%LIFEGUARD_DATA_DIR%\" goto :no_data
type nul > "%LIFEGUARD_DATA_DIR%\.write_test" 2>nul
if not exist "%LIFEGUARD_DATA_DIR%\.write_test" goto :no_write
del "%LIFEGUARD_DATA_DIR%\.write_test" >nul 2>&1
echo Data folder: %LIFEGUARD_DATA_DIR%

rem Node.js: PATH first, then the usual install places.
where node >nul 2>&1 && goto :node_ok
for %%D in ("%ProgramFiles%\nodejs" "%ProgramW6432%\nodejs" "%ProgramFiles(x86)%\nodejs" "%LOCALAPPDATA%\Programs\nodejs" "%APPDATA%\nvm\current" "%NVM_SYMLINK%") do if exist "%%~D\node.exe" set "PATH=%%~D;%PATH%"
where node >nul 2>&1 || goto :no_node
:node_ok

set "PYTHONPATH=backend"
set "LIFEGUARD_ENV=development"
set "LIFEGUARD_LIVE_ENABLED=true"
set "LIFEGUARD_SUPERVISOR_EXPECTED=false"
rem Only the first 4 enabled cameras start automatically; start others from the Cameras page. 0 = all.
if not defined LIFEGUARD_STARTUP_MAX_CAMERAS set "LIFEGUARD_STARTUP_MAX_CAMERAS=4"

rem 1. Keys (Windows DPAPI, this account). Created once and never overwritten: a new credential key would orphan every
rem    encrypted camera source, a new session key would log everyone out.
set "KEYS=%LIFEGUARD_DATA_DIR%\keys"
if exist "%KEYS%\session.key.dpapi" if exist "%KEYS%\credentials.key.dpapi" goto :keys_ok
if exist "%KEYS%\session.key.dpapi" goto :keys_partial
if exist "%KEYS%\credentials.key.dpapi" goto :keys_partial
echo First run on this data folder: creating the protected keys...
"%PY%" -m lifeguard keys init || goto :fail_keys
:keys_ok

rem 2. Schema: the API refuses to start unless the database is at the head revision of this build.
set "HAD_DB="
dir /b "%LIFEGUARD_DATA_DIR%\db\*.db" >nul 2>&1 && set "HAD_DB=1"
"%PY%" -m lifeguard config check >nul 2>&1 && goto :schema_ok
echo Preparing the database...
if defined HAD_DB (call :backup_db || goto :fail_backup)
"%PY%" -m lifeguard db upgrade || goto :fail_db
"%PY%" -m lifeguard config check || goto :fail_config
:schema_ok

rem 3. First admin (the password is typed blind, nothing is echoed).
"%PY%" -m lifeguard admin exists >nul 2>&1 && goto :admin_ok
echo.
echo No admin account yet - creating the first one.
set "LG_ADMIN="
set /p "LG_ADMIN=Admin username: "
if not defined LG_ADMIN goto :fail_admin
"%PY%" -m lifeguard admin create --username "%LG_ADMIN%" || goto :fail_admin
:admin_ok

rem cmd /k windows inherit the variables above and need no PowerShell script permission.
start "Life Guard API" /D "%ROOT%" cmd /k ""%PY%" -m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port %LG_API_PORT%"
start "Life Guard UI" /D "%ROOT%\frontend" cmd /k "npm run dev -- --port %LG_UI_PORT% --strictPort"
echo Started API (http://127.0.0.1:%LG_API_PORT%) and UI (http://localhost:%LG_UI_PORT%). Max cameras at start-up: %LIFEGUARD_STARTUP_MAX_CAMERAS%.
echo The first start can take a few minutes while the TensorRT engines are built; later starts reuse %LIFEGUARD_DATA_DIR%\trt_cache.

rem 4. Open the UI in the default browser as soon as it answers (waits up to 2 minutes).
echo Waiting for the UI, then opening the browser...
set /a TRIES=0
:wait_ui
curl -s -o nul --max-time 2 http://localhost:%LG_UI_PORT% && goto :open_ui
set /a TRIES+=1
if %TRIES% geq 60 goto :open_ui
timeout /t 2 >nul
goto :wait_ui
:open_ui
start "" "http://localhost:%LG_UI_PORT%"
timeout /t 3 >nul
exit /b 0

:backup_db
"%PY%" -m lifeguard db backup && exit /b 0
echo [WARNING] The online backup failed - copying the database file instead.
if not exist "%LIFEGUARD_DATA_DIR%\backups\" mkdir "%LIFEGUARD_DATA_DIR%\backups"
copy /y "%LIFEGUARD_DATA_DIR%\db\*.db" "%LIFEGUARD_DATA_DIR%\backups\" >nul
exit /b %ERRORLEVEL%

:no_venv
echo [ERROR] %ROOT%\venv\Scripts\python.exe not found - the Python venv is missing for this copy of the project.
goto :fail
:no_modules
echo [ERROR] %ROOT%\frontend\node_modules not found - run "npm install" in %ROOT%\frontend first.
goto :fail
:no_data
echo [ERROR] Data folder %LIFEGUARD_DATA_DIR% does not exist and this account cannot create it.
echo         Create it as an administrator, or set LIFEGUARD_DATA_DIR to a folder this account can write to.
echo         NOTE: the database, camera list and keys live in that folder - a new empty folder starts with no cameras.
goto :fail
:no_write
echo [ERROR] This account cannot write to %LIFEGUARD_DATA_DIR%. Give it Modify rights on that folder.
goto :fail
:no_node
echo [ERROR] Node.js not found for this account (looked in PATH, Program Files, %%LOCALAPPDATA%%\Programs\nodejs, nvm).
echo         Install Node.js for all users, or add its folder to PATH.
goto :fail
:keys_partial
echo [ERROR] Only one of the two key files exists in %KEYS%.
echo         Not creating new ones: that would orphan the encrypted camera sources or log everyone out.
echo         Restore the missing file from a backup, or move both files away to start with a new, empty setup.
goto :fail
:fail_keys
echo [ERROR] Could not create the protected keys in %KEYS%.
goto :fail
:fail_backup
echo [ERROR] The database could not be backed up, so it was not migrated. Free disk space or check %LIFEGUARD_DATA_DIR%\backups.
goto :fail
:fail_db
echo [ERROR] The database migration failed. The backup is in %LIFEGUARD_DATA_DIR%\backups.
goto :fail
:fail_config
echo [ERROR] The settings or keys cannot be loaded. Keys are bound to the Windows account that created them:
echo         start Life Guard from that account, or restore the key files for this one.
goto :fail
:fail_admin
echo [ERROR] The first admin account was not created. Start this file again.
goto :fail
:fail
echo.
pause
exit /b 1
