@echo off
setlocal enabledelayedexpansion
title Technical Docs AI Assistant

cd /d "%~dp0"

echo ============================================================
echo   Technical Docs AI Assistant - Starting up
echo ============================================================
echo.

REM ---------- 1. Detect Python ----------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python was not found on this machine.
    echo         Install Python 3.10 or newer from https://www.python.org/downloads/
    echo         During install, tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
echo [1/6] Python detected.

REM ---------- 2. Create the virtual environment ----------
set "VENV_PY=backend\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [2/6] Creating virtual environment ^(first run, this takes a moment^)...
    %PY% -m venv backend\.venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/6] Virtual environment already present.
)

REM ---------- 3. Install dependencies ----------
echo [3/6] Installing dependencies...
"%VENV_PY%" -m pip install --quiet --upgrade pip
"%VENV_PY%" -m pip install --quiet -r backend\requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

REM ---------- 4. Check for the API key ----------
if not exist "backend\.env" (
    echo.
    echo [4/6] No API key configured yet.
    echo       Get a free Gemini API key at: https://aistudio.google.com/app/apikey
    echo.
    set /p "APIKEY=      Paste your Gemini API key here: "
    if "!APIKEY!"=="" (
        echo [ERROR] No key entered. Cannot continue.
        pause
        exit /b 1
    )
    >  "backend\.env" echo GOOGLE_API_KEY=!APIKEY!
    >> "backend\.env" echo CHAT_MODEL=gemini-3.1-flash-lite
    >> "backend\.env" echo EMBEDDING_MODEL=gemini-embedding-001
    >> "backend\.env" echo TOP_K=5
    >> "backend\.env" echo CHUNK_SIZE=1000
    >> "backend\.env" echo CHUNK_OVERLAP=150
    >> "backend\.env" echo MAX_FILE_SIZE_MB=20
    >> "backend\.env" echo RELEVANCE_THRESHOLD=0.60
    echo       Saved to backend\.env
) else (
    echo [4/6] API key file found.
)

REM ---------- 5. Start the backend ----------
echo [5/6] Starting the server...
start "Technical Docs AI Assistant - Server" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

REM ---------- 6. Wait for health, then open the browser ----------
echo [6/6] Waiting for the server to come up...
set "READY="
for /l %%i in (1,1,30) do (
    if not defined READY (
        timeout /t 1 /nobreak >nul
        curl -s -o nul http://127.0.0.1:8000/api/health 2>nul && set "READY=1"
    )
)

if not defined READY (
    echo.
    echo [WARN] The server did not respond in time. Check the server window for errors.
    echo        If it is just slow, open http://127.0.0.1:8000 manually.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Ready! Opening http://127.0.0.1:8000
echo.
echo   Sample documents to try are in the sample_docs folder.
echo   Close the server window to stop the application.
echo ============================================================
start "" http://127.0.0.1:8000
timeout /t 3 /nobreak >nul
exit /b 0
