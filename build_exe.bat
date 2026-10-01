@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=gesture_env\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Could not find the project environment at %PYTHON%
    echo Create it and install requirements.txt before building.
    pause
    exit /b 1
)

"%PYTHON%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing the EXE builder into the project environment...
    "%PYTHON%" -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

echo Building AerosignGesture.exe...
"%PYTHON%" -m PyInstaller --noconfirm --clean Aerosign.spec
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build complete: %CD%\dist\AerosignGesture.exe
echo Copy that EXE to a Windows PC and double-click it to run.
pause
