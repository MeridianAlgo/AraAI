@echo off
REM Ara AI Stock Analysis Platform - Windows Installer
REM Enhanced security and error handling

setlocal EnableDelayedExpansion

echo.
echo ========================================
echo  Ara AI Stock Analysis Platform
echo  Windows Installation Script
echo ========================================
echo.

REM Check if running as administrator (optional, not required)
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with administrator privileges
) else (
    echo [INFO] Running with standard user privileges
    echo [INFO] This is fine - no admin rights needed
)

echo.
echo [1/6] Checking Python installation...

REM Check Python installation
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from: https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

echo.
echo [2/6] Checking pip installation...

python -m pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip is not available
    echo [INFO] Installing pip...
    python -m ensurepip --upgrade
    if %errorLevel% neq 0 (
        echo [ERROR] Failed to install pip
        pause
        exit /b 1
    )
)

echo [SUCCESS] pip is available

echo.
echo [3/6] Creating Ara AI directory structure...

REM Create directories in user's Documents folder (no admin rights needed)
set "ARA_DIR=%USERPROFILE%\Documents\AraAI"
set "DATA_DIR=%ARA_DIR%\data"
set "CACHE_DIR=%ARA_DIR%\cache"
set "LOGS_DIR=%ARA_DIR%\logs"
set "CONFIG_DIR=%ARA_DIR%\config"

if not exist "%ARA_DIR%" mkdir "%ARA_DIR%"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

echo [SUCCESS] Directory structure created at: %ARA_DIR%

echo.
echo [4/6] Installing MeridianAlgo package...

REM Install with user flag to avoid permission issues
python -m pip install --user --upgrade meridianalgo
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install MeridianAlgo package
    echo [INFO] Trying alternative installation method...
    python -m pip install --upgrade meridianalgo
    if %errorLevel% neq 0 (
        echo [ERROR] Installation failed completely
        pause
        exit /b 1
    )
)

echo [SUCCESS] MeridianAlgo package installed

echo.
echo [5/6] Creating configuration files...

REM Create config file
echo # Ara AI Configuration > "%CONFIG_DIR%\ara_config.ini"
echo [DEFAULT] >> "%CONFIG_DIR%\ara_config.ini"
echo data_dir = %DATA_DIR% >> "%CONFIG_DIR%\ara_config.ini"
echo cache_dir = %CACHE_DIR% >> "%CONFIG_DIR%\ara_config.ini"
echo logs_dir = %LOGS_DIR% >> "%CONFIG_DIR%\ara_config.ini"
echo verbose = false >> "%CONFIG_DIR%\ara_config.ini"
echo cache_enabled = true >> "%CONFIG_DIR%\ara_config.ini"
echo max_cache_age_hours = 24 >> "%CONFIG_DIR%\ara_config.ini"

REM Create desktop shortcut (optional)
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Ara AI.lnk"
echo [INFO] Creating desktop shortcut...

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = 'cmd.exe'; $Shortcut.Arguments = '/k ara --help'; $Shortcut.WorkingDirectory = '%ARA_DIR%'; $Shortcut.IconLocation = 'cmd.exe,0'; $Shortcut.Description = 'Ara AI Stock Analysis Platform'; $Shortcut.Save()" 2>nul

echo [SUCCESS] Configuration files created

echo.
echo [6/6] Testing installation...

REM Test the installation
echo [INFO] Testing ara command...
ara --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] 'ara' command not found in PATH
    echo [INFO] You may need to restart your command prompt
    echo [INFO] Or use: python -m meridianalgo.cli instead
) else (
    echo [SUCCESS] ara command is working
)

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Ara AI has been installed successfully!
echo.
echo Installation Directory: %ARA_DIR%
echo.
echo Usage:
echo   ara AAPL --days 5        # Predict AAPL for 5 days
echo   ara --system-info        # Show system information
echo   ara --help               # Show all commands
echo.
echo Alternative usage (if ara command not found):
echo   python -c "from meridianalgo import quick_predict; print(quick_predict('AAPL', 5))"
echo.
echo For support: https://github.com/MeridianAlgo/Ara
echo.

pause