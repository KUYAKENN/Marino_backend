@echo off
echo ========================================
echo FACE RECOGNITION SYSTEM STARTUP
echo ========================================
echo.

REM Check if Python is available
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "src\flask_app.py" (
    echo ERROR: src\flask_app.py not found
    echo Please ensure all face recognition files are present
    pause
    exit /b 1
)

if not exist "QUICK_SETUP.sql" (
    echo ERROR: QUICK_SETUP.sql not found
    echo Please ensure the database setup file is present
    pause
    exit /b 1
)

echo 1. Flask Server Setup
echo 2. Face Database Scanner
echo 3. Complete System Test
echo.
set /p choice="Select option (1-3): "

if "%choice%"=="1" goto start_server
if "%choice%"=="2" goto scan_faces
if "%choice%"=="3" goto test_system
goto invalid_choice

:start_server
echo.
echo ========================================
echo STARTING FLASK SERVER
echo ========================================
echo Server will run on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python src\flask_app.py
goto end

:scan_faces
echo.
echo ========================================
echo RUNNING FACE SCANNER
echo ========================================
echo This will scan existing face images from your database
echo.
cd /d "%~dp0"
python scan_faces.py
goto end

:test_system
echo.
echo ========================================
echo TESTING COMPLETE SYSTEM
echo ========================================
echo This will test all face recognition endpoints
echo.
cd /d "%~dp0"
if exist "test_complete_system.py" (
    python test_complete_system.py
) else (
    echo test_complete_system.py not found
    echo Running basic face scanner instead...
    python scan_faces.py
)
goto end

:invalid_choice
echo Invalid choice. Please select 1, 2, or 3
pause
goto end

:end
echo.
echo ========================================
echo OPERATION COMPLETE
echo ========================================
pause
