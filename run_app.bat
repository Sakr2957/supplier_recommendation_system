@echo off
REM Quick Start Script for Windows

echo ==========================================
echo Supplier Recommendation System
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

echo Python found
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo Dependencies installed
echo.

REM Ask about tests
set /p RUNTESTS="Run system tests before starting? (y/n): "
if /i "%RUNTESTS%"=="y" (
    echo Running tests...
    python test_system.py
    if errorlevel 1 (
        echo.
        set /p CONTINUE="Some tests failed. Continue anyway? (y/n): "
        if /i not "%CONTINUE%"=="y" exit /b 1
    )
)

REM Start Streamlit
echo.
echo ==========================================
echo Starting Streamlit App...
echo ==========================================
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
