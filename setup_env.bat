@echo off
echo Setting up CNN Assignment Environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv cnn_env

REM Activate virtual environment
echo Activating virtual environment...
call cnn_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete! 
echo To activate environment: cnn_env\Scripts\activate.bat
echo To run jupyter: jupyter notebook
pause