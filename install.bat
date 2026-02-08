@echo off
echo Installing CNN Assignment Dependencies...

REM Run setup script
call setup_env.bat

REM Install dependencies
echo Installing packages from requirements.txt...
cnn_env\Scripts\pip.exe install -r requirements.txt

echo.
echo Installation complete!
echo Run: cnn_env\Scripts\activate.bat to activate environment
echo Then: jupyter notebook to start Jupyter