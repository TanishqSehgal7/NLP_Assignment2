@echo off
echo Activating NLP Environment...
call nlp_env\Scripts\activate.bat
echo Environment activated! You can now run:
echo   jupyter notebook
echo   python your_script.py
echo.
echo To deactivate, type: deactivate
cmd /k