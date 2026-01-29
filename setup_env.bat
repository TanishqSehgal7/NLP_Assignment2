@echo off
echo Creating Python virtual environment...
python -m venv nlp_env

echo Activating virtual environment...
call nlp_env\Scripts\activate.bat

echo Installing packages from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

echo Registering Jupyter kernel...
python -m ipykernel install --user --name=nlp_env --display-name="NLP Environment"

echo.
echo Setup complete! 
echo To use the environment:
echo 1. Run: nlp_env\Scripts\activate.bat
echo 2. Start Jupyter: jupyter notebook
echo 3. Select "NLP Environment" kernel in the notebook
pause