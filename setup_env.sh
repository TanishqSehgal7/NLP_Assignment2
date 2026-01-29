#!/bin/bash
echo "Creating Python virtual environment..."
python3 -m venv nlp_env

echo "Activating virtual environment..."
source nlp_env/bin/activate

echo "Installing packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=nlp_env --display-name="NLP Environment"

echo ""
echo "Setup complete!"
echo "To use the environment:"
echo "1. Run: source nlp_env/bin/activate"
echo "2. Start Jupyter: jupyter notebook"
echo "3. Select 'NLP Environment' kernel in the notebook"