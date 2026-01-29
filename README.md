# Sentiment Analysis with Attention Mechanisms

## Setup Instructions

### 1. Environment Setup (Already Done!)
The Python virtual environment `nlp_env` has been created and all packages installed.

### 2. Activate Environment
**Windows:**
```bash
# Option 1: Use the activation script
activate_env.bat

# Option 2: Manual activation
nlp_env\Scripts\activate.bat
```

**Unix/Linux/Mac:**
```bash
source nlp_env/bin/activate
```

### 3. Start Jupyter Notebook
```bash
jupyter notebook
```

### 4. Select Kernel
- Open `sentiment_analysis_attention.ipynb`
- In Jupyter, go to Kernel → Change Kernel → Select "NLP Environment"

## Installed Packages
- torch>=1.9.0 (PyTorch for deep learning)
- transformers>=4.20.0 (Hugging Face transformers)
- pandas>=1.3.0 (Data manipulation)
- numpy>=1.21.0 (Numerical computing)
- matplotlib>=3.4.0 (Plotting)
- seaborn>=0.11.0 (Statistical visualization)
- scikit-learn>=1.0.0 (Machine learning)
- jupyter>=1.0.0 (Jupyter notebook)
- ipykernel (Jupyter kernel)
- tqdm (Progress bars)
- plotly (Interactive plots)

## Files Structure
```
Assignment2/
├── sentiment_analysis_attention.ipynb  # Main notebook
├── training.1600000.processed.noemoticon.csv  # Dataset
├── requirements.txt  # Package dependencies
├── setup_env.py  # Environment setup script
├── activate_env.bat  # Environment activation script
├── nlp_env/  # Virtual environment folder
└── README.md  # This file
```

## Usage
1. Activate the environment using `activate_env.bat`
2. Start Jupyter: `jupyter notebook`
3. Open the notebook and select "NLP Environment" kernel
4. Run all cells to execute the sentiment analysis pipeline

## Assignment Tasks Covered
- ✅ Task 1: Data Preprocessing (1 mark)
- ✅ Task 2: Baseline Model (2 marks)  
- ✅ Task 3: Attention-Based Model (3.5 marks)
- ✅ Task 4: Comparative Analysis (3.5 marks)

Total: 10 marks