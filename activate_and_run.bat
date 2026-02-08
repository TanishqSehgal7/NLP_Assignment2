@echo off
echo Activating CNN environment...
call cnn_env\Scripts\activate.bat

echo Environment activated successfully!
echo.
echo Available commands:
echo   jupyter notebook    - Start Jupyter Notebook
echo   python              - Run Python interpreter
echo   pip list            - Show installed packages
echo.
echo To run your notebook:
echo   jupyter notebook 2025AA05036_cnn_assignment.ipynb
echo.

cmd /k