@echo off
echo Checking Python version...
python --version
pause

echo Creating virtual environment if needed...
python -m venv venv
pause

echo Activating virtual environment...
call venv\Scripts\activate.bat
pause

echo Upgrading pip...
pip install --upgrade pip
pause

echo Installing requirements...
pip install -r requirements.txt
pause

echo Moving to app directory...
cd app
pause

echo Running Streamlit app...
streamlit run main.py
pause
