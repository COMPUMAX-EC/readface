@echo off
echo Activating virtual environment...
CALL venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

echo Running the emotion detection program...
python test_emotion_model.py

echo Script finished.
pause
