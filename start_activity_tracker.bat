@echo off
REM Activate the virtual environment
call ..\activityEnv\Scripts\activate.bat

REM Start the application
python main.py

REM Keep the window open after the script finishes (optional)
pause