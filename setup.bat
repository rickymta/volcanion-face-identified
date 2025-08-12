@echo off

echo Setting up Volcanion Face Identified project...

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs

REM Set up MongoDB (if running locally)
echo Note: Make sure MongoDB is running on localhost:27017
echo You can start MongoDB with: mongod

REM Run tests
echo Running tests...
python -m pytest tests/ -v

echo Setup complete!
echo To start the application:
echo uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
