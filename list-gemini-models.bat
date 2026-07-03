@echo off
REM Lists the Gemini model ids your API key can use. Put your key in .env first,
REM then copy one id (without the "models/" prefix) into GEMINI_MODEL in .env.
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -c "import os; from dotenv import load_dotenv; load_dotenv(); from google import genai; k=os.getenv('GEMINI_API_KEY'); print('No GEMINI_API_KEY in .env') if not k else [print(m.name) for m in genai.Client(api_key=k).models.list()]"
echo.
pause
