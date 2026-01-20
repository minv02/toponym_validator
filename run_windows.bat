@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Проверка python
python --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python не найден. Установите Python 3.11+ и отметьте "Add Python to PATH".
  pause
  exit /b 1
)

REM Создание venv
if not exist ".venv" (
  python -m venv .venv
)

REM Активация venv
call ".venv\\Scripts\\activate.bat"

REM Обновление pip
python -m pip install --upgrade pip

REM Установка зависимостей
python -m pip install -r requirements.txt

REM Запуск приложения
python app.py

pause