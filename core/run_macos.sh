#!/usr/bin/env bash
set -euo pipefail

# Перейти в папку проекта (где лежит этот скрипт)
cd "$(dirname "$0")"

# Проверка Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 не найден. Установите Python 3.11+."
  exit 1
fi

# Создание виртуального окружения, если нет
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Активация
source .venv/bin/activate

# Обновим pip (без фанатизма, но полезно)
python -m pip install --upgrade pip

# Установка зависимостей
python -m pip install -r requirements.txt

# Запуск
python app.py