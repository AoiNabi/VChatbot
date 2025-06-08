@echo off
echo 🔧 Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate

echo ⬇️ Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Todo listo. Ejecuta con:
echo     venv\Scripts\activate
echo     python main.py
pause