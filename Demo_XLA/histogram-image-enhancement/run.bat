@echo off
chcp 65001 > nul
echo =====================================================================
echo  KHOI CHAY UNG DUNG: HISTOGRAM VA TANG CUONG CHAT LUONG ANH
echo =====================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Kich hoat moi truong ao .venv...
    call .venv\Scripts\activate.bat
)

echo Dang khoi chay Streamlit...
python -m streamlit run app.py
pause
