@echo off
chcp 65001 >nul
echo ======================================
echo 博物馆平行语料OCR处理
echo 处理馆: 党史馆 + 抗日纪念馆
echo ======================================
cd /d "%~dp0"
python ocr_processor.py
pause
