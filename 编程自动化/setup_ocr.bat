@echo off
chcp 65001 >nul
echo ==========================================
echo 博物馆OCR环境安装脚本
echo ==========================================
echo.

echo [1/3] 检查Python版本...
python --version
echo.

echo [2/3] 安装 PaddleOCR 及相关依赖...
echo 这可能需要几分钟时间，请耐心等待...
echo.

pip install paddleocr paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [3/3] 安装其他依赖...
pip install numpy pillow

echo.
echo ==========================================
echo 安装完成！
echo ==========================================
echo.
echo 现在可以运行 OCR 处理脚本：
echo   python ocr_processor.py
echo.
pause
