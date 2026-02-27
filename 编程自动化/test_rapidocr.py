#!/usr/bin/env python3
from rapidocr_onnxruntime import RapidOCR
from pathlib import Path

engine = RapidOCR()
img_path = Path(r"E:\ai知识库\nlp大赛\原始语料\国家博物馆\IMG20260121104914.jpg")

result = engine(str(img_path))
print("Result type:", type(result))
print("\nFull result:")
print(result)
