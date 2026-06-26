"""
预下载 easyocr 模型（命令行跑，不限时）
部署后运行一次: python download_ocr_models.py

注意: ch_sim 只能和 en 搭配，ar 需要单独一个 Reader
"""
print("[OCR] 开始下载模型 en + ch_sim ...")
print("      首次约需几分钟，请耐心等待\n")

import easyocr
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

print("\n[OCR] 开始下载模型 ar ...\n")
reader_ar = easyocr.Reader(['ar'], gpu=False)

print("\n[OCR] 模型全部就绪 ✓")
