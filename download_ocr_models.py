"""
预下载 easyocr 模型（命令行跑，不限时）
部署后运行一次: python download_ocr_models.py
"""
print("[OCR] 开始下载模型 en + ar + ch_sim ...")
print("      首次约需几分钟，请耐心等待\n")

import easyocr
reader = easyocr.Reader(['en', 'ar', 'ch_sim'], gpu=False)

print("\n[OCR] 模型全部就绪 ✓")
