"""
预下载 easyocr 模型，在部署后运行一次即可：
  python download_ocr_models.py

之后启动 app 时 OCR 直接加载，不再触发下载。
"""
import easyocr

print("[OCR] 预下载模型 en + ar + ch_sim ...")
reader = easyocr.Reader(['en', 'ar', 'ch_sim'], gpu=False)
print("[OCR] 模型准备完成 ✓")
