"""
图片 OCR 文本提取 — 从产品图中识别品牌/型号/名称
用作搜图的前置步骤，文字匹配优先于视觉匹配
"""
import re
import threading
import numpy as np
from PIL import Image
import io


_reader = None
_reader_lock = threading.Lock()


_easyocr_available = None


def _get_reader():
    """懒加载 easyocr Reader（单例，线程安全）"""
    global _reader, _easyocr_available
    if _easyocr_available is False:
        return None
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is not None:
            return _reader
        if _easyocr_available is False:
            return None
        try:
            import easyocr
            print("[OCR] 加载 easyocr 模型（首次约30秒）...")
            _reader = easyocr.Reader(['en', 'ar', 'ch_sim'], gpu=False)
            _easyocr_available = True
            print("[OCR] 模型就绪")
            return _reader
        except ImportError:
            print("[OCR] easyocr 未安装，OCR 功能不可用")
            _easyocr_available = False
            return None
        except SystemExit:
            # gunicorn worker 超时信号 → 模型下载太慢
            print("[OCR] 模型下载超时（gunicorn timeout），OCR 已禁用")
            _easyocr_available = False
            return None
        except Exception as e:
            print(f"[OCR] 加载失败: {e}")
            _easyocr_available = False
            return None


def warmup_ocr():
    """启动时后台预热 OCR 模型（不阻塞服务）"""
    import threading
    def _load():
        print("[OCR] 后台预热中...")
        _get_reader()
    t = threading.Thread(target=_load, name='ocr-warmup', daemon=True)
    t.start()


def extract_text(image_bytes: bytes) -> list[tuple[str, float]]:
    """
    从图片中提取文本，返回 [(text, confidence), ...]
    按置信度降序排列
    """
    try:
        reader = _get_reader()
        if reader is None:
            return []  # OCR 未就绪，静默降级
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        results = reader.readtext(arr, detail=1)  # detail=1 返回 (bbox, text, confidence)
        # 只保留置信度 ≥ 0.5 的结果
        return [(t.strip(), float(c)) for _, t, c in results
                if t.strip() and float(c) >= 0.5]
    except Exception as e:
        print(f"[OCR] 提取失败: {e}")
        return []


def match_product_names(texts: list[str], db_names: list[str]) -> dict[str, list[str]]:
    """
    将 OCR 文本与数据库中的产品名做关键词匹配。
    返回 {db_product_name: [matched_keywords], ...}
    """
    if not texts or not db_names:
        return {}

    # 从 OCR 文本中提取有意义的关键词
    keywords = set()
    for t in texts:
        # 分词
        tokens = re.split(r'[\s\-/,.;:()\[\]{}]+', t.lower())
        for tok in tokens:
            # 保留有意义的词：>2字符，非纯数字
            tok = tok.strip()
            if len(tok) > 2 and not tok.isdigit():
                keywords.add(tok)

    if not keywords:
        return {}

    # 匹配
    matches = {}
    for name in db_names:
        if not name:
            continue
        name_lower = name.lower()
        matched = [kw for kw in keywords if kw in name_lower]
        if matched:
            matches[name] = matched

    return matches
