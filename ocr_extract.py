"""
图片 OCR 文本提取 — 从产品图中识别品牌/型号/名称
用作搜图的前置步骤，文字匹配优先于视觉匹配
"""
import re
import threading
import numpy as np
from PIL import Image
import io


# ch_sim 只能和 en 搭配，ar 需要单独的 Reader（gen1）
_reader_cn = None    # en + ch_sim
_reader_ar = None    # ar
_reader_lock = threading.Lock()
_easyocr_available = None


def _get_easyocr():
    """验证 easyocr 是否可用（不加载模型）"""
    global _easyocr_available
    if _easyocr_available is not None:
        return _easyocr_available
    try:
        import easyocr  # noqa: F401
        _easyocr_available = True
    except ImportError:
        _easyocr_available = False
    return _easyocr_available


def _get_reader():
    """懒加载 easyocr Readers（单例，线程安全）"""
    global _reader_cn, _reader_ar, _easyocr_available
    if not _get_easyocr():
        return None, None
    if _reader_cn is not None and _reader_ar is not None:
        return _reader_cn, _reader_ar
    with _reader_lock:
        if _reader_cn is not None and _reader_ar is not None:
            return _reader_cn, _reader_ar
        try:
            import easyocr
            if _reader_cn is None:
                print("[OCR] 加载中英文模型 (en+ch_sim) ...")
                _reader_cn = easyocr.Reader(['en', 'ch_sim'], gpu=False)
                print("[OCR] 中英文模型就绪")
            if _reader_ar is None:
                print("[OCR] 加载阿拉伯语模型 (ar) ...")
                _reader_ar = easyocr.Reader(['ar'], gpu=False)
                print("[OCR] 阿拉伯语模型就绪")
            return _reader_cn, _reader_ar
        except SystemExit:
            print("[OCR] 模型下载超时（gunicorn timeout），OCR 已禁用")
            _easyocr_available = False
            return None, None
        except Exception as e:
            print(f"[OCR] 加载失败: {e}")
            _easyocr_available = False
            return None, None


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
        reader_cn, reader_ar = _get_reader()
        if reader_cn is None and reader_ar is None:
            return []  # OCR 未就绪，静默降级
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        results = []
        if reader_cn is not None:
            results.extend(reader_cn.readtext(arr, detail=1))
        if reader_ar is not None:
            results.extend(reader_ar.readtext(arr, detail=1))
        # 按文本去重，保留置信度更高的
        seen = {}
        for _, t, c in results:
            t_stripped = t.strip()
            if t_stripped and float(c) >= 0.5:
                if t_stripped not in seen or float(c) > seen[t_stripped]:
                    seen[t_stripped] = float(c)
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)
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
