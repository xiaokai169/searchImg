"""
文本处理工具 — 中英文关键字提取
"""
import re


# ========== 中文停用字 ==========
CN_STOP_CHARS: set[str] = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没',
    '看', '好', '自', '这', '他', '她', '它', '那', '们', '与', '及', '或',
    '为', '以', '而', '从', '被', '把', '让', '向', '对', '于', '由', '其',
    '可', '能', '所', '但', '只', '各', '如', '此', '之', '后', '前', '中',
    '大', '小', '多', '少', '新', '旧', '高', '低', '长', '短', '快', '慢',
    '款', '型', '号', '色', '码', '尺', '寸', '品', '牌', '货', '价', '格',
    '包', '邮', '现', '批', '零', '售', '正', '保', '证', '原', '单', '装',
    '韩', '版', '欧', '美', '日', '风', '潮', '流', '百', '搭', '休', '闲',
    '春', '夏', '秋', '冬', '季', '男', '女', '通', '儿', '童', '老', '少',
    '轻', '奢', '简', '约', '复', '古', '刺', '绣', '印', '花', '纯', '棉',
}


# ========== 英文停用词 ==========
EN_STOP_WORDS: set[str] = {
    'inch', 'with', 'and', 'for', 'the', 'new', 'hot', 'best',
    'high', 'quality', 'premium', 'sale', 'free', 'size', 'color',
    'large', 'small', 'medium', 'style', 'model', 'brand', 'made',
    'china', 'product', 'goods', 'item', 'type', 'set', 'pack',
    'piece', 'unit', 'each', 'per', 'cm', 'mm', 'meter', 'gram',
    'kg', 'dual', 'sim', 'ram', 'rom', 'version', 'global', 'middle',
    'east', 'black', 'white', 'blue', 'grey', 'gold', 'silver',
    'red', 'green', 'pink', 'brown', 'yellow', 'purple', 'orange',
    'portable', 'smart', 'magnetic', 'liquid', 'silicone', 'matte',
    'flash', 'magic', 'tempered', 'glass', 'screen', 'protector',
}

# ========== 英文品牌名（跨品类通用，不能用作搜索词） ==========
EN_BRANDS: set[str] = {
    'xiaomi', 'redmi', 'samsung', 'apple', 'macbook', 'galaxy',
    'black', 'shark', 'huawei', 'honor', 'oneplus', 'oppo', 'vivo',
    'realme', 'nokia', 'motorola', 'google', 'pixel', 'lenovo',
    'dell', 'asus', 'acer', 'sony', 'lg', 'philips', 'panasonic',
    'bosch', 'siemens',
}


def extract_keywords_cn(text: str, min_len: int = 2) -> str:
    """
    从中文文本提取搜索关键字。
    使用字符级 bigram / trigram，过滤停用字和标点。
    返回空格分隔的关键字串，可直接存入 DB。
    """
    if not text:
        return ''

    # 只保留中文字符
    cn_chars = re.findall(r'[一-鿿]', text)
    if not cn_chars:
        return ''

    # 过滤停用字
    chars = [c for c in cn_chars if c not in CN_STOP_CHARS]
    if not chars:
        return ''

    keywords: set[str] = set()

    # bigram（2-gram）
    for i in range(len(chars) - 1):
        kw = chars[i] + chars[i + 1]
        keywords.add(kw)

    # trigram（3-gram）— 更精确的关键词
    for i in range(len(chars) - 2):
        kw = chars[i] + chars[i + 1] + chars[i + 2]
        keywords.add(kw)

    # 也保留单个字（作为基础搜索单位）
    for c in chars:
        keywords.add(c)

    return ' '.join(sorted(keywords))


def extract_keywords_en(text: str, min_len: int = 3) -> str:
    """
    从英文文本提取搜索关键字。
    分词 → 去停用词/品牌名 → 保留长度 > min_len 的 token。
    返回空格分隔的关键字串。
    """
    if not text:
        return ''

    tokens = re.split(r'[\s\-/,.;:()\[\]{}|]+', text.lower())
    keywords = {
        t for t in tokens
        if len(t) > min_len
        and t not in EN_STOP_WORDS
        and t not in EN_BRANDS
        and not t.isdigit()
        and not t.replace('.', '').isdigit()
    }

    return ' '.join(sorted(keywords))


def build_keywords(product_name_cn: str, product_name_en: str) -> tuple[str, str]:
    """
    便捷函数：同时提取中英文关键字。
    返回 (keywords_cn, keywords_en)
    """
    return extract_keywords_cn(product_name_cn), extract_keywords_en(product_name_en)
