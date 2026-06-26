"""
图像预处理流水线
标准化 → 尺寸缩放 → 中心裁切 → 归一化 → 转为模型输入
"""
import io
import warnings
import numpy as np
from PIL import Image, ImageOps
from config import MAX_IMAGE_SIZE

# 忽略调色板透明通道警告（不影响推理结果）
warnings.filterwarnings('ignore', message='Palette images with Transparency')


# ImageNet 标准化参数（与训练时一致）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    校验图片合法性
    返回: (是否合法, 错误信息)
    """
    if len(image_bytes) == 0:
        return False, "图片数据为空"

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return False, f"图片过大 ({len(image_bytes)} > {MAX_IMAGE_SIZE} bytes)"

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # 验证图片完整性
    except Exception as e:
        return False, f"图片格式无效: {e}"

    return True, ""


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    解码图片为 RGB PIL Image
    """
    img = Image.open(io.BytesIO(image_bytes))
    # 处理透明通道
    if img.mode in ('RGBA', 'LA', 'P'):
        # 有透明通道 → 白色背景
        if img.mode == 'P':
            img = img.convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert('RGB')


def resize_keep_ratio(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    等比缩放：短边缩放到 target_size，然后中心裁切正方形
    这是最常见的预处理方式（与 torchvision CenterCrop 一致）
    """
    # 计算缩放比例：使短边 = target_size
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 中心裁切 target_size × target_size
    left = (new_w - target_size) // 2
    top  = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))

    return img


def preprocess_for_model(image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    完整的预处理流水线：bytes → (1, 3, H, W) numpy 数组
    可直接喂入 ONNX 模型
    """
    # 1. 解码
    img = decode_image(image_bytes)

    # 2. 等比缩放 + 中心裁切
    img = resize_keep_ratio(img, target_size)

    # 3. 转换为 numpy 并归一化
    arr = np.array(img, dtype=np.float32) / 255.0

    # 4. ImageNet 标准化
    arr = (arr - _MEAN) / _STD

    # 5. HWC → CHW，添加 batch 维度
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr


def preprocess_for_display(image_bytes: bytes, max_size: int = 512) -> bytes:
    """
    预处理用于前端展示：等比缩放到合理尺寸，转为 JPEG 字节
    用于生成缩略图，减少前端加载压力
    """
    img = decode_image(image_bytes)
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85, optimize=True)
    return buf.getvalue()


def check_sharpness(image_bytes: bytes) -> tuple[float, bool, str]:
    """
    检测图片清晰度（拉普拉斯方差法 + 中心加权）
    产品图通常在白色背景上，整体方差低但中心区域应有足够细节。
    返回: (sharpness_score, is_clear, message)
    """
    from config import MIN_SHARPNESS
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape

        # 全图拉普拉斯方差
        lap_full = _laplacian_variance(arr)

        # 中心50%区域拉普拉斯方差（产品通常在中间）
        ch, cw = h // 4, w // 4
        center = arr[ch:h-ch, cw:w-cw]
        lap_center = _laplacian_variance(center) if center.size > 0 else 0.0

        # 取全图和中心区域的较高值（中心区域通常更有信息量）
        variance = max(lap_full, lap_center * 0.7)  # 中心权重稍降，防极端

        if variance >= MIN_SHARPNESS:
            return variance, True, ""
        else:
            return variance, False, (
                f"图片清晰度不足（得分 {variance:.1f}，要求 ≥{MIN_SHARPNESS}）。"
                f"请上传更清晰的图片。"
            )
    except Exception as e:
        return 0.0, False, f"清晰度检测失败: {e}"


def _laplacian_variance(arr: np.ndarray) -> float:
    """计算数组的拉普拉斯方差（边缘强度指标）"""
    laplacian = np.zeros_like(arr)
    laplacian[1:-1, 1:-1] = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, :-2] + arr[1:-1, 2:] -
        4 * arr[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def get_image_info(image_bytes: bytes) -> dict:
    """获取图片基本信息（不加载像素）"""
    img = Image.open(io.BytesIO(image_bytes))
    return {
        'format': img.format,
        'mode': img.mode,
        'width': img.width,
        'height': img.height,
        'size_bytes': len(image_bytes),
    }
