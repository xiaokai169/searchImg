"""
图像预处理流水线
解码（仅一次）→ EXIF 方向校正 → 尺寸封顶 → 模型输入 / 清晰度检测

核心约定
    所有业务路径都必须经由 decode_image / load_work_image 得到一张「工作图」，
    清晰度检测与 CLIP/ResNet 两个模型共用同一张图，全流程只解码一次。

    改造前：搜索一次对同一张 4000x3000 照片解码 3 遍（清晰度 1 遍 + 双模型各 1 遍），
    其中清晰度那遍还要转 float64（单个数组 96MB，实测瞬时峰值 386MB）。
"""
import io
import warnings
import numpy as np
from PIL import Image, ImageOps
from config import MAX_IMAGE_SIZE, MAX_IMAGE_PIXELS, WORK_IMAGE_MAX_SIZE

# 忽略调色板透明通道警告（不影响推理结果）
warnings.filterwarnings('ignore', message='Palette images with Transparency')


# ImageNet 标准化参数（与训练时一致）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    校验图片合法性（只解析文件头，不解码像素，开销极小）
    返回: (是否合法, 错误信息)
    """
    if len(image_bytes) == 0:
        return False, "图片数据为空"

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return False, (f"图片过大 ({len(image_bytes) / 1024 / 1024:.1f}MB > "
                       f"{MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB)")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # 读文件头里的尺寸不触发解码，是拦截解压炸弹最便宜的位置
        if img.width * img.height > MAX_IMAGE_PIXELS:
            return False, (f"图片像素过多 ({img.width}x{img.height} > "
                           f"{MAX_IMAGE_PIXELS})")
        img.verify()  # 验证图片完整性
    except Exception as e:
        return False, f"图片格式无效: {e}"

    return True, ""


def decode_image(image_bytes: bytes, max_size: int | None = None) -> Image.Image:
    """
    唯一解码入口：bytes → RGB PIL Image（已完成 EXIF 方向校正）

    max_size: 长边上限；None 表示不封顶。设了也只缩不放。

    ⚠️ 禁止对本函数的返回值做缓存。工作图 1024x768 RGB 约 2.25MB，若按现有
       LRUCache(CACHE_SIZE=200) 的规模缓存会直接占掉 450MB —— 那就是新的 OOM 源。
       复用范围仅限「单次请求 / 单个批次」内，用完立刻释放。
    """
    # 仅解析文件头，此时尚未解码像素
    img = Image.open(io.BytesIO(image_bytes))

    # 1) 解压炸弹保护 —— 字节数挡不住高压缩比图片
    #    2MB 的 20000x20000 PNG 解码成 RGB 要 1.2GB，但只有 5MB 上限是拦不住的
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"图片像素过多 ({img.width}x{img.height} > {MAX_IMAGE_PIXELS})"
        )

    # 2) JPEG 的 DCT 域降采样 —— 必须在任何像素访问之前调用，晚了就无效
    #    4000x3000 目标 1024 → 直接解成 2000x1500，RGB 缓冲从 36MB 降到 9MB
    #    对 PNG/BMP/GIF 是 no-op；它只降尺寸，不改 mode
    if max_size:
        img.draft(None, (max_size, max_size))

    # 3) EXIF 方向校正 —— 手机竖拍照片靠这一步摆正
    #    缺失时竖图在服务端是横躺的，resize_keep_ratio 的中心裁切会裁到完全
    #    错位的区域 → 搜不准。索引侧图片由 OBS 的 auto-orient,1 处理过，
    #    已经是正向图，所以这里对索引链路是 no-op，不需要重建索引。
    img = ImageOps.exif_transpose(img)

    # 4) 透明通道 → 白底
    #    注意 LA（灰度+透明）也必须先转 RGBA：Image.alpha_composite 要求两边
    #    mode 相同，否则抛 ValueError: images do not match。
    #    旧实现只把 P 转成 RGBA，导致 LA 模式的 PNG 上传后直接 500。
    if img.mode in ('RGBA', 'LA', 'P'):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert('RGB')

    # 5) 尺寸封顶
    #    thumbnail 内置 reducing_gap，会先做整数倍 box 降采样再插值，
    #    比直接 resize 更快且质量更好；它只缩不放
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return img


def load_work_image(image_bytes: bytes) -> Image.Image:
    """解码为「工作图」—— 业务路径的统一入口（长边封顶 WORK_IMAGE_MAX_SIZE）"""
    return decode_image(image_bytes, max_size=WORK_IMAGE_MAX_SIZE)


def resize_keep_ratio(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    等比缩放：短边缩放到 target_size，然后中心裁切正方形
    这是最常见的预处理方式（与 torchvision CenterCrop 一致）

    几何不变性：缩放比例 = target_size / min(w, h)，中心裁切框的比例只取决于
    这个比值。等比缩放的复合仍是一次等比缩放，所以前面无论把长边封顶到 1024
    还是保留 4000，模型看到的画面区域完全相同，差异仅在重采样质量。
    """
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 中心裁切 target_size × target_size
    left = (new_w - target_size) // 2
    top  = (new_h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


# 多尺度裁切时，小于此边长的档位直接跳过（再放大回 224 已无信息量）
MIN_ZOOM_SIDE = 128


def center_zoom_crops(img: Image.Image, zooms) -> tuple[list[Image.Image], list[float]]:
    """
    按多档 zoom 中心裁切出正方形区域，供多尺度检索使用。

    zoom 语义：裁切边长 = min(w, h) / zoom，即 zoom 越大裁得越紧。
    zoom=1.0 的裁切区域与 resize_keep_ratio 完全一致（都是中心 min(w,h) 正方形），
    所以"商品占满画面"的场景走的是与原实现相同的路径，不会退化。

    返回 (裁切列表, 实际生效的 zoom 列表) —— 过小的档位会被跳过，两者一一对应。
    裁切是惰性的（PIL 只记录区域，采样发生在后续 resize），调用本身几乎无开销。
    """
    w, h = img.size
    side_base = min(w, h)
    crops: list[Image.Image] = []
    used: list[float] = []
    for z in zooms:
        side = int(side_base / z)
        if side < MIN_ZOOM_SIDE:
            continue
        left = (w - side) // 2
        top = (h - side) // 2
        crops.append(img.crop((left, top, left + side, top + side)))
        used.append(float(z))
    return crops, used


def to_model_tensor(img: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    PIL 图 → (1, 3, H, W) float32 张量，可直接喂入 ONNX 模型
    """
    img = resize_keep_ratio(img, target_size)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD                      # ImageNet 标准化
    arr = arr.transpose(2, 0, 1)                    # HWC → CHW
    return np.expand_dims(arr, axis=0)              # 添加 batch 维


def to_model_tensor_batch(images: list[Image.Image], target_size: int = 224) -> np.ndarray:
    """多张 PIL 图 → (N, 3, H, W) float32 张量"""
    batch = [to_model_tensor(img, target_size)[0] for img in images]
    return np.stack(batch, axis=0)


# ==================== 向后兼容的 bytes 薄包装 ====================
# 保留原签名，让 indexer.py / sync_products.py 无需改动即自动获得
# "每张图只解码一次"的收益

def preprocess_for_model(image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    完整的预处理流水线：bytes → (1, 3, H, W) numpy 数组
    （薄包装：解码一次得到工作图，再转张量）
    """
    return to_model_tensor(load_work_image(image_bytes), target_size)


def check_sharpness(image_bytes: bytes) -> tuple[float, bool, str]:
    """
    清晰度检测（bytes 版薄包装）
    返回: (sharpness_score, is_clear, message)
    """
    return check_sharpness_image(load_work_image(image_bytes))


# ==================== 清晰度检测 ====================

def check_sharpness_image(img: Image.Image) -> tuple[float, bool, str]:
    """
    检测图片清晰度（拉普拉斯方差法 + 中心加权）

    入参必须是【工作图】（长边 = WORK_IMAGE_MAX_SIZE）。拉普拉斯方差与分辨率
    强相关，同一张图上不同分辨率的得分不可比 —— 阈值只在固定工作分辨率下才有
    唯一含义，这也正是必须先把图统一解码再检测的原因。

    返回: (sharpness_score, is_clear, message)
    """
    from config import MIN_SHARPNESS, SHARPNESS_ENFORCE
    try:
        # 用 int16 而非 float64：
        # 灰度值域 0-255，4 邻域拉普拉斯最大 ±1020，int16 足够且不溢出。
        # 实测结果与 float64 逐位相同，但 1024 图上耗时 8.8ms vs 28.7ms。
        # 更关键的是内存 —— float64 版在 12MP 图上单次调用瞬时分配 386MB，
        # 而本函数在推理锁之外被调用，10 并发时最坏能到 3.9GB。
        arr = np.asarray(img.convert('L'), dtype=np.int16)
        h, w = arr.shape

        # 全图拉普拉斯方差
        lap_full = _laplacian_variance(arr)

        # 中心50%区域拉普拉斯方差（产品通常在中间）
        ch, cw = h // 4, w // 4
        center = arr[ch:h - ch, cw:w - cw]
        lap_center = _laplacian_variance(np.ascontiguousarray(center)) if center.size else 0.0

        # 取全图和中心区域的较高值（中心区域通常更有信息量）
        variance = max(lap_full, lap_center * 0.7)   # 中心权重稍降，防极端

        if variance >= MIN_SHARPNESS or not SHARPNESS_ENFORCE:
            # 影子模式（SHARPNESS_ENFORCE=False）下照常算出分数并放行，
            # 分数由调用方透出到响应/日志，用于收集分布后再标定阈值
            return variance, True, ""

        return variance, False, (
            f"图片清晰度不足（得分 {variance:.1f}，要求 ≥{MIN_SHARPNESS:g}）。"
            f"请上传更清晰的图片。"
        )
    except Exception as e:
        # 检测本身异常时不应升级成"拒绝用户"，放行并记录
        return 0.0, True, f"清晰度检测失败: {e}"


def _laplacian_variance(arr: np.ndarray) -> float:
    """计算数组的拉普拉斯方差（边缘强度指标）。arr 须为 int16，值域 ±1020 不溢出。"""
    laplacian = np.zeros_like(arr)
    laplacian[1:-1, 1:-1] = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, :-2] + arr[1:-1, 2:] -
        4 * arr[1:-1, 1:-1]
    )
    return float(laplacian.var())


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


def preprocess_for_display(image_bytes: bytes, max_size: int = 512) -> bytes:
    """
    预处理用于前端展示：等比缩放到合理尺寸，转为 JPEG 字节
    用于生成缩略图，减少前端加载压力（当前项目内无调用方）
    """
    img = decode_image(image_bytes, max_size=max_size)

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85, optimize=True)
    return buf.getvalue()
