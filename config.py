"""
全局配置 — 双模型方案：CLIP粗召回 + ResNet50精排
适用：4核服务器，1万图片，零外部依赖
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 数据存储 ==========
DATA_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_PATH = os.path.join(DATA_DIR, 'images.db')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'faiss_clip_index.bin')
ID_MAP_PATH = os.path.join(DATA_DIR, 'id_map.json')
RESNET_FEATURES_PATH = os.path.join(DATA_DIR, 'resnet_features.npy')

# ========== 图片存储 ==========
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'}

# 上传字节上限 — 与 app.py 的 MAX_CONTENT_LENGTH 保持一致
# （此前本值是 5MB，而 MAX_CONTENT_LENGTH 和 API 文档都写 16MB，三方矛盾：
#   手机原图普遍 3-8MB，大量图片被直接拒收，用户感知为"传不上去"）
MAX_IMAGE_SIZE = 16 * 1024 * 1024

# 像素数上限 — 真正的解压炸弹防线
# 字节数挡不住高压缩比图片：2MB 的 20000x20000 PNG 解码后 RGB 需要 1.2GB。
# 在 Image.open() 之后、解码像素之前用文件头里的尺寸判定（读取尺寸不耗内存）
MAX_IMAGE_PIXELS = 50_000_000

# ========== 工作分辨率（性能与精度的核心开关） ==========
# 所有图片统一"只解码一次"并封顶到该长边，之后清晰度检测与双模型推理都复用
# 这一张工作图。等比缩放的复合仍是一次等比缩放，所以该中间步不会改变模型看到
# 的构图（中心裁切比例恒为 224/min(w,h)），只影响重采样质量。
#
# 为什么选 1024：库中索引向量基于 OBS 缩略图（长边 800，见 OBS_IMAGE_PROCESS）
# 计算，1024 与它同尺度域；实测库图原图长边分布为 800/1080/2048/4096 等，
# 封顶 1024 对现有索引 99.98% 是 no-op。
# ⚠️ MIN_SHARPNESS 的语义与本值绑定，改动本值必须重新标定清晰度阈值。
WORK_IMAGE_MAX_SIZE = 1024

# 前端 canvas 压缩参数（app.py 内联页面按此实现，保持前后端同源）
CLIENT_MAX_EDGE = 1024
CLIENT_JPEG_QUALITY = 0.9

# 图片清晰度最低要求（拉普拉斯方差）
# ⚠️ 语义：在【工作图】上计算（长边 = WORK_IMAGE_MAX_SIZE），不是原图！
# 拉普拉斯方差与分辨率强相关，旧注释里"白底产品图 100-500"是全分辨率的经验值，
# 在工作图上完全不适用（同一张真实产品图：2048² 上 45.3，1024 上 144.2）。
# 实测（1024 工作图）：清晰组 29~286，明确模糊组 1.3~12.8 —— 20 落在间隙内，
# 但对平坦白底产品图（29.2）只剩 1.46x 余量。误拒是用户可见的硬失败（直接 400），
# 误放只是召回质量下降，按危害不对称取低值。
MIN_SHARPNESS = 12.0

# 清晰度强制开关
# True  = 低于 MIN_SHARPNESS 直接拒绝搜索
# False = 影子模式：照常计算并把分数透出到响应/日志，但不拒绝
# 新阈值尚未经过真实用户 query 标定，先跑影子模式收集分布，标定后再改 True。
SHARPNESS_ENFORCE = False

# ========== 双模型配置 ==========
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# CLIP ViT-B/32 — 粗召回（语义级别）
CLIP_MODEL_PATH = os.path.join(MODELS_DIR, 'clip_vit_b32_visual.onnx')
CLIP_FEATURE_DIM = 512
CLIP_IMAGE_SIZE = 224

# ResNet50 — 细粒度精排（纹理/细节级别）
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'resnet50_feature.onnx')
RESNET_FEATURE_DIM = 2048
RESNET_IMAGE_SIZE = 224

# ONNX 推理线程数（4核建议 CLIP用2, ResNet用1，留1核给系统）
CLIP_NUM_THREADS = 2
RESNET_NUM_THREADS = 1

# ========== 两阶段检索配置 ==========
# 粗召回候选数（CLIP从FAISS取回的候选）
COARSE_TOP_K = 200

# 精排后返回数
FINAL_TOP_K = 20

# 检索策略：CLIP只做粗召回，ResNet做最终排序
# CLIP 语义太宽泛（所有白底产品图都像），ResNet 纹理区分度更好
# 注意：这个判断成立的前提是【查询图与库图同为白底商品图】。手机实拍图
# 混入背景后该前提不成立，此时真正的瓶颈是下面的多尺度查询，而不是排序方式。
USE_CLIP_FOR_RANKING = False   # False = CLIP仅粗召回，ResNet单独排序

# ========== 多尺度查询（手机实拍检索的关键） ==========
# resize_keep_ratio 是"短边缩到 224 + 中心裁切"，它隐含假设【商品占满画面】
# —— 库中的 OBS 白底商品图正是如此。但手机实拍时商品在画面中的占比由拍摄
# 距离决定：实测占比 >=70% 时检索很鲁棒（旋转 25 度 / 暗光 0.7 / 模糊均命中
# TOP1），而占比 <=60% 时完全失效 —— 不是分数低，是根本排不进候选，
# 把 MIN_TOP_SCORE 从 0.70 一路降到 0.40 命中数依然是 0。
#
# 解法：对查询图按多档 zoom 做中心裁切，各档独立检索，取 top1 分数最高的档。
# zoom 语义 = 裁切后边长是原短边的 1/zoom，即 zoom 越大裁得越紧：
#   1.0 → 商品占满(等价改造前的行为)   1.4 → 商品约占 70%
#   2.0 → 商品约占 50%                  2.8 → 商品约占 35%
# 实测 9/9 场景（含 35%+布料背景、木桌偏右、暗光、旋转 20 度）全部第 1 名命中。
#
# ⚠️ 每档约 140ms 推理，档位越多越慢（4 档约 560ms）。
#    白底图搜白底图时 zoom=1.0 分数最高会被自动选中，行为与改造前一致，不退化。
SEARCH_ZOOMS = (1.0, 1.4, 2.0, 2.8)

# 早停：某档 top1 分数达到此值就不再尝试更紧的档位。
# 白底图自匹配通常第一档就是 1.0，可省下其余档位的耗时；
# 设为 0 或 None 表示关闭早停、始终跑满所有档位（最稳，但最慢）。
ZOOM_EARLY_STOP_SCORE = 0.95

# 分数融合权重（仅当 USE_CLIP_FOR_RANKING=True 时生效）
FUSION_ALPHA = 0.35

# 分数拉伸：把原始余弦相似度映射到更宽的范围
# stretched = (score - SCORE_STRETCH_MIN) / (1.0 - SCORE_STRETCH_MIN)
# 裁剪到 [0, 1]
SCORE_STRETCH_MIN = 0.45       # 低于此值的原始分 → 映射后接近0

# ResNet 精排阈值（原始余弦相似度，未拉伸）
MIN_RESNET_SCORE = 0.50        # ResNet原始分低于此值的直接过滤
MIN_RELATIVE_SCORE = 0.65       # 低于最高分65%的过滤

# 最低融合分数阈值 — 单个结果低于此值直接过滤
MIN_FUSED_SCORE = 0.30          # 拉伸分 <0.30（即显示<30%）→ 不返回

# 最高分门槛 — 第一名融合分低于此值 → 全库没有匹配
#
# 0.70 折合原始 ResNet 余弦 0.835，这个值是按【白底图搜白底图】定的：
# 库内商品图互相检索的原始分是 0.998~1.000，余量充足。
# 但手机实拍照片与白底库图存在巨大域差异（实测：办公室地毯/玻璃反光/随手拍
# 角度 vs 影棚白底），同一批真实照片的原始分只有 0.78~0.82，被 0.835 全数挡死
# ——表现为"手机拍的照片直接返回空"。而对照人工核对，这些低分结果的 top5
# 其实匹配得相当准（无人机照片 top5 全是无人机、医疗设备照片 top5 全是医疗设备）。
#
# 0.60 折合原始余弦 0.78，恰好放行这批真实照片。
# ⚠️ 代价：纯噪声图的最高档能到 0.6966（原始 0.833），降阈值后也会漏出结果。
#    无法用单一阈值区分二者，前端应如实展示相似度百分比供用户判断。
MIN_TOP_SCORE = 0.60

# 品类自动分类
CATEGORY_CLASSIFY_CONFIDENCE = 0.02

# 自动识别出的品类是否用于过滤候选集（默认关闭）
# 注意：ALLOWED_CATEGORIES 是 7 个中文品类，而库里实际是 90+ 个英文行业大类，
# 两者体系不一致；CLIP 对这类粗粒度品类误判率高，一旦用作过滤器会把本可命中的
# 结果整片排除。关闭后仍会返回 predicted_category，仅不参与候选集裁剪。
AUTO_CATEGORY_FILTER = False

# 产品名称锚定过滤（核心防线）
# 以第一名产品名为锚，其他结果须共享关键词
NAME_ANCHOR_MIN_MATCH = 1       # 基础：至少匹配1个关键词
NAME_ANCHOR_MIN_RESULTS = 2     # 至少2个结果才触发过滤

# 共识过滤的启用门控：第一名显示分低于此值时不启用
#
# 实测：手机实拍照片（跨域检索）的第一名分数只有 0.65~0.68，此时 top1 本身就可能
# 是错的（照片与白底库图域差异大）。拿一个可能错误的结果当"锚"，会把它与正确结果
# 不一致的词当成共识，反而砍掉正确答案 —— 实测把 20 条砍到 1~4 条，而出现的共识词
# 是 'office' / 'bedside' / 'tethered' 这类明显跑偏的词。
#
# 白底图搜白底图时第一名接近 1.0，锚是可靠的，共识过滤仍应正常生效。
NAME_ANCHOR_MIN_TOP_SCORE = 0.80

# ========== 缓存 ==========
CACHE_SIZE = 200
CACHE_TTL = 300

# ========== 批量索引 ==========
BATCH_SIZE = 16  # 双模型推理更耗内存，减小批量
INDEX_PROGRESS_FILE = os.path.join(DATA_DIR, 'index_progress.json')

# ========== 品类 ==========
ALLOWED_CATEGORIES = ['包包', '鞋子', '衣服', '裤子', '裙子', '配饰', '其他']

# ========== 外部同步（Arab-Bee 产品数据源） ==========
# JWT Token 从环境变量读取，不硬编码在代码中
# 设置方式: export ARAB_BEE_TOKEN="your_jwt_token"
ARAB_BEE_API_URL = "https://biz.arab-bee.com/admin/product/list"
ARAB_BEE_TOKEN = os.environ.get("ARAB_BEE_TOKEN", "")
SYNC_PAGE_SIZE = 30
SYNC_MAX_PAGES = 4

# ========== 华为云 OBS 图片处理 ==========
# 拼接在图片 URL 后面，用于缩略图展示（不影响原图特征提取）
# m_lfit 是 limit-fit：**只缩不放**，原图长边小于目标值时会原样返回
# 注意参数是下划线语法 w_800 / q_85（华为云 OBS 规格），不是 w=800
OBS_IMAGE_PROCESS_TMPL = 'x-image-process=image/resize,m_lfit,w_{w}/quality,q_{q}/auto-orient,1'


def obs_image_url(url: str, w: int = 800, q: int = 85) -> str:
    """为 OBS 图片 URL 拼接缩放参数（按需选用 ? 或 & 分隔符）"""
    sep = '&' if '?' in url else '?'
    return f"{url}{sep}{OBS_IMAGE_PROCESS_TMPL.format(w=w, q=q)}"


# 兼容既有引用（sync_products.py 直接做字符串拼接）
OBS_IMAGE_PROCESS = '?' + OBS_IMAGE_PROCESS_TMPL.format(w=800, q=85)

# ========== Flask ==========
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# ========== 并发保护 ==========
MAX_CONCURRENT_REQUESTS = 10      # 最大并发请求数，超出返回 503
REQUEST_TIMEOUT = 30              # 单个请求最大处理时间（秒）
