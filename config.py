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
MAX_IMAGE_SIZE = 5 * 1024 * 1024

# 图片清晰度最低要求（拉普拉斯方差，越高越清晰）
# 白底产品图通常 100-500，模糊图 < 50
MIN_SHARPNESS = 20              # 低于此值拒绝搜索（白底产品图边缘少，分数偏低属正常）

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
USE_CLIP_FOR_RANKING = False   # False = CLIP仅粗召回，ResNet单独排序

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
MIN_TOP_SCORE = 0.70            # 第一名 <70% → "库中无匹配商品"

# 品类自动分类
CATEGORY_CLASSIFY_CONFIDENCE = 0.02

# 产品名称锚定过滤（核心防线）
# 以第一名产品名为锚，其他结果须共享关键词
NAME_ANCHOR_MIN_MATCH = 1       # 基础：至少匹配1个关键词
NAME_ANCHOR_MIN_RESULTS = 2     # 至少2个结果才触发过滤

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
ARAB_BEE_TOKEN = os.environ.get("ARAB_BEE_TOKEN", "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpYXQiOjE3ODIzNTM3ODYsImV4cCI6MTc4MjUyNjU4Niwicm9sZXMiOlsiUk9MRV9BRE1JTiJdLCJ1c2VybmFtZSI6ImFyYWIifQ.FKSFd-r6gPS6St6hpvJ1ONEuJ-DBltg65_uHQARQTH0vE2y0LhqUHjBN81KiDGsYfjtdWqQj5hlng8LqwpqxARZuKBF-1x9QSO05CHFzvPgCQWDvcXV1-KZ4rT6MNc79kUZ1kJ9k4Lm2sGhhIrIUQrrOpLRg1xThSLF1EBaBMst5RIWhMmIClwSO9OngAqmUiJCkyJktonsvJq-jVrdphQwpk5I5_NX7mGD12UIrT1-vIDNeDyTi7GlgXx__5XoCVn_K2zyC75_h_IOXZ8eKn0WLnG_uj26NtzoCPLXd9VGZv5QR51E7euQcTtcJDi_7X2Qq7DKL_zW9DYJyGlieWw")
SYNC_PAGE_SIZE = 30
SYNC_MAX_PAGES = 4

# ========== 华为云 OBS 图片处理 ==========
# 拼接在图片 URL 后面，用于缩略图展示（不影响原图特征提取）
OBS_IMAGE_PROCESS = '?x-image-process=image/resize,m_lfit,w_800;quality,q_85;auto-orient,1'

# ========== Flask ==========
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
