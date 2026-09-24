"""
双模型特征提取器
- CLIP ViT-B/32 (512维): 语义级粗召回
- ResNet50 (2048维): 细粒度精排
线程安全，支持批量推理

接口分两层
    核心层：接收 PIL Image（extract_*_from_image），由调用方持有工作图，
            可与清晰度检测共用同一次解码。
    兼容层：接收 bytes（extract_clip / extract_both / ...），签名与改造前一致，
            内部解码一次得到工作图后转交核心层。indexer.py / sync_products.py
            无需改动即自动获得"每张图只解码一次"的收益。
"""
import threading
import numpy as np
import onnxruntime as ort
from config import (
    CLIP_MODEL_PATH, CLIP_IMAGE_SIZE, CLIP_NUM_THREADS, CLIP_FEATURE_DIM,
    RESNET_MODEL_PATH, RESNET_IMAGE_SIZE, RESNET_NUM_THREADS, RESNET_FEATURE_DIM,
)
from preprocess import (
    preprocess_for_model, load_work_image, to_model_tensor, to_model_tensor_batch,
)


def _create_session(model_path: str, num_threads: int) -> ort.InferenceSession:
    """创建 ONNX 推理会话"""
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=['CPUExecutionProvider']
    )


class DualModelExtractor:
    """双模型特征提取器（线程安全单例）"""

    def __init__(self):
        print(f"[Extractor] 加载 CLIP 模型: {CLIP_MODEL_PATH}")
        self.clip_session = _create_session(CLIP_MODEL_PATH, CLIP_NUM_THREADS)
        self.clip_input = self.clip_session.get_inputs()[0].name
        self.clip_output = self.clip_session.get_outputs()[0].name
        # 验证维度
        dummy = np.random.randn(1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE).astype(np.float32)
        out = self.clip_session.run([self.clip_output], {self.clip_input: dummy})
        self.clip_dim = out[0].shape[1]
        print(f"  CLIP: {self.clip_dim}维")

        print(f"[Extractor] 加载 ResNet50 模型: {RESNET_MODEL_PATH}")
        self.resnet_session = _create_session(RESNET_MODEL_PATH, RESNET_NUM_THREADS)
        self.resnet_input = self.resnet_session.get_inputs()[0].name
        self.resnet_output = self.resnet_session.get_outputs()[0].name
        out = self.resnet_session.run([self.resnet_output], {self.resnet_input: dummy})
        self.resnet_dim = out[0].shape[1]
        print(f"  ResNet50: {self.resnet_dim}维")

        # 单锁保护两个 session。
        # 不拆成 clip_lock / resnet_lock：CLIP_NUM_THREADS=2 + RESNET_NUM_THREADS=1
        # 已经是 3 个推理线程打 4 核，ORT 线程池空闲时会自旋抢 CPU，拆锁后两个
        # 池会互相偷时间、ResNet 延迟反而变差；且 enable_cpu_mem_arena=False 下
        # 两个 session 并发 run 会让瞬时内存峰值叠加（本项目上次事故就是 OOM）。
        self._lock = threading.Lock()

    # ==================== 内部工具 ====================

    @staticmethod
    def _l2(vecs: np.ndarray, axis=None) -> np.ndarray:
        """L2 归一化（axis=None 处理单向量，axis=1 处理批次）"""
        norms = np.linalg.norm(vecs, axis=axis, keepdims=True) + 1e-8
        return vecs / norms

    def _tensors(self, img) -> tuple[np.ndarray, np.ndarray]:
        """取 (clip_tensor, resnet_tensor)；两模型输入尺寸相同时共用同一个张量"""
        if CLIP_IMAGE_SIZE == RESNET_IMAGE_SIZE:
            t = to_model_tensor(img, CLIP_IMAGE_SIZE)
            return t, t
        return (to_model_tensor(img, CLIP_IMAGE_SIZE),
                to_model_tensor(img, RESNET_IMAGE_SIZE))

    def _tensors_batch(self, imgs: list) -> tuple[np.ndarray, np.ndarray]:
        if CLIP_IMAGE_SIZE == RESNET_IMAGE_SIZE:
            t = to_model_tensor_batch(imgs, CLIP_IMAGE_SIZE)
            return t, t
        return (to_model_tensor_batch(imgs, CLIP_IMAGE_SIZE),
                to_model_tensor_batch(imgs, RESNET_IMAGE_SIZE))

    def _run_clip(self, tensor: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.clip_session.run([self.clip_output], {self.clip_input: tensor})

    def _run_resnet(self, tensor: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.resnet_session.run([self.resnet_output], {self.resnet_input: tensor})

    # ==================== 核心层：接收 PIL 工作图 ====================
    # 调用方持有工作图，可与清晰度检测共用同一次解码

    def extract_clip_from_image(self, img) -> np.ndarray:
        """提取 CLIP 特征向量（512维，L2归一化）"""
        tensor, _ = self._tensors(img)
        out = self._run_clip(tensor)
        return self._l2(out[0].flatten().astype(np.float32))

    def extract_resnet_from_image(self, img) -> np.ndarray:
        """提取 ResNet50 特征向量（2048维，L2归一化）"""
        _, tensor = self._tensors(img)
        out = self._run_resnet(tensor)
        return self._l2(out[0].flatten().astype(np.float32))

    def extract_both_from_image(self, img) -> dict:
        """一次张量转换，同时喂给两个模型"""
        t_clip, t_resnet = self._tensors(img)
        clip_out = self._run_clip(t_clip)
        resnet_out = self._run_resnet(t_resnet)
        return {
            'clip': self._l2(clip_out[0].flatten().astype(np.float32)),
            'resnet': self._l2(resnet_out[0].flatten().astype(np.float32)),
        }

    def extract_both_batch_from_images(self, imgs: list) -> dict:
        """批量：一次张量转换喂两个模型（输入尺寸相同时只需 resize 一次）"""
        t_clip, t_resnet = self._tensors_batch(imgs)
        clip_out = self._run_clip(t_clip)
        resnet_out = self._run_resnet(t_resnet)
        return {
            'clip': self._l2(clip_out[0].astype(np.float32), axis=1),
            'resnet': self._l2(resnet_out[0].astype(np.float32), axis=1),
        }

    # ==================== 兼容层：接收 bytes（签名与改造前一致） ====================
    # indexer.py / sync_products.py / app.py 走这些入口，无需改动

    def extract_clip(self, image_bytes: bytes) -> np.ndarray:
        return self.extract_clip_from_image(load_work_image(image_bytes))

    def extract_resnet(self, image_bytes: bytes) -> np.ndarray:
        return self.extract_resnet_from_image(load_work_image(image_bytes))

    def extract_both(self, image_bytes: bytes) -> dict:
        """同时提取 CLIP 和 ResNet 特征（入库用）。只解码一次。"""
        return self.extract_both_from_image(load_work_image(image_bytes))

    def extract_clip_batch(self, images: list[bytes]) -> np.ndarray:
        """批量提取 CLIP 特征（N, 512），已 L2 归一化"""
        return self.extract_clip_batch_from_images(
            [load_work_image(b) for b in images])

    def extract_resnet_batch(self, images: list[bytes]) -> np.ndarray:
        """批量提取 ResNet 特征（N, 2048），已 L2 归一化"""
        return self.extract_resnet_batch_from_images(
            [load_work_image(b) for b in images])

    def extract_both_batch(self, images: list[bytes]) -> dict:
        """批量同时提取"""
        return self.extract_both_batch_from_images(
            [load_work_image(b) for b in images])

    def extract_clip_batch_from_images(self, imgs: list) -> np.ndarray:
        t_clip, _ = self._tensors_batch(imgs)
        return self._l2(self._run_clip(t_clip)[0].astype(np.float32), axis=1)

    def extract_resnet_batch_from_images(self, imgs: list) -> np.ndarray:
        _, t_resnet = self._tensors_batch(imgs)
        return self._l2(self._run_resnet(t_resnet)[0].astype(np.float32), axis=1)


# 全局单例
_extractor: DualModelExtractor | None = None
_extractor_lock = threading.Lock()


def get_extractor() -> DualModelExtractor:
    """获取全局双模型提取器单例"""
    global _extractor
    if _extractor is None:
        with _extractor_lock:
            if _extractor is None:
                _extractor = DualModelExtractor()
    return _extractor
