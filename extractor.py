"""
双模型特征提取器
- CLIP ViT-B/32 (512维): 语义级粗召回
- ResNet50 (2048维): 细粒度精排
线程安全，支持批量推理
"""
import threading
import numpy as np
import onnxruntime as ort
from config import (
    CLIP_MODEL_PATH, CLIP_IMAGE_SIZE, CLIP_NUM_THREADS, CLIP_FEATURE_DIM,
    RESNET_MODEL_PATH, RESNET_IMAGE_SIZE, RESNET_NUM_THREADS, RESNET_FEATURE_DIM,
)
from preprocess import preprocess_for_model


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

        self._lock = threading.Lock()

    # ==================== CLIP 特征（粗召回用） ====================

    def extract_clip(self, image_bytes: bytes) -> np.ndarray:
        """提取 CLIP 特征向量（512维，L2归一化）"""
        tensor = preprocess_for_model(image_bytes, target_size=CLIP_IMAGE_SIZE)
        with self._lock:
            result = self.clip_session.run([self.clip_output], {self.clip_input: tensor})
        vec = result[0].flatten().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def extract_clip_batch(self, images: list[bytes]) -> np.ndarray:
        """批量提取 CLIP 特征（N, 512），已 L2 归一化"""
        batch = []
        for img_bytes in images:
            t = preprocess_for_model(img_bytes, target_size=CLIP_IMAGE_SIZE)
            batch.append(t[0])
        batch = np.stack(batch, axis=0)
        with self._lock:
            result = self.clip_session.run([self.clip_output], {self.clip_input: batch})
        vecs = result[0].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    # ==================== ResNet 特征（精排用） ====================

    def extract_resnet(self, image_bytes: bytes) -> np.ndarray:
        """提取 ResNet50 特征向量（2048维，L2归一化）"""
        tensor = preprocess_for_model(image_bytes, target_size=RESNET_IMAGE_SIZE)
        with self._lock:
            result = self.resnet_session.run([self.resnet_output], {self.resnet_input: tensor})
        vec = result[0].flatten().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def extract_resnet_batch(self, images: list[bytes]) -> np.ndarray:
        """批量提取 ResNet 特征（N, 2048），已 L2 归一化"""
        batch = []
        for img_bytes in images:
            t = preprocess_for_model(img_bytes, target_size=RESNET_IMAGE_SIZE)
            batch.append(t[0])
        batch = np.stack(batch, axis=0)
        with self._lock:
            result = self.resnet_session.run([self.resnet_output], {self.resnet_input: batch})
        vecs = result[0].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    # ==================== 联合提取 ====================

    def extract_both(self, image_bytes: bytes) -> dict:
        """同时提取 CLIP 和 ResNet 特征（入库用）"""
        return {
            'clip': self.extract_clip(image_bytes),
            'resnet': self.extract_resnet(image_bytes),
        }

    def extract_both_batch(self, images: list[bytes]) -> dict:
        """批量同时提取"""
        return {
            'clip': self.extract_clip_batch(images),
            'resnet': self.extract_resnet_batch(images),
        }


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
