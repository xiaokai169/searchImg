"""
两阶段检索引擎
阶段1: CLIP向量 FAISS 粗召回 Top200
阶段2: ResNet50 细粒度精排 + 分数融合
"""
import os
import json
import threading
import numpy as np
import faiss
from config import (
    FAISS_INDEX_PATH, ID_MAP_PATH, RESNET_FEATURES_PATH,
    CLIP_FEATURE_DIM, RESNET_FEATURE_DIM,
    COARSE_TOP_K, FINAL_TOP_K, FUSION_ALPHA,
    USE_CLIP_FOR_RANKING, SCORE_STRETCH_MIN,
    MIN_RESNET_SCORE, MIN_RELATIVE_SCORE,
    MIN_FUSED_SCORE, MIN_TOP_SCORE, CATEGORY_CLASSIFY_CONFIDENCE,
    NAME_ANCHOR_MIN_MATCH, NAME_ANCHOR_MIN_RESULTS,
)


class TwoStageEngine:
    """CLIP 粗召回 + ResNet50 精排引擎（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 0

        # FAISS 索引：存储 CLIP 向量（用于粗召回）
        inner = faiss.IndexFlatIP(CLIP_FEATURE_DIM)  # 内积 = 余弦相似度
        self.clip_index = faiss.IndexIDMap(inner)

        # ResNet 特征存储：faiss_id → resnet_vector (2048维)
        # 使用 dict 存储，查询时批量计算相似度
        self.resnet_features: dict[int, np.ndarray] = {}

    # ==================== 写入 ====================

    def add(self, clip_vectors: np.ndarray, resnet_vectors: np.ndarray,
            ids: np.ndarray) -> None:
        """
        批量添加双模型向量
        clip_vectors:  (N, 512)  float32，已 L2 归一化
        resnet_vectors: (N, 2048) float32，已 L2 归一化
        ids:           (N,)      int64
        """
        if len(clip_vectors) == 0:
            return
        clip_vectors = np.asarray(clip_vectors, dtype=np.float32)
        resnet_vectors = np.asarray(resnet_vectors, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)

        with self._lock:
            self.clip_index.add_with_ids(clip_vectors, ids)
            for i, fid in enumerate(ids):
                self.resnet_features[int(fid)] = resnet_vectors[i]
            self._next_id = max(self._next_id, int(ids.max()) + 1)

    def add_single(self, clip_vec: np.ndarray, resnet_vec: np.ndarray,
                   faiss_id: int) -> None:
        """添加单张图片的双模型向量"""
        self.add(
            np.expand_dims(clip_vec, axis=0),
            np.expand_dims(resnet_vec, axis=0),
            np.array([faiss_id], dtype=np.int64),
        )

    def allocate_id(self) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1
            return fid

    # ==================== 两阶段检索 ====================

    def search(self, clip_query: np.ndarray, resnet_query: np.ndarray,
               top_k: int = FINAL_TOP_K,
               category_filter: list[int] | None = None,
               min_resnet: float = MIN_RESNET_SCORE,
               min_relative: float = MIN_RELATIVE_SCORE,
               min_fused: float = MIN_FUSED_SCORE,
               min_top: float = MIN_TOP_SCORE) -> list[dict]:
        """
        两阶段检索
        阶段1: CLIP FAISS粗召回 → 候选集
        阶段2: ResNet50细粒度精排 → 分数拉伸 → 双阈值过滤
        """
        clip_query = np.asarray(clip_query, dtype=np.float32)
        resnet_query = np.asarray(resnet_query, dtype=np.float32)

        with self._lock:
            if self.clip_index.ntotal == 0:
                return []

            # === 阶段1: CLIP 粗召回（语义级） ===
            fetch_k = min(COARSE_TOP_K, self.clip_index.ntotal)
            distances, indices = self.clip_index.search(
                np.expand_dims(clip_query, axis=0), fetch_k
            )

            candidates = []
            for dist, fid in zip(distances[0], indices[0]):
                if fid < 0:
                    continue
                if category_filter is not None and len(category_filter) > 0:
                    if int(fid) not in category_filter:
                        continue
                candidates.append({
                    'faiss_id': int(fid),
                    'clip_score': round(float(dist), 4),
                })

            if not candidates:
                return []

            # === 阶段2: ResNet50 细粒度精排 ===
            candidate_ids = [c['faiss_id'] for c in candidates]
            resnet_raw = self._compute_resnet_scores(resnet_query, candidate_ids)

            for c, r_raw in zip(candidates, resnet_raw):
                r_raw = float(r_raw)
                c['resnet_score'] = round(r_raw, 4)

                # 分数拉伸：把 [0.45, 1.0] 映射到 [0, 1]
                stretched = (r_raw - SCORE_STRETCH_MIN) / (1.0 - SCORE_STRETCH_MIN)
                stretched = max(0.0, min(1.0, stretched))  # 裁剪
                c['fused_score'] = round(stretched, 4)

            # === 排序 + 双阈值过滤 ===
            # 按拉伸后的 ResNet 分排序
            candidates.sort(key=lambda x: x['fused_score'], reverse=True)

            if not candidates:
                return []

            top_score = candidates[0]['fused_score']
            # 最高分低于门槛 → 全库没有真正匹配
            if top_score < min_top:
                return []
            # 有效阈值: ResNet原始分≥min_resnet 且 拉伸分≥最高拉伸分×min_relative 且 拉伸分≥min_fused
            results = []
            for c in candidates:
                if c['resnet_score'] < min_resnet:
                    continue
                if c['fused_score'] < top_score * min_relative:
                    continue
                if c['fused_score'] < min_fused:
                    continue
                results.append(c)

            return results[:top_k]

    def _compute_resnet_scores(self, query: np.ndarray,
                                candidate_ids: list[int]) -> np.ndarray:
        """
        批量计算 ResNet 余弦相似度
        query: (2048,) 已归一化
        candidate_ids: 候选 faiss_id 列表
        返回: (len(candidates),) 相似度数组
        """
        # 收集候选的 ResNet 向量
        vecs = []
        for fid in candidate_ids:
            v = self.resnet_features.get(fid)
            if v is not None:
                vecs.append(v)
            else:
                vecs.append(np.zeros(RESNET_FEATURE_DIM, dtype=np.float32))

        matrix = np.stack(vecs, axis=0)  # (M, 2048)
        # 余弦相似度 = 内积（均已 L2 归一化）
        scores = np.dot(matrix, query)    # (M,)
        return scores

    # ==================== 品类分类 ====================

    def build_category_prototypes(self) -> dict[str, np.ndarray]:
        """
        为每个品类构建 CLIP 原型向量（平均值）。
        返回 {category_name: (512,) L2归一化向量}
        """
        from database import get_all_images
        images = get_all_images()

        # 按品类聚合 CLIP 向量
        cat_vecs: dict[str, list[np.ndarray]] = {}
        for img in images:
            fid = img['faiss_id']
            cat = str(img['category'])
            try:
                vec = self.clip_index.reconstruct(int(fid))
                cat_vecs.setdefault(cat, []).append(np.asarray(vec, dtype=np.float32))
            except Exception:
                continue

        prototypes = {}
        for cat, vecs in cat_vecs.items():
            if len(vecs) >= 2:  # 至少2张图才有统计意义
                avg = np.mean(np.stack(vecs, axis=0), axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-8)
                prototypes[cat] = avg
            else:
                prototypes[cat] = vecs[0]  # 单张直接用

        return prototypes

    def classify_category(self, clip_vec: np.ndarray,
                          prototypes: dict[str, np.ndarray] | None = None,
                          min_confidence: float = CATEGORY_CLASSIFY_CONFIDENCE
                          ) -> tuple[str | None, float, dict[str, float]]:
        """
        用 CLIP 特征对图片做品类分类。
        返回 (predicted_category, confidence, all_scores_dict)
        置信度 < min_confidence 时返回 (None, ...) 表示不确定。
        """
        if not prototypes:
            return None, 0.0, {}

        clip_vec = np.asarray(clip_vec, dtype=np.float32).flatten()
        scores = {}
        for cat, proto in prototypes.items():
            scores[cat] = float(np.dot(clip_vec, proto))

        # 按分数排序
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = sorted_cats[0]

        if len(sorted_cats) >= 2:
            second_score = sorted_cats[1][1]
            # 置信度 = 第一名与第二名的差距
            confidence = round(best_score - second_score, 4)
        else:
            confidence = best_score

        if confidence < min_confidence:
            return None, confidence, dict(sorted_cats)

        return best_cat, confidence, dict(sorted_cats)

    @staticmethod
    def filter_by_product_name(results: list[dict],
                                min_match: int = NAME_ANCHOR_MIN_MATCH,
                                min_results: int = NAME_ANCHOR_MIN_RESULTS
                                ) -> list[dict]:
        """
        产品名称锚定过滤：
        以第一名（最高视觉分）的产品名为基准，提取关键词，
        只保留产品名与基准词有交集的结果。品牌名自动忽略。
        过滤后结果 < min_results 时不过滤（避免误杀小结果集）。
        """
        if not results or len(results) < min_results:
            return results  # 太少，不冒险

        import re

        # 品牌名（跨品类通用，不能用作区分词）
        BRANDS = {
            'xiaomi', 'redmi', 'samsung', 'apple', 'macbook', 'galaxy',
            'black', 'shark', 'huawei', 'honor', 'oneplus', 'oppo', 'vivo',
            'realme', 'nokia', 'motorola', 'google', 'pixel', 'lenovo',
            'dell', 'asus', 'acer', 'sony', 'lg', 'philips', 'panasonic',
            'bosch', 'siemens',
        }

        # 通用停用词
        STOPS = {
            'inch', 'with', 'and', 'for', 'the', 'new', 'hot', 'best',
            'high', 'quality', 'premium', 'sale', 'free', 'size', 'color',
            'large', 'small', 'medium', 'style', 'model', 'brand', 'made',
            'china', 'product', 'goods', 'item', 'type', 'set', 'pack',
            'piece', 'unit', 'each', 'per', 'cm', 'mm', 'meter', 'gram',
            'kg', 'dual', 'sim', 'ram', 'rom', 'version', 'global', 'middle',
            'east', 'black', 'white', 'blue', 'grey', 'gold', 'silver',
            'portable', 'smart', 'magnetic', 'liquid', 'silicone', 'matte',
            'flash', 'magic', 'tempered', 'glass', 'screen', 'protector',
        }

        def extract_keywords(name: str) -> set[str]:
            if not name:
                return set()
            tokens = re.split(r'[\s\-/,.;:()\[\]{}|]+', name.lower())
            return {
                t for t in tokens
                if len(t) > 3 and t not in STOPS and t not in BRANDS
                and not t.isdigit() and not t.replace('.', '').isdigit()
            }

        # 以第一名产品名为锚
        anchor_words = extract_keywords(results[0].get('product_name', ''))
        if not anchor_words:
            return results  # 锚点无有效词，不过滤

        # 自适应匹配数：锚点词多 → 要求匹配更多
        required = min_match if len(anchor_words) <= 2 else max(2, len(anchor_words) // 2)

        # 过滤
        filtered = [results[0]]  # 第一名永远保留
        for r in results[1:]:
            rwords = extract_keywords(r.get('product_name', ''))
            if len(anchor_words & rwords) >= required:
                filtered.append(r)

        # 返回过滤结果（第一名永远保留，至少1条）
        return filtered

    # ==================== 查询 ====================

    @property
    def total(self) -> int:
        return self.clip_index.ntotal

    @property
    def resnet_count(self) -> int:
        return len(self.resnet_features)

    # ==================== 持久化 ====================

    def save(self) -> None:
        """保存 FAISS 索引 + ResNet 特征到磁盘"""
        with self._lock:
            # 保存 FAISS CLIP 索引
            faiss.write_index(self.clip_index, FAISS_INDEX_PATH)

            # 保存 ResNet 特征（faiss_id 排序后存入 numpy）
            ids = sorted(self.resnet_features.keys())
            if ids:
                matrix = np.stack([self.resnet_features[i] for i in ids], axis=0)
                np.save(RESNET_FEATURES_PATH, matrix)
                # 同时保存 id 顺序
                np.save(RESNET_FEATURES_PATH.replace('.npy', '_ids.npy'),
                        np.array(ids, dtype=np.int64))
            else:
                # 空特征
                np.save(RESNET_FEATURES_PATH, np.empty((0, RESNET_FEATURE_DIM), dtype=np.float32))

            # 保存 next_id
            with open(ID_MAP_PATH, 'w') as f:
                json.dump({"next_id": self._next_id}, f)

        print(f"[Engine] 已保存: {self.clip_index.ntotal} CLIP向量, "
              f"{len(self.resnet_features)} ResNet特征")

    def load(self) -> bool:
        """从磁盘加载索引和特征"""
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"[Engine] 索引文件不存在，创建新索引")
            return False

        try:
            # 加载 FAISS CLIP 索引
            self.clip_index = faiss.read_index(FAISS_INDEX_PATH)
            n = self.clip_index.ntotal
            print(f"[Engine] CLIP索引已加载: {n} 个向量")

            # 加载 ResNet 特征
            ids_path = RESNET_FEATURES_PATH.replace('.npy', '_ids.npy')
            if os.path.exists(RESNET_FEATURES_PATH) and os.path.exists(ids_path):
                matrix = np.load(RESNET_FEATURES_PATH)          # (N, 2048)
                ids = np.load(ids_path)                          # (N,)
                self.resnet_features = {
                    int(fid): matrix[i] for i, fid in enumerate(ids)
                }
                print(f"[Engine] ResNet特征已加载: {len(self.resnet_features)} 个")

            # 恢复 next_id
            if os.path.exists(ID_MAP_PATH):
                with open(ID_MAP_PATH, 'r') as f:
                    data = json.load(f)
                self._next_id = data.get('next_id', n)
            else:
                self._next_id = n

            return True

        except Exception as e:
            print(f"[Engine] 加载失败: {e}，创建新索引")
            self.clip_index = faiss.IndexIDMap(faiss.IndexFlatIP(CLIP_FEATURE_DIM))
            self.resnet_features = {}
            return False


# 全局单例
_engine: TwoStageEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> TwoStageEngine:
    """获取全局检索引擎单例"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TwoStageEngine()
    return _engine
