"""
两阶段检索引擎
阶段1: CLIP向量 FAISS 粗召回 Top200
阶段2: ResNet50 细粒度精排 + 分数融合
"""
import os
import re
import json
import math
import threading
from collections import Counter

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
    CONSENSUS_RERANK_TOP_N, CONSENSUS_RERANK_WEIGHT,
    CONSENSUS_MIN_DF, CONSENSUS_MIN_DF_RATIO,
)

# ==================== 产品名 / 关键词分词 ====================
# 这些常量为 filter_by_product_name（硬过滤）与 rerank_by_consensus（共识重排）共用。
# 两者对"什么词能代表品类"的判据一致，只是使用方式不同（过滤 vs 重排）。

# 品牌名：跨品类通用，不能用作区分词
_BRANDS = {
    'xiaomi', 'redmi', 'samsung', 'apple', 'macbook', 'galaxy',
    'black', 'shark', 'huawei', 'honor', 'oneplus', 'oppo', 'vivo',
    'realme', 'nokia', 'motorola', 'google', 'pixel', 'lenovo',
    'dell', 'asus', 'acer', 'sony', 'lg', 'philips', 'panasonic',
    'bosch', 'siemens',
}

# 通用停用词
_STOPS = {
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

# 关键词/产品名的分词边界
_TOKEN_SPLIT_RE = re.compile(r'[\s\-/,.;:()\[\]{}|]+')


def _consensus_terms(r: dict) -> set[str]:
    """
    提取一条结果的「品类词」，供共识投票使用。

    信号覆盖 keywords_en / keywords_cn / product_name 三个字段。
    keywords_* 是入库时构建的商品词表（库中 98.8% 覆盖），比 product_name 裸分词
    干净且完整 —— 此前它在检索链路里只写不读。
    """
    terms: set[str] = set()
    for field in ('keywords_en', 'keywords_cn', 'product_name'):
        s = r.get(field) or ''
        if not s:
            continue
        for t in _TOKEN_SPLIT_RE.split(str(s).lower()):
            t = t.strip()
            if len(t) < 2 or t in _STOPS or t in _BRANDS:
                continue
            if t.isdigit() or t.replace('.', '').isdigit():
                continue
            terms.add(t)
    return terms


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """先写临时文件再原子替换，避免写到一半进程退出留下损坏的 .npy"""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, array)
    os.replace(tmp, path)


class TwoStageEngine:
    """CLIP 粗召回 + ResNet50 精排引擎（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 0
        self._dirty = False          # 是否有未落盘的写入

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
            self._dirty = True

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
        if not images or self.clip_index.ntotal == 0:
            return {}

        # IndexIDMap 不实现 reconstruct，需穿过内层 IndexFlat 按内部位置取。
        # 一次性取全量，避免逐条 C++ 调用
        inner = self.clip_index.index
        all_vecs = inner.reconstruct_n(0, inner.ntotal)           # (N, 512)
        id_map = faiss.vector_to_array(self.clip_index.id_map)    # 内部位置 → 外部 faiss_id
        pos_of = {int(fid): i for i, fid in enumerate(id_map)}

        # 按品类聚合 CLIP 向量
        cat_vecs: dict[str, list[np.ndarray]] = {}
        for img in images:
            pos = pos_of.get(int(img['faiss_id']))
            if pos is None:            # 库里存在但索引里没有（漏提取），跳过
                continue
            cat_vecs.setdefault(str(img['category']), []).append(all_vecs[pos])

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

        def extract_keywords(name: str) -> set[str]:
            if not name:
                return set()
            tokens = _TOKEN_SPLIT_RE.split(name.lower())
            result = set()
            for t in tokens:
                if not t or t in _STOPS or t in _BRANDS:
                    continue
                if t.isdigit() or t.replace('.', '').isdigit():
                    continue
                # 英文词 ≥2 字符即保留（如 CT, USB），中文按单字拆分
                if len(t) >= 2:
                    result.add(t)
                    for ch in t:
                        if '一' <= ch <= '鿿':  # 中文字符
                            result.add(ch)
            return result

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

    # ==================== 共识重排 ====================

    @staticmethod
    def rerank_by_consensus(results: list[dict],
                            top_n: int = CONSENSUS_RERANK_TOP_N,
                            weight: float = CONSENSUS_RERANK_WEIGHT,
                            min_df: int = CONSENSUS_MIN_DF,
                            min_df_ratio: float = CONSENSUS_MIN_DF_RATIO,
                            ) -> tuple[list[dict], set[str], int]:
        """
        用 top-K 结果集的词频共识重排候选（只改顺序，不删结果）。

        与 filter_by_product_name 的三点区别，正是后者救不了手机实拍的原因：
          基准 —— 取 top-K 全体投票，而非 top1 单条当锚 → 对错误的 top1 免疫
                  （实测 459b 的 top1 是 'Office chair'（错），拿它当锚会砍光正确答案）
          信号 —— 叠加 keywords_en / keywords_cn，而非只用 product_name 裸分词
          动作 —— 重排而非硬过滤 → 结果集不会因为重排而变空

        打分：取该结果命中的共识词中「文档频率最高的那个」，按全局最高 df 归一化后
        乘以 weight 加分。用「最强 df」而非「df 求和」，是为了让加分体现【归属于哪个
        词簇】而不是【匹配上几个词】—— 否则次要词簇能靠词数堆出更高加分：
        实测 31a5 中埋地灯簇合计匹配 4 个词、移动CT簇只匹配 2 个，求和会让埋地灯反超。

        返回 (重排后的列表, 共识词集合, 被加分的条数)
        """
        if len(results) < 2 or top_n < 2:
            return results, set(), 0

        window = results[:top_n]                 # 已按 fused_score 降序
        term_sets = [_consensus_terms(r) for r in window]

        # 文档频率：每个词出现在多少条候选里
        df: Counter = Counter()
        for ts in term_sets:
            df.update(ts)

        threshold = max(min_df, math.ceil(len(window) * min_df_ratio))
        consensus = {w for w, c in df.items() if c >= threshold}
        if not consensus:
            return results, set(), 0

        max_df = max(df[w] for w in consensus)
        boosted = 0
        for r, ts in zip(window, term_sets):
            best = max((df[w] for w in (ts & consensus)), default=0)
            coverage = best / max_df
            r['consensus_score'] = round(coverage, 4)
            if coverage <= 0:
                continue
            r['fused_score'] = round(min(1.0, r['fused_score'] + weight * coverage), 4)
            boosted += 1

        # window 之外的结果不参与投票也不加分，天然排在 window 之后
        # （window 本就是分数最高的一批，加分只会拉大差距，不会破坏该顺序）
        window.sort(key=lambda x: x['fused_score'], reverse=True)
        return window + results[top_n:], consensus, boosted

    # ==================== 查询 ====================

    @property
    def total(self) -> int:
        return self.clip_index.ntotal

    @property
    def resnet_count(self) -> int:
        return len(self.resnet_features)

    @property
    def dirty(self) -> bool:
        """是否存在尚未落盘的写入"""
        return self._dirty

    # ==================== 持久化 ====================

    def save(self) -> None:
        """保存 FAISS 索引 + ResNet 特征到磁盘（原子替换，避免半写损坏）"""
        with self._lock:
            # 保存 FAISS CLIP 索引
            faiss.write_index(self.clip_index, FAISS_INDEX_PATH + '.tmp')
            os.replace(FAISS_INDEX_PATH + '.tmp', FAISS_INDEX_PATH)

            # 保存 ResNet 特征（faiss_id 排序后存入 numpy）
            ids = sorted(self.resnet_features.keys())
            ids_path = RESNET_FEATURES_PATH.replace('.npy', '_ids.npy')
            if ids:
                matrix = np.stack([self.resnet_features[i] for i in ids], axis=0)
            else:
                # 空特征
                matrix = np.empty((0, RESNET_FEATURE_DIM), dtype=np.float32)
            _atomic_save_npy(RESNET_FEATURES_PATH, matrix)
            _atomic_save_npy(ids_path, np.array(ids, dtype=np.int64))

            # 保存 next_id
            tmp = ID_MAP_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({"next_id": self._next_id}, f)
            os.replace(tmp, ID_MAP_PATH)

            self._dirty = False

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
