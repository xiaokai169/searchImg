
import faiss
import numpy as np
import json

# 加载索引
index = faiss.read_index('data/faiss_clip_index.bin')
print(f'向量总数: {index.ntotal}')
print(f'向量维度: {index.d}')

# IndexIDMap 不支持 reconstruct，搜索一个随机向量来间接查看
# 用零向量搜索，返回最近的向量
q = np.random.randn(1, 512).astype(np.float32)
q = q / np.linalg.norm(q)
D, I = index.search(q, 3)

print(f'\n前3个最近向量:')
for i, (dist, fid) in enumerate(zip(D[0], I[0])):
    print(f'  faiss_id={fid}, 相似度(余弦)={dist:.4f}')

# 看 id_map.json 知道 next_id
with open('data/id_map.json') as f:
    print(f'\n{json.load(f)}')