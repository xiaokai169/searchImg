# 以图搜图 v2.0 — API 接口文档

**Base URL:** `http://<服务器IP>:5000`

---

## 1. 健康检查

> 不限流，供负载均衡/监控探测使用

```
GET /api/health
```

**响应：**
```json
{
  "code": 0,
  "status": "ok",
  "concurrency": {
    "max": 10,
    "in_use": 0
  }
}
```

| 字段 | 说明 |
|------|------|
| in_use | 当前正在处理的请求数 |
| max | 最大并发上限 |

---

## 2. 系统初始化状态

```
GET /api/init
```

**响应：**
```json
{
  "code": 0,
  "msg": "系统就绪",
  "data": {
    "indexed_vectors": 500,
    "db_records": 500,
    "clip_dim": 512,
    "resnet_dim": 2048
  }
}
```

---

## 3. 统计信息

```
GET /api/stats
```

**响应：**
```json
{
  "code": 0,
  "data": {
    "db_records": 500,
    "indexed_vectors": 500,
    "resnet_features": 500,
    "by_category": [
      { "category": "包包", "count": 200 },
      { "category": "鞋子", "count": 150 }
    ],
    "cache_size": 3,
    "clip_dim": 512,
    "resnet_dim": 2048
  }
}
```

---

## 4. 以图搜图（核心）

> 上传图片 → 品类自动识别 → CLIP 粗召回 → ResNet 精排 → 共识过滤

```
POST /api/search
Content-Type: multipart/form-data
```

### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | ✅ | — | 待搜索的图片（JPG/PNG/WebP/BMP/GIF，≤16MB） |
| — | — | — | — | **建议**：调用方先把图压到长边 ≤1024、JPEG q≈0.9 再上传（内置页面即如此）。手机原图 3-8MB 压后约 200KB，上传与后端解码都快一个数量级，且匹配精度不变 |
| category | String | 否 | — | 限定品类，不传则自动识别。可选值见下方 |
| top_k | Int | 否 | 20 | 返回结果数量 |
| auto_category | String | 否 | "1" | 是否自动识别品类，"0" 关闭 |

### 品类枚举

`包包` `鞋子` `衣服` `裤子` `裙子` `配饰` `其他`

### 成功响应

```json
{
  "code": 0,
  "data": {
    "results": [
      {
        "id": 10,
        "faiss_id": 0,
        "image_url": "https://obs.example.com/images/bag001.jpg",
        "category": "包包",
        "product_name": "Leather Handbag",
        "product_name_cn": "真皮手提包",
        "product_id": "SKU001",
        "clip_score": 0.82,
        "resnet_score": 0.76,
        "fused_score": 0.79
      }
    ],
    "total_db": 500,
    "query_time_ms": 320.5,
    "breakdown": {
      "decode_ms": 48.2,
      "sharpness_ms": 9.1,
      "feature_extraction_ms": 250.1,
      "search_rerank_ms": 70.4
    },
    "model": "CLIP-ViT-B/32 + ResNet50",
    "fusion_alpha": 0.35,
    "cached": false,
    "predicted_category": "包包",
    "predict_confidence": 0.85,
    "filter_source": "auto",
    "consensus_applied": true,
    "consensus_word": "handbag",
    "results_before_consensus": 15,
    "top_score": 0.79,
    "min_threshold": 0.7,
    "ocr_keywords": [],
    "ocr_raw_texts": [],
    "text_boost_applied": false,
    "sharpness": 142.6,
    "sharpness_threshold": 12.0,
    "sharpness_enforced": false,
    "zoom_used": 1.4,
    "zooms_available": [1.0, 1.4, 2.0, 2.8],
    "best_zoom_score": 0.6758
  }
}
```

> `sharpness`：图片清晰度得分（拉普拉斯方差）。**该分数在长边 1024 的工作图上计算**，
> 不要与全分辨率的数值比较。`sharpness_enforced=false` 表示当前为影子模式，
> 分数仅供参考、不参与拒绝。
>
> `zoom_used` / `zooms_available` / `best_zoom_score`：多尺度检索诊断信息。
> 服务端对查询图做多档中心裁切后分别检索（zoom 越大裁得越紧，用于应对商品在
> 画面中占比不足、背景干扰较大的手机实拍照片），最终采用 top1 分数最高的那一档。
> `zoom_used` 即该档，`best_zoom_score` 为其 top1 分数。

### 结果字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Int | 图片数据库 ID |
| image_url | String | 图片 OBS 地址 |
| category | String | 产品品类 |
| product_name | String | 产品英文名 |
| product_name_cn | String | 产品中文名 |
| product_id | String | 产品 SKU/编号 |
| clip_score | Float | CLIP 语义相似度（0~1） |
| resnet_score | Float | ResNet 纹理细节相似度（0~1） |
| fused_score | Float | 融合得分（0~1），**按此字段排序，值越高越相似** |

### 元数据字段说明

| 字段 | 说明 |
|------|------|
| predicted_category | 自动识别的品类 |
| predict_confidence | 识别置信度（0~1） |
| filter_source | `auto`=自动识别 / `manual`=手动指定 / `auto_empty`=品类无数据全库搜索 |
| consensus_applied | 是否触发了产品名共识过滤 |
| consensus_word | 共识关键词 |
| results_before_consensus | 过滤前的结果数 |
| top_score | 第一名融合分 |
| cached | 是否命中缓存 |

### 无匹配结果

```json
{
  "code": 0,
  "data": {
    "results": [],
    "total_db": 500,
    "top_score": 0.0,
    "min_threshold": 0.7
  }
}
```

### 图片不清晰

```json
{
  "code": -1,
  "msg": "图片清晰度不足（得分 8.3，要求 ≥12）。请上传更清晰的图片。",
  "sharpness": 8.3,
  "sharpness_threshold": 12.0
}
```

> 仅当 `sharpness_enforced=true` 时才可能出现此错误。得分在长边 1024 的工作图上计算。

### 并发已满

```json
{
  "code": -1,
  "msg": "服务器繁忙（当前并发 10），请稍后重试"
}
```
> HTTP 状态码：503

---

## 5. 入库单张图片

```
POST /api/add_image
Content-Type: multipart/form-data
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | 图片文件 |
| image_url | String | 否 | 图片 OBS 地址 |
| category | String | 否 | 品类，默认"其他" |
| product_name | String | 否 | 产品英文名 |
| product_name_cn | String | 否 | 产品中文名 |
| product_id | String | 否 | 产品 ID |
| keywords_cn | String | 否 | 中文搜索关键词 |
| keywords_en | String | 否 | 英文搜索关键词 |

### 响应

```json
{
  "code": 0,
  "msg": "入库成功",
  "data": {
    "id": 1,
    "faiss_id": 0,
    "clip_score": 0.95,
    "category": "包包"
  }
}
```

---

## 6. 批量入库

```
POST /api/add_batch
Content-Type: multipart/form-data
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | ✅ | 多个图片文件 |
| image_urls | String | 否 | URL JSON 数组 `["url1","url2"]` |
| product_names | String | 否 | 英文名 JSON 数组 |
| product_names_cn | String | 否 | 中文名 JSON 数组 |
| product_ids | String | 否 | ID JSON 数组 |
| keywords_cn_list | String | 否 | 中文关键词 JSON 数组 |
| keywords_en_list | String | 否 | 英文关键词 JSON 数组 |
| category | String | 否 | 品类 |

### 响应

```json
{
  "code": 0,
  "msg": "5成功 0失败",
  "data": {
    "success": 5,
    "failed": 0
  }
}
```

---

## 7. 索引管理

### 查看索引重建进度

```
GET /api/index/status
```

```json
{
  "code": 0,
  "data": {
    "status": "idle",
    "progress": 0,
    "total": 0
  }
}
```

### 重建索引

```
POST /api/index/rebuild
```

```json
{ "code": 0, "msg": "已启动索引重建" }
```

### 手动保存索引

```
POST /api/save_index
```

```json
{ "code": 0, "msg": "索引已保存, 500 个向量" }
```

---

## 错误码说明

| HTTP 状态码 | code | 含义 |
|-------------|------|------|
| 200 | 0 | 成功 |
| 400 | -1 | 参数错误/图片无效/清晰度不足 |
| 409 | -1 | 索引重建进行中，不可重复操作 |
| 500 | -1 | 服务内部错误 |
| 503 | -1 | 并发已满，请稍后重试 |

---

## curl 示例

```bash
# 健康检查
curl http://localhost:5000/api/health

# 以图搜图
curl -X POST http://localhost:5000/api/search \
  -F "file=@/path/to/image.jpg"

# 限定品类搜索
curl -X POST http://localhost:5000/api/search \
  -F "file=@/path/to/image.jpg" \
  -F "category=包包"

# 入库单张
curl -X POST http://localhost:5000/api/add_image \
  -F "file=@/path/to/image.jpg" \
  -F "image_url=https://obs.example.com/img.jpg" \
  -F "category=包包" \
  -F "product_name=Handbag" \
  -F "product_name_cn=手提包"

# 批量入库
curl -X POST http://localhost:5000/api/add_batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "image_urls=[\"https://obs.example.com/1.jpg\",\"https://obs.example.com/2.jpg\"]" \
  -F "category=鞋子"

# 查看统计
curl http://localhost:5000/api/stats
```

---

## Web 界面

浏览器直接访问 `http://<服务器IP>:5000/` 即可使用图形化搜索界面。
