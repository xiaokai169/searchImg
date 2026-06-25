"""
以图搜图 v2.0 — CLIP 粗召回 + ResNet50 精排
不存本地文件，只存向量 + OBS URL
"""
import os
import time
import hashlib
import threading
from flask import Flask, request, jsonify, render_template_string

from config import (
    FINAL_TOP_K, CACHE_SIZE, DATA_DIR,
    CLIP_FEATURE_DIM, RESNET_FEATURE_DIM, FUSION_ALPHA,
    MIN_TOP_SCORE,
)
from database import init_database, get_total_count, get_category_stats
from engine import get_engine
from extractor import get_extractor
from preprocess import validate_image, check_sharpness
from indexer import add_single_image, add_images_batch, rebuild_index, get_index_progress

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(DATA_DIR, exist_ok=True)

_init_done = False
_init_lock = threading.Lock()


def _ensure_init():
    global _init_done
    if _init_done:
        return
    with _init_lock:
        if _init_done:
            return
        init_database()
        engine = get_engine()
        if not engine.load():
            print("[Init] 创建新索引")
        _init_done = True


class LRUCache:
    def __init__(self, maxsize: int = CACHE_SIZE):
        self._cache, self._order, self._maxsize = {}, [], maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
        return None

    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._maxsize:
                del self._cache[self._order.pop(0)]
            self._cache[key] = value
            self._order.append(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def __len__(self):
        return len(self._cache)


_search_cache = LRUCache(CACHE_SIZE)
_prototypes_cache: dict | None = None       # 品类原型缓存
_prototypes_lock = threading.Lock()


def _get_prototypes() -> dict:
    """获取或构建品类CLIP原型向量（懒加载，有锁缓存）"""
    global _prototypes_cache
    if _prototypes_cache is not None:
        return _prototypes_cache
    with _prototypes_lock:
        if _prototypes_cache is not None:
            return _prototypes_cache
        engine = get_engine()
        _prototypes_cache = engine.build_category_prototypes()
        print(f"[Prototypes] 已构建 {len(_prototypes_cache)} 个品类原型")
        return _prototypes_cache


def _cache_key(data: bytes, cat: str = '') -> str:
    return f"{hashlib.md5(data).hexdigest()}:{cat}"


def _find_consensus_word(all_results: list[dict], filtered: list[dict]) -> str:
    """找出用于过滤的共识词（用于前端展示）"""
    if not filtered or len(filtered) >= len(all_results):
        return ''
    # 从被过滤掉的结果中找出差异词
    import re
    STOP_WORDS = {
        'inch', 'with', 'and', 'for', 'the', 'new', 'hot', 'best',
        'high', 'quality', 'premium', 'sale', 'free', 'size', 'color',
        'large', 'small', 'medium', 'style', 'model', 'brand', 'made',
        'china', 'product', 'goods', 'item', 'type', 'set', 'pack',
        'piece', 'unit', 'each', 'per', 'cm', 'mm', 'meter', 'gram', 'kg',
    }
    def extract_words(name: str) -> set[str]:
        if not name:
            return set()
        tokens = re.split(r'[\s\-/,.;:()\[\]{}]+', name.lower())
        return {t for t in tokens if len(t) > 3 and t not in STOP_WORDS
                and not t.isdigit() and not t.replace('.', '').isdigit()}

    filtered_words = set()
    for r in filtered:
        filtered_words |= extract_words(r.get('product_name', ''))
    removed_words = set()
    for r in all_results:
        if r not in filtered:
            removed_words |= extract_words(r.get('product_name', ''))

    # 共识词 = 在保留结果中出现但在被移除结果中不出现的词
    consensus = filtered_words - removed_words
    return max(consensus, key=len) if consensus else ''


def _resolve(engine_results: list[dict]) -> list[dict]:
    from database import get_image_by_faiss_id
    results = []
    for item in engine_results:
        img = get_image_by_faiss_id(item['faiss_id'])
        if img is None:
            continue
        results.append({
            "id": int(img['id']),
            "image_url": str(img['image_url']),
            "category": str(img['category']),
            "product_name": str(img.get('product_name', '') or ''),
            "product_name_cn": str(img.get('product_name_cn', '') or ''),
            "product_id": str(img.get('product_id', '') or ''),
            "clip_score": float(item['clip_score']),
            "resnet_score": float(item['resnet_score']),
            "fused_score": float(item['fused_score']),
        })
    return results


# ==================== API ====================

@app.route('/api/init', methods=['GET'])
def api_init():
    try:
        _ensure_init()
        e = get_engine()
        return jsonify({"code": 0, "msg": "系统就绪",
                        "data": {"indexed_vectors": e.total, "db_records": get_total_count(),
                                 "clip_dim": CLIP_FEATURE_DIM, "resnet_dim": RESNET_FEATURE_DIM}})
    except Exception as ex:
        return jsonify({"code": -1, "msg": str(ex)}), 500


@app.route('/api/add_image', methods=['POST'])
def api_add_image():
    """入库单张：图片 bytes + image_url + category + 产品中英文名 + 关键字"""
    _ensure_init()
    try:
        if 'file' not in request.files:
            return jsonify({"code": -1, "msg": "请上传图片文件"}), 400
        file = request.files['file']
        image_url = request.form.get('image_url', '').strip()
        category = request.form.get('category', '其他').strip() or '其他'
        product_name = request.form.get('product_name', '').strip()
        product_name_cn = request.form.get('product_name_cn', '').strip()
        product_id = request.form.get('product_id', '').strip()
        keywords_cn = request.form.get('keywords_cn', '').strip()
        keywords_en = request.form.get('keywords_en', '').strip()
        if not image_url:
            image_url = file.filename or 'unknown'
        result = add_single_image(file.read(), image_url, category, product_name,
                                  product_name_cn, product_id, keywords_cn, keywords_en)
        _search_cache.clear()
        return jsonify({"code": 0, "msg": "入库成功", "data": result})
    except ValueError as e:
        return jsonify({"code": -1, "msg": str(e)}), 400
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)}), 500


@app.route('/api/add_batch', methods=['POST'])
def api_add_batch():
    """批量入库：files[] + image_urls(JSON) + category"""
    _ensure_init()
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"code": -1, "msg": "请上传图片文件"}), 400
        import json as _json
        urls = _json.loads(request.form.get('image_urls', '[]'))
        names = _json.loads(request.form.get('product_names', '[]'))
        names_cn = _json.loads(request.form.get('product_names_cn', '[]'))
        product_ids = _json.loads(request.form.get('product_ids', '[]'))
        keywords_cn_list = _json.loads(request.form.get('keywords_cn_list', '[]'))
        keywords_en_list = _json.loads(request.form.get('keywords_en_list', '[]'))
        category = request.form.get('category', '其他').strip() or '其他'
        items = []
        for i, f in enumerate(files):
            if not f.filename:
                continue
            url = urls[i] if i < len(urls) else f.filename
            pname = names[i] if i < len(names) else ''
            pname_cn = names_cn[i] if i < len(names_cn) else ''
            pid = str(product_ids[i]) if i < len(product_ids) else ''
            kw_cn = keywords_cn_list[i] if i < len(keywords_cn_list) else ''
            kw_en = keywords_en_list[i] if i < len(keywords_en_list) else ''
            items.append((f.read(), url, category, pname, pname_cn, pid, kw_cn, kw_en))
        if not items:
            return jsonify({"code": -1, "msg": "没有有效图片"}), 400
        result = add_images_batch(items)
        _search_cache.clear()
        return jsonify({"code": 0, "msg": f"{result['success']}成功 {result['failed']}失败", "data": result})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """以图搜图：上传图片 → 品类识别 → CLIP粗召回 → ResNet精排"""
    _ensure_init()
    t0 = time.time()
    try:
        if 'file' not in request.files:
            return jsonify({"code": -1, "msg": "请上传图片"}), 400
        file = request.files['file']
        image_bytes = file.read()
        category = request.form.get('category', '').strip()
        top_k = request.form.get('top_k', FINAL_TOP_K, type=int)
        auto_category = request.form.get('auto_category', '1') == '1'

        ok, err = validate_image(image_bytes)
        if not ok:
            return jsonify({"code": -1, "msg": err}), 400

        # 清晰度检测
        sharpness, is_clear, sharp_msg = check_sharpness(image_bytes)
        if not is_clear:
            return jsonify({"code": -1, "msg": sharp_msg, "sharpness": sharpness}), 400

        ck = _cache_key(image_bytes, category)
        cached = _search_cache.get(ck)
        if cached:
            cached['cached'] = True
            return jsonify({"code": 0, "data": cached})

        extractor = get_extractor()
        t_feat = time.time()
        clip_vec = extractor.extract_clip(image_bytes)
        resnet_vec = extractor.extract_resnet(image_bytes)
        feat_ms = round((time.time() - t_feat) * 1000, 1)

        # ====== OCR 文字提取 + 验证 ======
        ocr_keywords = []
        ocr_raw_texts = []
        try:
            from ocr_extract import extract_text
            ocr_raw = extract_text(image_bytes)
            if ocr_raw:
                ocr_raw_texts = [t for t, _ in ocr_raw]
                # 分词 + 去噪
                import re
                raw_keywords = set()
                for t, conf in ocr_raw:
                    for tok in re.split(r'[\s\-/,.;:()\[\]{}]+', t.lower()):
                        tok = tok.strip()
                        if len(tok) > 2 and not tok.isdigit():
                            raw_keywords.add(tok)
                # 验证：只保留在库里产品名中真实出现过的词
                if raw_keywords:
                    from database import get_all_images
                    all_imgs = get_all_images()
                    all_names = ' '.join(
                        (img.get('product_name') or '').lower() for img in all_imgs
                    )
                    ocr_keywords = [kw for kw in raw_keywords if kw in all_names]
                    if ocr_keywords:
                        print(f"[Search] OCR 有效关键词: {ocr_keywords}")
                    elif raw_keywords:
                        print(f"[Search] OCR 全部被过滤: {list(raw_keywords)[:10]}")
        except Exception as ex:
            print(f"[Search] OCR 未就绪: {ex}")

        # ====== 品类自动识别 ======
        predicted_cat = None
        predict_confidence = 0.0
        predict_scores = None
        if auto_category and not category:
            # 用户未手动选品类时才自动识别
            try:
                prototypes = _get_prototypes()
                if prototypes:
                    predicted_cat, predict_confidence, predict_scores = (
                        get_engine().classify_category(clip_vec, prototypes)
                    )
            except Exception as ex:
                print(f"[Search] 品类识别失败: {ex}")

        # 品类过滤：手动选择优先，否则用自动识别结果
        cat_filter = None
        filter_source = 'none'
        if category:
            from database import get_all_images
            imgs = get_all_images(category=category)
            cat_filter = [img['faiss_id'] for img in imgs]
            filter_source = 'manual'
            if not cat_filter:
                response_data = {
                    "results": [], "total_db": get_total_count(), "query_time_ms": 0,
                    "predicted_category": category, "filter_source": "manual"
                }
                return jsonify({"code": 0, "data": response_data})
        elif predicted_cat:
            from database import get_all_images
            imgs = get_all_images(category=predicted_cat)
            cat_filter = [img['faiss_id'] for img in imgs]
            filter_source = 'auto'
            if not cat_filter:
                # 品类下无数据，fallback 到全库搜索
                cat_filter = None
                filter_source = 'auto_empty'

        t_s = time.time()
        eng_results = get_engine().search(clip_vec, resnet_vec, top_k=top_k,
                                           category_filter=cat_filter, min_top=MIN_TOP_SCORE)
        search_ms = round((time.time() - t_s) * 1000, 1)

        results = _resolve(eng_results)

        # ====== OCR 文案加分（辅助，不影响视觉搜图主流程） ======
        text_boost_applied = False
        if ocr_keywords and results:
            for r in results:
                pname = (r.get('product_name') or '').lower()
                if not pname:
                    continue
                # 每个匹配到的关键词 +0.05 分（最多 +0.15）
                hits = sum(1 for kw in ocr_keywords if kw in pname)
                if hits > 0:
                    boost = min(hits * 0.05, 0.15)
                    old_score = r['fused_score']
                    r['fused_score'] = round(min(1.0, old_score + boost), 4)
                    r['text_boost'] = round(boost, 4)
                    text_boost_applied = True
            # 按加权后的分数重排
            if text_boost_applied:
                results.sort(key=lambda x: x['fused_score'], reverse=True)

        # 记录最高分（用于前端判断"是真没有还是被过滤了"）
        top_result_score = results[0]['fused_score'] if results else 0.0

        # ====== 产品名称共识过滤 ======
        consensus_applied = False
        consensus_word = ''
        results_before = len(results)
        if results:
            filtered = get_engine().filter_by_product_name(results)
            if len(filtered) < len(results):
                # 找出共识词用于前端显示
                for r in filtered:
                    if r.get('product_name'):
                        # 取第一个有意义词作为展示
                        break
                consensus_applied = True
                consensus_word = _find_consensus_word(results, filtered)
                results = filtered

        total_ms = round((time.time() - t0) * 1000, 1)

        data = {
            "results": results, "total_db": int(get_total_count()),
            "query_time_ms": float(total_ms),
            "breakdown": {"feature_extraction_ms": float(feat_ms), "search_rerank_ms": float(search_ms)},
            "model": "CLIP-ViT-B/32 + ResNet50", "fusion_alpha": float(FUSION_ALPHA), "cached": False,
            "predicted_category": predicted_cat,
            "predict_confidence": predict_confidence,
            "predict_scores": predict_scores,
            "filter_source": filter_source,
            "consensus_applied": consensus_applied,
            "consensus_word": consensus_word,
            "results_before_consensus": results_before,
            "top_score": round(top_result_score, 4),
            "min_threshold": float(MIN_TOP_SCORE),
            "ocr_keywords": ocr_keywords,
            "ocr_raw_texts": ocr_raw_texts,
            "text_boost_applied": text_boost_applied,
        }
        _search_cache.put(ck, data)
        return jsonify({"code": 0, "data": data})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    _ensure_init()
    try:
        e = get_engine()
        return jsonify({"code": 0, "data": {
            "db_records": get_total_count(), "indexed_vectors": e.total,
            "resnet_features": e.resnet_count, "by_category": get_category_stats(),
            "cache_size": len(_search_cache),
            "clip_dim": CLIP_FEATURE_DIM, "resnet_dim": RESNET_FEATURE_DIM,
        }})
    except Exception as ex:
        return jsonify({"code": -1, "msg": str(ex)}), 500


@app.route('/api/index/status', methods=['GET'])
def api_index_status():
    return jsonify({"code": 0, "data": get_index_progress()})


@app.route('/api/index/rebuild', methods=['POST'])
def api_rebuild_index():
    _ensure_init()
    try:
        rebuild_index()
        return jsonify({"code": 0, "msg": "已启动索引重建"})
    except RuntimeError as e:
        return jsonify({"code": -1, "msg": str(e)}), 409


@app.route('/api/save_index', methods=['POST'])
def api_save_index():
    """手动保存索引到磁盘"""
    _ensure_init()
    try:
        e = get_engine()
        if e.total > 0:
            e.save()
            return jsonify({"code": 0, "msg": f"索引已保存, {e.total} 个向量"})
        return jsonify({"code": 0, "msg": "索引为空，跳过保存"})
    except Exception as ex:
        return jsonify({"code": -1, "msg": str(ex)}), 500


# ==================== Web 搜索界面 ====================

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>以图搜图 v2.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#333}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:24px;text-align:center}
.header h1{font-size:24px;margin-bottom:4px}
.header p{font-size:13px;opacity:.7}
.container{max-width:960px;margin:0 auto;padding:20px}
.zone{border:2px dashed #d0d5dd;border-radius:12px;padding:36px;text-align:center;background:#fff;cursor:pointer;transition:all .2s}
.zone:hover,.zone.dragover{border-color:#0f3460;background:#f0f4ff}
.zone img.preview{max-width:260px;max-height:260px;border-radius:8px;margin-bottom:8px;display:none}
.zone .icon{font-size:40px;margin-bottom:8px}
.zone .text{font-size:14px;color:#666}
.zone .hint{font-size:11px;color:#999;margin-top:4px}
.row{display:flex;gap:10px;margin-top:16px;justify-content:center;flex-wrap:wrap}
.row select{padding:8px 14px;border:1px solid #d0d5dd;border-radius:8px;font-size:14px}
.row button{padding:10px 28px;background:#0f3460;color:#fff;border:none;border-radius:8px;font-size:15px;cursor:pointer}
.row button:disabled{opacity:.4;cursor:not-allowed}
.stats{display:flex;gap:12px;justify-content:center;margin:16px 0;flex-wrap:wrap}
.stat-item{background:#fff;border-radius:10px;padding:14px 20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.stat-item .num{font-size:22px;font-weight:700;color:#0f3460}
.stat-item .label{font-size:11px;color:#999}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-top:14px}
.card{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);transition:transform .15s}
.card:hover{transform:translateY(-2px)}
.card img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.card .info{padding:10px}
.card .score{font-size:20px;font-weight:700;color:#e94560}
.card .cat{font-size:11px;color:#888}
.card .bar{height:3px;background:#eee;border-radius:2px;margin-top:4px}
.card .bar div{height:100%;background:#4caf50;border-radius:2px}
.loading{text-align:center;padding:40px}
.spinner{width:36px;height:36px;border:3px solid #e0e0e0;border-top-color:#e94560;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
.info-line{text-align:center;font-size:12px;color:#888;margin:8px 0}
input[type=file]{display:none}
</style>
</head>
<body>
<div class="header"><h1>以图搜图 v2.0</h1><p>CLIP ViT-B/32 + ResNet50 | 双模型融合</p></div>
<div class="container">
<div class="stats" id="stats">
<div class="stat-item"><div class="num" id="stat-total">-</div><div class="label">已索引</div></div>
<div class="stat-item"><div class="num" id="stat-clip">512d</div><div class="label">CLIP</div></div>
<div class="stat-item"><div class="num" id="stat-resnet">2048d</div><div class="label">ResNet</div></div>
</div>
<div class="zone" id="zone">
<div class="icon" id="icon">&#128228;</div>
<img class="preview" id="pv" alt="">
<div class="text" id="ut">点击上传或拖拽图片到此处</div>
<div class="hint">JPG/PNG/WebP，最大5MB</div>
</div>
<input type="file" id="fi" accept="image/*">
<div class="row">
<select id="cat"><option value="">自动识别品类</option></select>
<button id="sb" disabled>搜索相似图片</button>
</div>
<div class="info-line" id="bd"></div>
<div class="info-line" id="cat-info" style="color:#0f3460;font-weight:500"></div>
<div id="res"></div>
</div>
<script>
var sf=null;
var sb=document.getElementById('sb');
var fi=document.getElementById('fi');
var cat=document.getElementById('cat');
var res=document.getElementById('res');
var bd=document.getElementById('bd');
var catInfo=document.getElementById('cat-info');
var zone=document.getElementById('zone');
var ut=document.getElementById('ut');
var pv=document.getElementById('pv');
var icon=document.getElementById('icon');

zone.addEventListener('click',function(){fi.click()});
fi.addEventListener('change',function(){if(this.files[0])h(this.files[0])});

function h(f){
  if(!f)return;
  sf=f;
  sb.disabled=false;
  ut.textContent=f.name;
  icon.style.display='none';
  var r=new FileReader();
  r.onload=function(e){pv.src=e.target.result;pv.style.display='block'};
  r.readAsDataURL(f);
}

zone.addEventListener('dragover',function(e){e.preventDefault();zone.classList.add('dragover')});
zone.addEventListener('dragleave',function(){zone.classList.remove('dragover')});
zone.addEventListener('drop',function(e){
  e.preventDefault();zone.classList.remove('dragover');
  if(e.dataTransfer.files[0])h(e.dataTransfer.files[0]);
});

async function s(){
  if(!sf)return;
  sb.disabled=true;
  sb.textContent='搜索中...';
  bd.textContent='';
  res.innerHTML='<div class="loading"><div class="spinner"></div><div>双模型推理中...</div></div>';

  var fd=new FormData();
  fd.append('file',sf);
  if(cat.value)fd.append('category',cat.value);
  fd.append('auto_category',cat.value?'0':'1');

  try{
    var r=await fetch('/api/search',{method:'POST',body:fd});
    var j=await r.json();
    if(j.code!==0){
      var errIcon=j.sharpness!==undefined?'📷':'⚠️';
      var errColor=j.sharpness!==undefined?'#e67e22':'#c00';
      var errMsg=j.msg;
      if(j.sharpness!==undefined){
        errMsg='<b>图片清晰度不足，无法搜索</b><br><small style="color:#999">清晰度得分: '+j.sharpness.toFixed(1)+' | 最低要求: 80<br>请上传更清晰的图片后重试</small>';
      }
      res.innerHTML='<div style="text-align:center;padding:40px;color:'+errColor+'">'+errIcon+' '+errMsg+'</div>';
      return;
    }
    var d=j.data;
    bd.textContent='耗时 '+d.query_time_ms+'ms | 特征提取 '+d.breakdown.feature_extraction_ms+'ms + 检索 '+d.breakdown.search_rerank_ms+'ms'+(d.cached?' (缓存命中)':'');
    // 品类识别结果
    if(d.ocr_raw_texts && d.ocr_raw_texts.length>0){
      var rawTxt=d.ocr_raw_texts.join(' | ');
      if(d.ocr_keywords && d.ocr_keywords.length>0){
        catInfo.textContent='📝 图中识别: '+rawTxt+' → 有效词: ['+d.ocr_keywords.join(', ')+'] → 搜索结果已加分';
      }else{
        catInfo.textContent='📝 图中识别: '+rawTxt+' → 未匹配到库中产品，已忽略';
      }
    }else if(d.predicted_category){
      catInfo.textContent='🔍 识别品类: '+d.predicted_category+' (置信度: '+(d.predict_confidence*100).toFixed(1)+'%) | 搜索范围: 该品类';
    }else if(d.filter_source==='manual'){
      catInfo.textContent='📌 手动限定品类: '+cat.value;
    }else{
      catInfo.textContent='⚠️ 品类不确定，已搜索全库';
    }

    if(!d.results||!d.results.length){
      var reason='';
      var topPct = d.top_score ? (d.top_score*100).toFixed(1) : '0';
      var thresholdPct = d.min_threshold ? (d.min_threshold*100).toFixed(0) : '65';
      if(topPct <= 0){
        reason = '视觉搜索未找到任何候选';
      }else if(parseFloat(topPct) < parseFloat(thresholdPct)){
        reason = '最高相似度 '+topPct+'%，未达到 '+thresholdPct+'% 门槛';
      }else if(d.consensus_applied){
        reason = '产品类型与库中已有品类不一致，已自动过滤';
      }else{
        reason = '视觉相似度过低或品类不匹配';
      }
      res.innerHTML='<div style="text-align:center;padding:40px;color:#999">'
        +'未找到匹配商品'
        +'<br><small style="color:#bbb">'+reason+'</small>'
        +'<br><small style="color:#ccc;font-size:11px">库中可能没有相同品类的商品</small>'
        +'</div>';
      return;
    }

    // 共识过滤提示
    if(d.consensus_applied && d.consensus_word){
      catInfo.textContent+=' | 🎯 产品共识: "'+d.consensus_word+'" ('+d.results_before_consensus+'→'+d.results.length+'条)';
    }

    var h='<div class="grid">';
    d.results.forEach(function(r){
      var url=r.image_url||r.url||'';
      var score=r.fused_score||0;
      var pct=(score*100).toFixed(1);
      var catName=r.category||'';
      var pname=r.product_name||'';
      var barColor=score>0.7?'#4caf50':score>0.5?'#ff9800':'#f44336';
      h+='<div class="card">';
      h+='<a href="'+url+'" target="_blank"><img src="'+url+'" loading="lazy" onerror="this.parentElement.parentElement.style.display=\\'none\\'"></a>';
      h+='<div class="info">';
      if(r.text_boost && r.text_boost>0){
        h+='<div class="score">'+pct+'% <span style="font-size:10px;color:#0f3460">📝+'+(r.text_boost*100).toFixed(0)+'%</span></div>';
      }else{
        h+='<div class="score">'+pct+'%</div>';
      }
      h+='<div class="cat">'+catName+'</div>';
      if(pname)h+='<div class="cat" style="font-size:10px;color:#666;line-height:1.3;max-height:2.6em;overflow:hidden">'+pname+'</div>';
      h+='<div class="bar"><div style="width:'+pct+'%;background:'+barColor+'"></div></div>';
      h+='</div></div>';
    });
    h+='</div>';
    res.innerHTML=h;
  }catch(e){
    res.innerHTML='<div style="text-align:center;padding:40px;color:#c00">网络错误: '+e.message+'</div>';
  }finally{
    sb.disabled=false;
    sb.textContent='搜索相似图片';
  }
}

sb.addEventListener('click', s);

fetch('/api/stats').then(function(r){return r.json()}).then(function(j){
  if(j.code===0){
    document.getElementById('stat-total').textContent=j.data.db_records;
    document.getElementById('stat-clip').textContent=j.data.clip_dim+'d';
    document.getElementById('stat-resnet').textContent=j.data.resnet_dim+'d';
    // 填充品类下拉
    var cats=j.data.by_category||[];
    cats.forEach(function(c){
      var o=document.createElement('option');
      o.value=c.category;o.textContent=c.category+' ('+c.count+')';
      cat.appendChild(o);
    });
  }
});
</script>
</body>
</html>"""


@app.route('/')
def index():
    _ensure_init()
    return render_template_string(_HTML)


if __name__ == '__main__':
    import signal, atexit

    def _shutdown():
        try:
            e = get_engine()
            if e.total > 0:
                e.save()
                print("[Shutdown] 索引已保存")
        except:
            pass

    atexit.register(_shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: (_shutdown(), exit(0)))
        except:
            pass

    print("=" * 55)
    print("  以图搜图 v2.0 — CLIP + ResNet50 双模型")
    print(f"  地址: http://0.0.0.0:5000")
    print("=" * 55)
    _ensure_init()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
