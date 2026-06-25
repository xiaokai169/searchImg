"""
SQLite 数据库层 — 只存元数据 + OBS URL，不存本地文件
向量由 FAISS 索引管理
"""
import sqlite3
import threading
from contextlib import contextmanager
from config import DATABASE_PATH

_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    if not hasattr(_local, 'conn') or _local.conn is None:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return _local.conn


@contextmanager
def get_db():
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_database():
    """建表：image_url 存 OBS 原始链接"""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id    INTEGER NOT NULL UNIQUE,
            image_url   TEXT    NOT NULL,
            category    TEXT    NOT NULL DEFAULT '其他',
            product_name TEXT   NOT NULL DEFAULT '',
            product_name_cn TEXT NOT NULL DEFAULT '',
            product_id  TEXT    NOT NULL DEFAULT '',
            keywords_cn TEXT    NOT NULL DEFAULT '',
            keywords_en TEXT    NOT NULL DEFAULT '',
            file_size   INTEGER NOT NULL DEFAULT 0,
            width       INTEGER DEFAULT 0,
            height      INTEGER DEFAULT 0,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_images_category ON images(category);
        CREATE INDEX IF NOT EXISTS idx_images_faiss_id ON images(faiss_id);
    """)
    # 兼容旧库：逐步添加缺失的列
    for col, col_type in [
        ('product_name', 'TEXT NOT NULL DEFAULT \'\''),
        ('product_name_cn', 'TEXT NOT NULL DEFAULT \'\''),
        ('product_id', 'TEXT NOT NULL DEFAULT \'\''),
        ('keywords_cn', 'TEXT NOT NULL DEFAULT \'\''),
        ('keywords_en', 'TEXT NOT NULL DEFAULT \'\''),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM images LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE images ADD COLUMN {col} {col_type}")
            conn.commit()
            print(f"[DB] 已添加 {col} 列")
    conn.commit()
    print("[DB] 数据库初始化完成")


def insert_image(faiss_id: int, image_url: str, category: str = '其他',
                 product_name: str = '', product_name_cn: str = '',
                 product_id: str = '', keywords_cn: str = '',
                 keywords_en: str = '', file_size: int = 0,
                 width: int = 0, height: int = 0) -> int:
    with get_db() as db:
        cur = db.execute(
            """INSERT INTO images (faiss_id, image_url, category, product_name,
                                   product_name_cn, product_id, keywords_cn, keywords_en,
                                   file_size, width, height)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (faiss_id, image_url, category, product_name or '',
             product_name_cn or '', str(product_id or ''), keywords_cn or '',
             keywords_en or '', file_size, width, height)
        )
        return cur.lastrowid


def get_image_by_faiss_id(faiss_id: int) -> dict | None:
    conn = _get_connection()
    row = conn.execute("SELECT * FROM images WHERE faiss_id = ?", (faiss_id,)).fetchone()
    return dict(row) if row else None


def get_all_images(category: str = '') -> list[dict]:
    conn = _get_connection()
    if category:
        rows = conn.execute("SELECT * FROM images WHERE category = ? ORDER BY id", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_total_count(category: str = '') -> int:
    conn = _get_connection()
    if category:
        row = conn.execute("SELECT COUNT(*) as cnt FROM images WHERE category = ?", (category,)).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
    return row['cnt']


def get_category_stats() -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT category, COUNT(*) as count FROM images GROUP BY category ORDER BY count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_image(db_id: int) -> bool:
    with get_db() as db:
        cur = db.execute("DELETE FROM images WHERE id = ?", (db_id,))
        return cur.rowcount > 0


def image_url_exists(url: str) -> bool:
    conn = _get_connection()
    row = conn.execute("SELECT 1 FROM images WHERE image_url = ?", (url,)).fetchone()
    return row is not None
