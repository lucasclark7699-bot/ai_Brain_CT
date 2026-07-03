"""
SQLite 数据库层：建表、CRUD 操作、WAL 模式
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "ai_monitor.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_tag TEXT DEFAULT '',
            user_input TEXT NOT NULL,
            ai_output TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_tokens INTEGER DEFAULT 0,
            model_name TEXT DEFAULT '',
            request_model TEXT DEFAULT '',
            provider_name TEXT DEFAULT '',
            response_time_ms INTEGER DEFAULT 0,
            logprobs_data TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            alert_type TEXT NOT NULL,
            description TEXT NOT NULL,
            severity TEXT DEFAULT 'warning',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            acknowledged INTEGER DEFAULT 0,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS api_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            model TEXT NOT NULL,
            supports_logprobs INTEGER DEFAULT 1,
            is_active INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_api_configs_active ON api_configs(is_active);
        CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_conv_project ON conversations(project_tag);
        CREATE INDEX IF NOT EXISTS idx_kw_conversation ON keywords(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_kw_keyword ON keywords(keyword);
        CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts(acknowledged);
    """)
    conn.commit()
    _ensure_conversation_columns(conn)
    conn.close()


def _ensure_conversation_columns(conn: sqlite3.Connection):
    expected_columns = {
        'request_model': "TEXT DEFAULT ''",
        'provider_name': "TEXT DEFAULT ''",
        'response_time_ms': "INTEGER DEFAULT 0",
    }
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(conversations);").fetchall()}
    for column, definition in expected_columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} {definition}")


# ===================== Conversations CRUD =====================

def save_conversation(project_tag: str, user_input: str, ai_output: str,
                      total_tokens: int = 0, model_name: str = "",
                      request_model: str = "", provider_name: str = "",
                      response_time_ms: int = 0, logprobs_data: dict = None) -> int:
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO conversations (project_tag, user_input, ai_output,
           total_tokens, model_name, request_model, provider_name,
           response_time_ms, logprobs_data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_tag, user_input, ai_output, total_tokens, model_name,
         request_model, provider_name, response_time_ms,
         json.dumps(logprobs_data or {}, ensure_ascii=False))
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def get_conversations(project_tag: str = "", limit: int = 100,
                      offset: int = 0) -> list[dict]:
    conn = get_connection()
    if project_tag:
        rows = conn.execute(
            """SELECT * FROM conversations WHERE project_tag = ?
               ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            (project_tag, limit, offset)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversations_ordered(project_tag: str = "", limit: int = 100) -> list[dict]:
    """按时间正序获取对话（用于衰减曲线）"""
    conn = get_connection()
    if project_tag:
        rows = conn.execute(
            """SELECT * FROM conversations WHERE project_tag = ?
               ORDER BY timestamp ASC LIMIT ?""",
            (project_tag, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp ASC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation_by_id(conv_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_conversations_by_keyword(keyword: str, limit: int = 50) -> list[dict]:
    """根据关键词查找相关对话"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT DISTINCT c.* FROM conversations c
           INNER JOIN keywords k ON c.id = k.conversation_id
           WHERE k.keyword = ?
           ORDER BY c.timestamp DESC LIMIT ?""",
        (keyword, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_project_tags() -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT project_tag FROM conversations WHERE project_tag != '' ORDER BY project_tag"
    ).fetchall()
    conn.close()
    return [r["project_tag"] for r in rows]


def get_conversation_count(project_tag: str = "") -> int:
    conn = get_connection()
    if project_tag:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE project_tag = ?",
            (project_tag,)
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
    conn.close()
    return row["cnt"] if row else 0


# ===================== Keywords CRUD =====================

def save_keywords(conversation_id: int, keywords: list[tuple[str, float]]):
    """保存关键词列表，格式 [(keyword, weight), ...] """
    conn = get_connection()
    conn.executemany(
        "INSERT INTO keywords (conversation_id, keyword, weight) VALUES (?, ?, ?)",
        [(conversation_id, kw, w) for kw, w in keywords]
    )
    conn.commit()
    conn.close()


def get_all_keywords(project_tag: str = "") -> list[dict]:
    """获取所有关键词及其统计信息"""
    conn = get_connection()
    if project_tag:
        rows = conn.execute(
            """SELECT k.keyword, COUNT(*) as freq, AVG(k.weight) as avg_weight
               FROM keywords k
               JOIN conversations c ON c.id = k.conversation_id
               WHERE c.project_tag = ?
               GROUP BY k.keyword ORDER BY freq DESC""",
            (project_tag,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT keyword, COUNT(*) as freq, AVG(weight) as avg_weight
               FROM keywords GROUP BY keyword ORDER BY freq DESC"""
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_keyword_pairs(min_cooccur: int = 1, project_tag: str = "") -> list[dict]:
    """获取关键词共现对（用于星空图连线）"""
    conn = get_connection()
    if project_tag:
        rows = conn.execute(
            """SELECT k1.keyword as source, k2.keyword as target,
                      COUNT(*) as cooccur
               FROM keywords k1
               JOIN keywords k2 ON k1.conversation_id = k2.conversation_id
               JOIN conversations c ON c.id = k1.conversation_id
               WHERE k1.keyword < k2.keyword AND c.project_tag = ?
               GROUP BY k1.keyword, k2.keyword
               HAVING cooccur >= ?
               ORDER BY cooccur DESC
               LIMIT 200""",
            (project_tag, min_cooccur)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT k1.keyword as source, k2.keyword as target,
                      COUNT(*) as cooccur
               FROM keywords k1
               JOIN keywords k2 ON k1.conversation_id = k2.conversation_id
               WHERE k1.keyword < k2.keyword
               GROUP BY k1.keyword, k2.keyword
               HAVING cooccur >= ?
               ORDER BY cooccur DESC
               LIMIT 200""",
            (min_cooccur,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ===================== Alerts CRUD =====================

def save_alert(conversation_id: Optional[int], alert_type: str,
               description: str, severity: str = "warning") -> int:
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO alerts (conversation_id, alert_type, description, severity)
           VALUES (?, ?, ?, ?)""",
        (conversation_id, alert_type, description, severity)
    )
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid


def get_alerts(acknowledged: int = -1, limit: int = 50) -> list[dict]:
    conn = get_connection()
    if acknowledged >= 0:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE acknowledged = ? ORDER BY created_at DESC LIMIT ?",
            (acknowledged, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int):
    conn = get_connection()
    conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def get_unread_alert_count() -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0"
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0
