"""
通用工具函数
"""
import re
from datetime import datetime


def format_time(ts: str) -> str:
    """格式化时间戳为可读格式"""
    if not ts:
        return ""
    try:
        # 尝试多种格式
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.strftime("%m-%d %H:%M")
            except ValueError:
                continue
        return ts[:16]
    except Exception:
        return str(ts)[:16]


def parse_tag_command(text: str) -> tuple[str, str]:
    """
    解析 /tag 命令
    返回 (tag_name, cleaned_text)
    如果无 tag 命令，tag_name 为 None
    """
    pattern = r'^/tag\s+(\S+)'
    match = re.match(pattern, text.strip())
    if match:
        tag = match.group(1)
        cleaned = text[match.end():].strip()
        return tag, cleaned
    return None, text


def clean_text(text: str) -> str:
    """清洗文本：移除多余空白、统一标点"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_tokens(count: int) -> str:
    """格式化 Token 数"""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def sanitize_text(text: str) -> str:
    """移除或替换非法 Unicode 字符，避免 API 请求体和 JSON 序列化失败。"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")


def sanitize_float(val, default=0.0):
    """清洗 NaN/Inf 为安全值，避免 Plotly JSON 序列化报错"""
    import math
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return float(val)


def get_severity_color(severity: str) -> str:
    """获取预警严重程度对应的颜色"""
    colors = {
        "info": "#2196F3",
        "warning": "#FF9800",
        "danger": "#F44336",
        "critical": "#B71C1C",
    }
    return colors.get(severity, "#757575")


def get_severity_icon(severity: str) -> str:
    """获取预警严重程度对应的图标"""
    icons = {
        "info": "ℹ️",
        "warning": "⚠️",
        "danger": "🔴",
        "critical": "🚨",
    }
    return icons.get(severity, "📌")
