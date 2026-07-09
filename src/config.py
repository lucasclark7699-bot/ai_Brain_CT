"""
配置加载模块：读取 config.yaml，管理 API 供应商配置
"""
import os
import tempfile
import streamlit as st
import yaml


CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")


def _default_config() -> dict:
    """返回一份安全的默认配置（config 缺失或损坏时使用）"""
    return {"providers": [], "active_provider": 0, "analysis": {}, "alert": {}}


@st.cache_data(ttl=30, show_spinner=False)
def load_config() -> dict:
    """加载 YAML 配置文件（缓存 30 秒）。

    对损坏/半截写入的 YAML 做容错：解析失败时不抛异常，
    回退到默认配置并打印警告，避免整个应用被一个坏配置文件拖崩。
    """
    if not os.path.exists(CONFIG_PATH):
        return _default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or _default_config()
    except (yaml.YAMLError, OSError) as e:
        st.warning(f"⚠️ config.yaml 解析失败，已临时使用默认配置：{e}")
        return _default_config()


def save_config(config: dict):
    """原子保存配置到 YAML 文件。

    先写临时文件再 os.replace 原子替换，避免并发读取（如 Streamlit
    文件监听器触发的自动重载）读到半截内容导致 YAMLError 崩溃。
    """
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    dir_name = os.path.dirname(CONFIG_PATH)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def get_active_provider() -> dict:
    """获取当前激活的 API 供应商配置"""
    cfg = load_config()
    providers = cfg.get("providers", [])
    idx = cfg.get("active_provider", 0)
    if 0 <= idx < len(providers):
        return providers[idx]
    return {}


def get_provider_names() -> list[str]:
    """获取所有供应商名称列表"""
    cfg = load_config()
    return [p.get("name", f"Provider {i}") for i, p in enumerate(cfg.get("providers", []))]


def get_analysis_config() -> dict:
    """获取分析引擎配置"""
    cfg = load_config()
    return cfg.get("analysis", {})


def get_alert_config() -> dict:
    """获取预警配置"""
    cfg = load_config()
    return cfg.get("alert", {})
