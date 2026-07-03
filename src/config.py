"""
配置加载模块：读取 config.yaml，管理 API 供应商配置
"""
import os
import streamlit as st
import yaml


CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")


@st.cache_data(ttl=30, show_spinner=False)
def load_config() -> dict:
    """加载 YAML 配置文件（缓存 30 秒）"""
    if not os.path.exists(CONFIG_PATH):
        return {"providers": [], "active_provider": 0, "analysis": {}, "alert": {}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict):
    """保存配置到 YAML 文件"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


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
