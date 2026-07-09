"""
AI 大脑可视化仪表盘（AI-Monitor）
主入口：Streamlit 多标签页路由
用法: streamlit run app.py
"""
import streamlit as st
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import init_db, get_unread_alert_count
from src.config import load_config, get_provider_names, get_active_provider, save_config
from src.api_client import APIClientFactory, APIClient
from src.panels.chat_panel import render_chat_panel
from src.panels.star_map import render_star_map
from src.panels.heatmap import render_heatmap
from src.panels.decay_curve import render_decay_curve
from src.panels.alerts_panel import render_alerts_panel
from src.panels.model_diagnostics import render_model_diagnostics
from src.panels.model_verification import render_model_verification


# ===================== 页面配置 =====================
st.set_page_config(
    page_title="AI 监控仪 | AI-Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== 初始化数据库 =====================
init_db()


# ===================== 侧边栏：API 配置 =====================
def render_sidebar():
    """渲染侧边栏：API 配置与供应商切换"""
    with st.sidebar:
        st.title("🧠 AI 监控仪")
        st.caption("记录 · 分析 · 预警")

        st.divider()

        # ---- 添加 / 编辑供应商（前端填写，写回 config.yaml）----
        st.subheader("API 配置")
        with st.expander("➕ 添加 / 编辑供应商", expanded=False):
            with st.form("provider_form", clear_on_submit=False):
                name = st.text_input("供应商名称", placeholder="如：阿里通义千问")
                base_url = st.text_input(
                    "Base URL",
                    placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                api_key = st.text_input("API Key", type="password", placeholder="sk-...")
                model = st.text_input("模型名", placeholder="qwen-plus")
                supports_logprobs = st.checkbox("支持 Logprobs（用于真假检测）", value=True)
                submitted = st.form_submit_button("💾 保存供应商")

                if submitted:
                    if not name or not base_url or not api_key or not model:
                        st.error("请完整填写：名称、Base URL、API Key、模型名")
                    else:
                        cfg = load_config()
                        providers = cfg.get("providers", []) or []
                        # 同名则更新，否则新增
                        for p in providers:
                            if p.get("name") == name:
                                p.update({
                                    "base_url": base_url,
                                    "api_key": api_key,
                                    "model": model,
                                    "supports_logprobs": supports_logprobs,
                                })
                                break
                        else:
                            providers.append({
                                "name": name,
                                "base_url": base_url,
                                "api_key": api_key,
                                "model": model,
                                "supports_logprobs": supports_logprobs,
                            })
                        cfg["providers"] = providers
                        # 若 active_provider 越界（如新增后删过），重置为 0
                        if cfg.get("active_provider", 0) >= len(providers):
                            cfg["active_provider"] = 0
                        save_config(cfg)
                        load_config.clear()  # 清掉缓存，否则 30s 内读不到新值
                        st.success(f"已保存供应商「{name}」")
                        st.rerun()

        st.divider()

        # ---- 供应商选择 ----
        cfg = load_config()
        provider_names = get_provider_names()
        active_idx = cfg.get("active_provider", 0)

        if not provider_names:
            st.warning("还没有任何 API 供应商，请先在上方「➕ 添加 / 编辑供应商」中填写并保存。")
            return None

        selected_name = st.selectbox(
            "选择 API 供应商",
            provider_names,
            index=min(active_idx, len(provider_names) - 1),
        )

        # 切换供应商时，把选择持久化写回 config（仅变化时写，避免每次渲染都写文件）
        if selected_name and provider_names.index(selected_name) != active_idx:
            cfg["active_provider"] = provider_names.index(selected_name)
            save_config(cfg)
            load_config.clear()

        # 显示当前配置摘要
        providers = cfg.get("providers", [])
        for i, p in enumerate(providers):
            if p.get("name") == selected_name:
                with st.expander("当前连接信息", expanded=False):
                    st.caption(f"Base URL: {p.get('base_url', '')}")
                    st.caption(f"Model: {p.get('model', '')}")
                    st.caption(f"Logprobs 支持: {'✅' if p.get('supports_logprobs') else '❌'}")
                    api_key_preview = (p.get("api_key", "") or "")[:12] + "***" if p.get("api_key") else "未设置"
                    st.caption(f"API Key: {api_key_preview}")

                # 删除供应商（两步确认，避免误删）
                if st.button("🗑️ 删除该供应商", key=f"del_{i}"):
                    st.session_state[f"confirm_del_{i}"] = True
                if st.session_state.get(f"confirm_del_{i}", False):
                    st.warning(f"确认删除「{selected_name}」？删除后将在 config.yaml 中移除该项。")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ 确认删除", key=f"del_ok_{i}"):
                            dcfg = load_config()
                            dproviders = dcfg.get("providers", []) or []
                            del_idx = next((j for j, q in enumerate(dproviders) if q.get("name") == selected_name), None)
                            if del_idx is not None:
                                removed = dproviders.pop(del_idx)
                                dcfg["providers"] = dproviders
                                old_active = dcfg.get("active_provider", 0)
                                if del_idx < old_active:
                                    dcfg["active_provider"] = old_active - 1
                                elif del_idx == old_active:
                                    dcfg["active_provider"] = max(0, min(old_active, len(dproviders) - 1))
                                if len(dproviders) == 0:
                                    dcfg["active_provider"] = 0
                                save_config(dcfg)
                                load_config.clear()
                                st.success(f"已删除供应商「{removed.get('name')}」")
                                st.session_state[f"confirm_del_{i}"] = False
                                st.rerun()
                    with c2:
                        if st.button("❌ 取消", key=f"del_cancel_{i}"):
                            st.session_state[f"confirm_del_{i}"] = False
                            st.rerun()

                # 测试连接
                if st.button("测试连接", key=f"test_{i}"):
                    client = APIClientFactory.create_from_config(p)
                    success, msg = client.test_connection()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

                # 创建客户端（使用缓存避免每次渲染重建）
                client = _get_cached_client(
                    provider_name=p.get("name", ""),
                    base_url=p.get("base_url", ""),
                    api_key=p.get("api_key", ""),
                    model=p.get("model", ""),
                )
                st.session_state.api_client = client
                return client

        return None

    return None


# ===================== API 客户端缓存 =====================
@st.cache_resource(show_spinner=False)
def _get_cached_client(provider_name: str, base_url: str, api_key: str, model: str) -> APIClient:
    """缓存 API 客户端，避免每次 render 重建"""
    return APIClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider_name=provider_name,
    )


# ===================== 顶部预警条 =====================
def render_alert_banner():
    """全局顶部预警横幅"""
    unread = get_unread_alert_count()
    if unread > 0:
        st.warning(
            f"🔔 检测到 {unread} 条未读预警！请查看「预警中心」了解详情。",
            icon="🔔",
        )


# ===================== 面板异常隔离 =====================
def _safe_render(label: str, fn, *args, **kwargs):
    """异常隔离渲染单个面板。

    单个面板出错时只显示该面板的错误提示，不会拖垮整个应用；
    但 st.rerun() 内部会抛出 RerunException，该异常必须向上穿透，
    交由 Streamlit 运行器处理，否则 rerun 会被吞掉、面板表现为卡死/报错。
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        if type(e).__name__ == "RerunException":
            raise
        st.error(f"{label}出错：{e}")


# ===================== 主页面 =====================
def main():
    # 渲染侧边栏并获取客户端
    client = render_sidebar()

    # 渲染顶部预警条
    render_alert_banner()

    # 主标题
    st.title("AI 大脑可视化仪表盘")
    st.caption("像心电图仪一样监控 AI 的每一次心跳 —— 记录行为、画出轨迹、揪出幻觉")

    # 标签页
    tabs = st.tabs([
        "💬 对话",
        "🌟 记忆星空图",
        "🔥 注意力热力图",
        "📉 记忆衰减曲线",
        "🧠 模型诊断",
        "� 模型验证",
        "�🚨 预警中心",
    ])

    # 对话标签
    with tabs[0]:
        if client is None:
            st.warning("请在侧边栏配置并选择 API 供应商后，即可开始对话。")
            st.info(
                "👈 在左侧边栏「➕ 添加 / 编辑供应商」中填写 API 信息并保存，即可开始对话。\n\n"
                "常用 Base URL：\n"
                "- OpenAI: https://api.openai.com/v1\n"
                "- DeepSeek: https://api.deepseek.com/v1\n"
                "- 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1\n\n"
                "支持任何兼容 OpenAI 接口的服务。"
            )
        else:
            _safe_render("对话面板", render_chat_panel, client)

    # 星空图
    with tabs[1]:
        _safe_render("记忆星空图", render_star_map)

    # 热力图
    with tabs[2]:
        _safe_render("注意力热力图", render_heatmap)

    # 衰减曲线
    with tabs[3]:
        _safe_render("记忆衰减曲线", render_decay_curve)

    # 模型诊断
    with tabs[4]:
        _safe_render("模型诊断", render_model_diagnostics)

    # 模型验证
    with tabs[5]:
        _safe_render("模型验证", render_model_verification)

    # 预警中心
    with tabs[6]:
        _safe_render("预警中心", render_alerts_panel)

    # 页脚
    st.divider()
    st.caption(
        "AI Brain Visualization Dashboard v1.0 | "
        "Powered by Streamlit + Plotly + SQLite | "
        "记录 AI 的每一次心跳"
    )


if __name__ == "__main__":
    main()
