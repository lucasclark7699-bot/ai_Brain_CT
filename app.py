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
from src.config import load_config, get_provider_names, get_active_provider
from src.api_client import APIClientFactory
from src.panels.chat_panel import render_chat_panel
from src.panels.star_map import render_star_map
from src.panels.heatmap import render_heatmap
from src.panels.decay_curve import render_decay_curve
from src.panels.alerts_panel import render_alerts_panel


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

        # 供应商选择
        st.subheader("API 配置")
        cfg = load_config()
        provider_names = get_provider_names()
        active_idx = cfg.get("active_provider", 0)

        if not provider_names:
            st.warning("未配置 API 供应商，请在 config.yaml 中配置。")
            return None

        selected_name = st.selectbox(
            "选择 API 供应商",
            provider_names,
            index=active_idx,
        )

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

                # 测试连接
                if st.button("测试连接", key=f"test_{i}"):
                    client = APIClientFactory.create_from_config(p)
                    success, msg = client.test_connection()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

                # 创建客户端
                return APIClientFactory.create_from_config(p)

        return None

    return None


# ===================== 顶部预警条 =====================
def render_alert_banner():
    """全局顶部预警横幅"""
    unread = get_unread_alert_count()
    if unread > 0:
        st.warning(
            f"🔔 检测到 {unread} 条未读预警！请查看「预警中心」了解详情。",
            icon="🔔",
        )


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
        "🚨 预警中心",
    ])

    # 对话标签
    with tabs[0]:
        if client is None:
            st.warning("请在侧边栏配置并选择 API 供应商后，即可开始对话。")
            st.info(
                "编辑 `config.yaml` 文件，配置你的 API 信息：\n"
                "- OpenAI: https://api.openai.com/v1\n"
                "- DeepSeek: https://api.deepseek.com/v1\n"
                "- 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1\n\n"
                "也可使用任何兼容 OpenAI 接口的服务。"
            )
        else:
            render_chat_panel(client)

    # 星空图
    with tabs[1]:
        render_star_map()

    # 热力图
    with tabs[2]:
        render_heatmap()

    # 衰减曲线
    with tabs[3]:
        render_decay_curve()

    # 预警中心
    with tabs[4]:
        render_alerts_panel()

    # 页脚
    st.divider()
    st.caption(
        "AI Brain Visualization Dashboard v1.0 | "
        "Powered by Streamlit + Plotly + SQLite | "
        "记录 AI 的每一次心跳"
    )


if __name__ == "__main__":
    main()
