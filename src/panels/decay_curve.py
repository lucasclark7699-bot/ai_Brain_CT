"""
记忆衰减曲线面板：Plotly 折线图 + 时间滑块
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.analyzer import calc_memory_decay
from src.config import get_analysis_config
from src.database import get_all_project_tags
from utils.helpers import format_time, sanitize_float


@st.cache_data(ttl=300)
def _cached_calc_decay(project_tag: str, window_size: int):
    """记忆衰减计算（TF-IDF 相似度重计算，缓存避免切 tab 反复算）"""
    return calc_memory_decay(project_tag, window_size=window_size)


def render_decay_curve():
    """渲染记忆衰减曲线"""
    st.header("语义连贯性曲线")
    st.caption("追踪每条对话与历史上下文的语义连贯度。断崖式下跌通常意味着话题大幅切换，而非“记忆衰减”。")

    cfg = get_analysis_config()
    danger_threshold = cfg.get("memory_danger_threshold", 0.3)

    # 项目标签过滤
    all_tags = get_all_project_tags()
    tag_filter = st.selectbox("按项目筛选", ["全部"] + all_tags, key="decay_tag_filter")
    project_tag = "" if tag_filter == "全部" else tag_filter

    # 窗口大小
    col1, col2 = st.columns([1, 3])
    with col1:
        window = st.slider("分析窗口", 10, 100, 50, 5)

    decay_data = _cached_calc_decay(project_tag, window_size=window)

    if not decay_data or len(decay_data) < 2:
        st.info("需要至少 2 条对话数据才能生成衰减曲线。请在聊天面板中多发送几条消息。")
        return

    indices = [d["index"] for d in decay_data]
    accuracies = [sanitize_float(d["accuracy"], 1.0) for d in decay_data]
    timestamps = [format_time(d["timestamp"]) for d in decay_data]

    # 创建图表
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("语义连贯度趋势", "话题切换标记"),
    )

    # 主曲线
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=accuracies,
            mode="lines+markers",
            name="语义连贯度",
            line=dict(color="#42A5F5", width=2.5),
            marker=dict(
                size=8,
                color=accuracies,
                colorscale=[
                    [0, "#F44336"],
                    [0.5, "#FF9800"],
                    [1, "#4CAF50"],
                ],
                showscale=True,
                colorbar=dict(
                    title="连贯度",
                    len=0.5,
                    y=0.8,
                ),
            ),
            hovertemplate="对话 #%{x}<br>准确度: %{y:.3f}<br>时间: %{customdata}<extra></extra>",
            customdata=timestamps,
        ),
        row=1, col=1,
    )

    # 危险阈值线
    fig.add_hline(
        y=danger_threshold,
        line_dash="dash",
        line_color="#F44336",
        annotation_text=f"话题切换阈值 ({danger_threshold})",
        annotation_position="top right",
        row=1, col=1,
    )

    # 危险区间标记（下方子图）
    danger_zones = []
    for d in decay_data:
        if sanitize_float(d["accuracy"], 1.0) < danger_threshold:
            danger_zones.append(1)
        else:
            danger_zones.append(0)

    if any(danger_zones):
        # 用不同颜色标记
        colors = ["#F44336" if z else "#E0E0E0" for z in danger_zones]
        fig.add_trace(
            go.Bar(
                x=indices,
                y=[1] * len(indices),
                marker_color=colors,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )

    # 统计信息
    avg_accuracy = np.mean(accuracies)
    min_accuracy = min(accuracies)
    min_idx = accuracies.index(min_accuracy)
    trend = "上升" if len(accuracies) >= 2 and accuracies[-1] > accuracies[0] else "下降"

    # 布局
    fig.update_layout(
        height=550,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(title_text="对话序号", row=2, col=1)
    fig.update_yaxes(title_text="记忆准确度", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    st.plotly_chart(fig)

    # 统计面板
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均连贯度", f"{avg_accuracy:.3f}")
    with col2:
        st.metric("最低连贯度", f"{min_accuracy:.3f}",
                  delta=f"对话 #{min_idx}", delta_color="off")
    with col3:
        st.metric("趋势", trend, delta_color="normal" if trend == "上升" else "inverse")
    with col4:
        danger_count = sum(1 for d in decay_data if sanitize_float(d["accuracy"], 1.0) < danger_threshold)
        st.metric("话题切换点", danger_count, delta=f"阈值 {danger_threshold}")

    # 时间滑块回放
    st.divider()
    st.subheader("连贯度回放")
    st.caption("拖动滑块，查看各阶段对话与历史上下文的语义连贯度")

    if len(decay_data) > 2:
        replay_idx = st.slider("选择对话序号", 0, len(decay_data) - 1, len(decay_data) - 1)

        if replay_idx < len(decay_data):
            d = decay_data[replay_idx]
            acc = sanitize_float(d["accuracy"], 1.0)
            accuracy_color = (
                "#4CAF50" if acc > 0.7 else
                "#FF9800" if acc > danger_threshold else
                "#F44336"
            )
            st.markdown(
                f"**对话 #{d['index']}** | 时间: {format_time(d['timestamp'])} | "
                f"语义连贯度: <span style='color:{accuracy_color};font-weight:bold'>{acc:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"摘要: {d['preview']}...")
    else:
        st.info("数据点不足，无法启用回放功能。")
