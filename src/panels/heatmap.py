"""
注意力热力图面板：基于 logprobs 的词级别关注度可视化
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.database import get_conversations
from src.analyzer import calc_attention_scores
from utils.helpers import format_time, sanitize_float
import json


@st.cache_data(ttl=300)
def _cached_get_conversations(limit: int):
    """读取对话记录（缓存，避免切 tab / 重渲染反复查库）"""
    return get_conversations(limit=limit)


def render_heatmap():
    """渲染注意力热力图"""
    st.header("注意力热力图")
    st.caption("词级别关注度监控 —— 看看 AI 到底在关注哪些词")

    # 获取对话列表
    conversations = _cached_get_conversations(limit=50)
    if not conversations:
        st.info("暂无对话数据。请先在聊天面板中发送消息。")
        return

    # 选择对话
    conv_options = [
        f"#{c['id']} {format_time(c.get('timestamp', ''))} - {(c.get('user_input', '') or '')[:40]}"
        for c in conversations
    ]
    selected_idx = st.selectbox(
        "选择一条对话查看注意力分布",
        range(len(conv_options)),
        format_func=lambda i: conv_options[i],
        key="heatmap_conv_select"
    )

    if selected_idx is not None:
        conv = conversations[selected_idx]
        user_input = conv.get("user_input", "") or ""

        st.subheader("用户输入原文")
        st.markdown(f"> {user_input}")

        # 尝试解析 logprobs
        logprobs_data = None
        try:
            raw = conv.get("logprobs_data", "{}")
            if isinstance(raw, str):
                logprobs_data = json.loads(raw)
            elif isinstance(raw, dict):
                logprobs_data = raw
        except (json.JSONDecodeError, TypeError):
            pass

        # 计算注意力分数
        scores = calc_attention_scores(user_input, logprobs_data)

        if scores:
            _render_word_heatmap(scores)
            _render_score_table(scores)
        else:
            st.warning("无法计算该对话的注意力分数（文本为空或格式异常）。")


def _render_word_heatmap(scores: list[dict]):
    """渲染词级别热力图"""
    words = [s["word"] for s in scores]
    score_values = [sanitize_float(s["score"], 0.0) for s in scores]

    if not words:
        return

    # 使用 Plotly 热力图
    # 将词排列为矩阵形式（自动换行）
    cols_per_row = 10
    rows = []
    row_labels = []
    col_labels = []
    z = []

    for i in range(0, len(words), cols_per_row):
        row_words = words[i:i + cols_per_row]
        row_scores = score_values[i:i + cols_per_row]
        # 补齐到 cols_per_row
        while len(row_words) < cols_per_row:
            row_words.append("")
            row_scores.append(0)
        rows.append(row_scores)
        col_labels = [f"词{j + 1}" for j in range(cols_per_row)]

    # 转置使词语按列排列
    z = np.array(rows)
    row_labels = [f"行{i + 1}" for i in range(len(rows))]

    # 创建热力图（使用标注显示词语本身）
    annotations = []
    for i, row in enumerate(rows):
        for j, (word, score) in enumerate(zip(words[i * cols_per_row: (i + 1) * cols_per_row],
                                               score_values[i * cols_per_row: (i + 1) * cols_per_row])):
            if word:
                annotations.append(dict(
                    x=j, y=i,
                    text=f"<b>{word}</b><br>{score:.2f}",
                    showarrow=False,
                    font=dict(
                        size=max(10, min(18, 12 + score * 8)),
                        color="white" if score > 0.5 else "#333"
                    )
                ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        colorscale=[
            [0, "#E3F2FD"],      # 浅蓝：低关注
            [0.25, "#90CAF9"],
            [0.5, "#42A5F5"],    # 中蓝：中等关注
            [0.75, "#EF5350"],   # 浅红：较高关注
            [1, "#B71C1C"],      # 深红：高关注
        ],
        hoverongaps=False,
        hovertemplate="词语: %{text}<br>注意力: %{z:.3f}<extra></extra>",
        text=[[words[i * cols_per_row + j] if i * cols_per_row + j < len(words) else ""
               for j in range(cols_per_row)] for i in range(len(rows))],
    ))

    fig.update_layout(
        title="AI 注意力分布（颜色越深 = AI 越关注该词）",
        height=max(200, 50 * len(rows)),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig)

    # 图例说明
    cols_desc = st.columns(5)
    desc_items = [
        ("#E3F2FD", "低关注 (0~0.2)"),
        ("#90CAF9", "一般 (0.2~0.4)"),
        ("#42A5F5", "中等 (0.4~0.6)"),
        ("#EF5350", "较高 (0.6~0.8)"),
        ("#B71C1C", "高关注 (0.8~1.0)"),
    ]
    for i, (color, label) in enumerate(desc_items):
        with cols_desc[i]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:16px;height:16px;background:{color};border-radius:3px"></div>'
                f'<span style="font-size:12px">{label}</span></div>',
                unsafe_allow_html=True,
            )


def _render_score_table(scores: list[dict]):
    """渲染注意力分数详情表格"""
    with st.expander("查看详细注意力分数", expanded=False):
        # 按分数降序排列
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

        # 构建表格数据
        table_data = []
        for s in sorted_scores:
            # 根据分数给颜色
            if s["score"] > 0.7:
                color = "#B71C1C"
            elif s["score"] > 0.5:
                color = "#EF5350"
            elif s["score"] > 0.3:
                color = "#42A5F5"
            else:
                color = "#90CAF9"

            table_data.append({
                "词语": s["word"],
                "注意力分数": f"{s['score']:.4f}",
                "关注程度": (
                    "🔥 极高" if s["score"] > 0.8 else
                    "🔴 高" if s["score"] > 0.6 else
                    "🟡 中" if s["score"] > 0.4 else
                    "🔵 低" if s["score"] > 0.2 else
                    "⚪ 极低"
                ),
            })

        st.dataframe(
            table_data,
            hide_index=True,
            height=400,
        )
