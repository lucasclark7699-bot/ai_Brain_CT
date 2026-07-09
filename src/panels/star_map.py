"""
记忆星空图面板：Plotly 力导向网络图
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

from src.analyzer import build_keyword_graph
from src.database import get_conversations_by_keyword, get_all_project_tags
from utils.helpers import format_time, sanitize_float


@st.cache_data(ttl=300)
def _cached_build_graph_and_layout(limit: int, project_tag: str):
    """构建关键词关联图 + 力导向布局（重计算，缓存避免切 tab 反复算）"""
    graph_data = build_keyword_graph(limit=limit, project_tag=project_tag)
    if not graph_data.get("nodes"):
        return graph_data, {}
    positions = _force_directed_layout(graph_data["nodes"], graph_data["edges"])
    return graph_data, positions


def render_star_map():
    """渲染记忆星空图（关键词关联网络）"""
    st.header("记忆星空图")
    st.caption("关键词关联网络 —— 一眼看出 AI 把哪些概念关联在一起")

    # 项目标签过滤
    all_tags = get_all_project_tags()
    tag_filter = st.selectbox("按项目筛选", ["全部"] + all_tags, key="star_tag_filter")
    project_tag = "" if tag_filter == "全部" else tag_filter

    if st.button("刷新星空图"):
        _cached_build_graph_and_layout.clear()
        st.rerun()

    # 构建图数据（带缓存，避免切 tab 反复重算力导向布局）
    graph_data, positions = _cached_build_graph_and_layout(limit=50, project_tag=project_tag)

    if not graph_data["nodes"]:
        st.info("暂无数据。请先在聊天面板中发送几条消息，系统会自动提取关键词并构建关联网络。")
        return

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # 构建 Plotly 网络图
    fig = go.Figure()

    # 添加连线
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src in positions and tgt in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            width = max(0.3, min(3, sanitize_float(edge["weight"], 1) * 0.6))
            opacity = min(1.0, sanitize_float(edge["weight"], 1) * 0.15)
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=f"rgba(100, 150, 200, {opacity})"),
                hoverinfo="text",
                text=f"{src} ⟷ {tgt}<br>共现次数: {edge['weight']}",
                showlegend=False,
            ))

    # 添加节点
    for node in nodes:
        name = node["id"]
        if name in positions:
            x, y = positions[name]
            size = node["size"]
            freq = node["freq"]

            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=px.colors.qualitative.Set2[
                        hash(name) % len(px.colors.qualitative.Set2)
                    ],
                    line=dict(width=1, color="rgba(255,255,255,0.8)"),
                ),
                text=name,
                textposition="top center",
                textfont=dict(size=max(8, size * 0.5)),
                hoverinfo="text",
                hovertext=f"关键词: {name}<br>出现次数: {freq}",
                showlegend=False,
            ))

    fig.update_layout(
        title="关键词关联网络（力导向布局）",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig)

    # 点击节点查看相关对话
    st.divider()
    st.subheader("查看关键词相关对话")
    keyword_list = [n["id"] for n in nodes[:20]]
    selected_kw = st.selectbox("选择关键词查看相关对话", ["-- 选择 --"] + keyword_list)

    if selected_kw and selected_kw != "-- 选择 --":
        conversations = get_conversations_by_keyword(selected_kw, limit=20)
        if conversations:
            st.caption(f"共找到 {len(conversations)} 条与「{selected_kw}」相关的对话：")
            for conv in conversations:
                with st.container():
                    st.markdown(
                        f"**{format_time(conv.get('timestamp', ''))}** "
                        f"`{conv.get('project_tag', '') or '无标签'}`"
                    )
                    st.caption(f"Q: {(conv.get('user_input', '') or '')[:100]}")
                    st.caption(f"A: {(conv.get('ai_output', '') or '')[:100]}")
                st.divider()
        else:
            st.info(f"未找到与「{selected_kw}」相关的对话。")


def _force_directed_layout(nodes: list, edges: list, iterations: int = 100) -> dict:
    """
    简易力导向布局算法
    返回 {node_id: (x, y), ...}
    """
    n = len(nodes)
    if n == 0:
        return {}

    # 初始化随机位置
    np.random.seed(42)
    positions = {}
    for node in nodes:
        positions[node["id"]] = np.random.rand(2) * 2 - 1  # [-1, 1]

    # 构建邻接表
    adj = {node["id"]: [] for node in nodes}
    for edge in edges:
        s, t = edge["source"], edge["target"]
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)

    # 力导向迭代
    area = n * 2
    k = math.sqrt(area / n)  # 理想距离
    temperature = 1.0
    cooling = 0.95

    node_ids = [node["id"] for node in nodes]
    pos_array = np.array([positions[nid] for nid in node_ids])

    for _ in range(iterations):
        displacement = np.zeros((n, 2))

        # 计算斥力（所有节点对之间）
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos_array[i] - pos_array[j]
                dist = np.linalg.norm(delta)
                if dist < 0.01:
                    dist = 0.01
                repulsion = (k * k) / dist
                disp = (delta / dist) * repulsion
                displacement[i] += disp
                displacement[j] -= disp

        # 计算引力（相连节点之间）
        for i, nid in enumerate(node_ids):
            for neighbor in adj[nid]:
                j = node_ids.index(neighbor)
                delta = pos_array[j] - pos_array[i]
                dist = np.linalg.norm(delta)
                if dist < 0.01:
                    dist = 0.01
                attraction = (dist * dist) / k
                disp = (delta / dist) * attraction
                displacement[i] += disp
                displacement[j] -= disp

        # 更新位置
        for i in range(n):
            disp_norm = np.linalg.norm(displacement[i])
            if disp_norm > 0:
                pos_array[i] += (displacement[i] / disp_norm) * min(disp_norm, temperature)

        # 限制在画布内
        pos_array = np.clip(pos_array, -2, 2)
        temperature *= cooling

    # 转换回字典
    for i, nid in enumerate(node_ids):
        positions[nid] = (float(pos_array[i][0]), float(pos_array[i][1]))

    return positions
