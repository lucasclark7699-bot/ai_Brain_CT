"""
模型诊断面板：展示模型来源、响应时延、模型可信度与异常行为指标
"""
import streamlit as st
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.database import get_conversations
from src.analyzer import detect_model_symptoms
from utils.helpers import format_time, sanitize_float


@st.cache_data(ttl=300)
def _cached_get_conversations(limit: int):
    """读取对话记录（缓存，避免切 tab / 重渲染反复查库）"""
    return get_conversations(limit=limit)


@st.cache_data(ttl=300)
def _cached_detect_symptoms(recent_n: int):
    """模型症状分析（TF-IDF 相似度 O(n²) 重计算，缓存避免反复算）"""
    return detect_model_symptoms(recent_n=recent_n)


def _detect_origin_from_record(conv: dict) -> tuple[str, str]:
    request_model = (conv.get("request_model") or "").lower()
    provider_name = (conv.get("provider_name") or "").lower()
    base_desc = []

    official_prefixes = [
        "gpt-", "gpt4", "gpt4o", "gpt-4", "gpt-3.5", "text-davinci", "text-curie"
    ]
    is_official = "openai" in provider_name
    if is_official:
        if any(request_model.startswith(prefix) for prefix in official_prefixes):
            return "官方模型", "当前供应商地址携带 OpenAI 特征，模型名也符合官方命名。"
        return "疑似非官方模型", "接口来源看起来是官方，但模型名不符合常见官方模型名称，可能是兼容接口或改名模型。"

    if any(request_model.startswith(prefix) for prefix in official_prefixes):
        return "疑似套壳模型", "模型名看起来像官方模型，但当前供应商不是 OpenAI 官方域名。"

    return "非官方模型", "当前模型与供应商来源都不符合官方命名/域名特征。"


def render_model_diagnostics():
    st.header("模型诊断")
    st.caption("基于历史对话分析模型来源、响应性能和潜在异常行为。")

    conversations = _cached_get_conversations(limit=200)
    if not conversations:
        st.info("暂无对话数据。请先在聊天面板中发送消息。")
        return

    total = len(conversations)
    avg_resp = sanitize_float(sum((c.get("response_time_ms") or 0) for c in conversations) / total, 0)
    avg_tokens = sanitize_float(sum((c.get("total_tokens") or 0) for c in conversations) / total, 0)
    provider_counter = Counter((c.get("provider_name") or "Unknown") for c in conversations)
    model_counter = Counter((c.get("request_model") or c.get("model_name") or "Unknown") for c in conversations)

    origin_labels = [
        _detect_origin_from_record(c)[0] for c in conversations
    ]
    origin_counter = Counter(origin_labels)
    symptoms = _cached_detect_symptoms(recent_n=50)
    suspicious_count = origin_counter.get("疑似套壳模型", 0) + origin_counter.get("疑似非官方模型", 0)
    reliability = "高" if suspicious_count == 0 else ("中" if suspicious_count <= 3 else "低")
    reliability_color = "#4CAF50" if reliability == "高" else ("#FF9800" if reliability == "中" else "#F44336")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("对话总数", total)
    col2.metric("平均响应时延", f"{avg_resp:.0f} ms")
    col3.metric("平均 Token", f"{avg_tokens:.1f}")
    col4.markdown(
        f"<div style='padding:10px;border-radius:10px;background:{reliability_color};color:#fff;'>"
        f"<strong>模型可信度：{reliability}</strong>" 
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("模型症状概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("幻觉次数", symptoms["hallucination_count"])
    c2.metric("上下文断裂次数", symptoms["context_break_count"])
    c3.metric("记忆错乱次数", symptoms["memory_confusion_count"])
    c4.metric("疲惫评分", f"{symptoms['fatigue_score']:.2f}")

    st.caption(
        f"最近 {symptoms['total']} 条对话平均相似度 {symptoms['avg_similarity']:.2f}，"
        f"平均回答长度 {symptoms['avg_response_length']:.0f} 字。"
    )

    if symptoms["hallucination_count"] > 0 or symptoms["context_break_count"] > 0:
        st.warning(
            "检测到模型可能存在幻觉或上下文断裂，可在“最近诊断记录”中进一步复查具体对话。"
        )
    elif symptoms["fatigue_score"] > 0.4:
        st.info("检测到模型疲惫倾向，最近生成更短或不稳定的回复。")
    else:
        st.success("当前模型症状较轻，整体行为较为稳定。")

    symptom_trend = symptoms.get("symptom_trend", [])
    if symptom_trend:
        st.divider()
        st.subheader("症状趋势图")

        indices = [item["index"] for item in symptom_trend]
        timestamps = [format_time(item["timestamp"]) for item in symptom_trend]
        hallucinations = [1 if item["hallucination"] else 0 for item in symptom_trend]
        context_breaks = [1 if item["context_break"] else 0 for item in symptom_trend]
        memory_confusions = [1 if item["memory_confusion"] else 0 for item in symptom_trend]
        fatigue_flags = [item["fatigue_flag"] for item in symptom_trend]

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("幻觉趋势", "上下文断裂趋势", "记忆错乱趋势", "疲惫趋势"),
        )

        fig.add_trace(
            go.Scatter(
                x=indices,
                y=hallucinations,
                mode="markers+lines",
                name="幻觉",
                line=dict(color="#E53935", width=2),
                marker=dict(size=10, symbol="x"),
                hovertemplate="#%{x} 幻觉事件: %{y}<br>时间: %{customdata}<extra></extra>",
                customdata=timestamps,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=indices,
                y=context_breaks,
                mode="markers+lines",
                name="上下文断裂",
                line=dict(color="#FB8C00", width=2),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="#%{x} 上下文断裂: %{y}<br>时间: %{customdata}<extra></extra>",
                customdata=timestamps,
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=indices,
                y=memory_confusions,
                mode="markers+lines",
                name="记忆错乱",
                line=dict(color="#8E24AA", width=2),
                marker=dict(size=10, symbol="circle"),
                hovertemplate="#%{x} 记忆错乱: %{y}<br>时间: %{customdata}<extra></extra>",
                customdata=timestamps,
            ),
            row=3, col=1,
        )

        fig.add_trace(
            go.Bar(
                x=indices,
                y=fatigue_flags,
                name="疲惫事件",
                marker_color="#6A1B9A",
                opacity=0.6,
                hovertemplate="#%{x} 疲惫事件: %{y}<br>时间: %{customdata}<extra></extra>",
                customdata=timestamps,
            ),
            row=4, col=1,
        )

        fig.update_layout(
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )

        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["无", "有"], row=1, col=1)
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["无", "有"], row=2, col=1)
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["无", "有"], row=3, col=1)
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["无", "有"], row=4, col=1)
        fig.update_xaxes(title_text="对话序号", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    cols = st.columns(3)

    with cols[0]:
        st.subheader("来源分布")
        for provider, count in provider_counter.items():
            st.metric(provider, count)

    with cols[1]:
        st.subheader("模型可信度分析")
        for label, count in origin_counter.items():
            st.metric(label, count)

    with cols[2]:
        st.subheader("模型名一致性")
        mismatch_count = sum(
            1 for c in conversations
            if (c.get("request_model") or "").lower() != (c.get("model_name") or "").lower()
        )
        st.metric("请求模型与返回模型不一致", mismatch_count)

    st.divider()
    st.subheader("最近 10 条模型诊断记录")

    recent = conversations[:10]
    for conv in recent:
        origin_label, origin_desc = _detect_origin_from_record(conv)
        model_name = conv.get("model_name") or "Unknown"
        request_model = conv.get("request_model") or "Unknown"
        provider = conv.get("provider_name") or "Unknown"
        resp_ms = conv.get("response_time_ms") or 0
        timestamp = format_time(conv.get("timestamp", ""))

        with st.expander(f"#{conv.get('id')} {timestamp} - {provider} / {request_model}", expanded=False):
            st.markdown(f"**请求模型**: {request_model}  ")
            st.markdown(f"**实际返回模型**: {model_name}  ")
            st.markdown(f"**供应商**: {provider}  ")
            st.markdown(f"**响应耗时**: {resp_ms} ms  ")
            st.markdown(f"**诊断结果**: {origin_label}  ")
            st.markdown(f"**说明**: {origin_desc}  ")
            st.markdown(f"**用户输入**: {conv.get('user_input', '')[:120]}  ")
            st.markdown(f"**AI 输出**: {conv.get('ai_output', '')[:220]}  ")

    st.divider()
    st.subheader("最常见模型与供应商")

    st.markdown("**模型使用频率**")
    for model, count in model_counter.most_common(8):
        st.markdown(f"- {model}: {count} 次")

    st.markdown("**供应商调用频率**")
    for provider, count in provider_counter.most_common(8):
        st.markdown(f"- {provider}: {count} 次")
