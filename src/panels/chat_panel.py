"""
聊天面板：对话界面、/tag 标签命令解析、自动存储对话到数据库
"""
import streamlit as st
from src.database import (
    save_conversation, save_keywords, get_conversations, get_all_project_tags,
    get_unread_alert_count,
)
from src.api_client import APIClient
from src.analyzer import extract_keywords, detect_contradictions
from src.config import get_alert_config
from utils.helpers import parse_tag_command, format_time, format_tokens
import json


def render_chat_panel(client: APIClient):
    """渲染聊天面板"""
    st.header("AI 对话")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_tag" not in st.session_state:
        st.session_state.current_tag = ""
    if "logprobs_cache" not in st.session_state:
        st.session_state.logprobs_cache = []
    if "pending_tag_input" not in st.session_state:
        st.session_state.pending_tag_input = ""

    # 标签栏
    col_tag, col_clear = st.columns([3, 1])
    with col_tag:
        current_tag_display = st.session_state.current_tag or "（无标签）"
        st.caption(f"当前项目标签: **{current_tag_display}**")

    with col_clear:
        if st.button("清除对话"):
            st.session_state.messages = []
            st.rerun()

    # 标签快捷输入区
    with st.expander("标签管理", expanded=False):
        tag_input = st.text_input(
            "输入 `/tag 项目名` 切换标签",
            value=st.session_state.pending_tag_input,
            key="tag_input_field",
            placeholder="/tag 项目A"
        )
        if tag_input != st.session_state.pending_tag_input:
            st.session_state.pending_tag_input = tag_input

        tag_name, _ = parse_tag_command(tag_input)
        if tag_name:
            if st.button(f"切换到标签: {tag_name}"):
                st.session_state.current_tag = tag_name
                st.session_state.pending_tag_input = ""
                st.rerun()

        # 显示已有标签
        existing_tags = get_all_project_tags()
        if existing_tags:
            st.caption("已有标签（点击切换）：")
            cols = st.columns(min(len(existing_tags), 5))
            for i, tag in enumerate(existing_tags):
                with cols[i % 5]:
                    if st.button(tag, key=f"tag_{tag}"):
                        st.session_state.current_tag = tag
                        st.rerun()

    st.divider()

    # 对话历史显示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])

    # 聊天输入
    if prompt := st.chat_input("输入消息..."):
        # 解析 /tag 命令
        tag_name, clean_prompt = parse_tag_command(prompt)
        if tag_name:
            st.session_state.current_tag = tag_name
            st.session_state.pending_tag_input = ""
            if not clean_prompt:
                # 仅切换标签，不发送消息
                st.rerun()
            prompt = clean_prompt

        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "meta": ""
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 API
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中..."):
                # 构建消息历史（只发送必要字段）
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                response = client.chat(api_messages, enable_logprobs=True)

                if response.error:
                    st.error(f"API 错误: {response.error}")
                    ai_content = f"[错误] {response.error}"
                    logprobs_data = {}
                else:
                    st.markdown(response.content)
                    ai_content = response.content
                    logprobs_data = response.logprobs or {}

                    # 显示 Token 信息
                    st.caption(
                        f"模型: {response.model_name} | "
                        f"Token: {format_tokens(response.total_tokens)} | "
                        f"标签: {st.session_state.current_tag or '无'}"
                    )

        # 保存到数据库
        conv_id = save_conversation(
            project_tag=st.session_state.current_tag,
            user_input=prompt,
            ai_output=ai_content,
            total_tokens=response.total_tokens,
            model_name=response.model_name,
            logprobs_data=logprobs_data,
        )

        # 提取并保存关键词
        if ai_content and "[错误]" not in ai_content:
            keywords = extract_keywords(f"{prompt} {ai_content}")
            if keywords:
                save_keywords(conv_id, keywords)

        # 存储到 session
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_content,
            "meta": f"Token: {format_tokens(response.total_tokens)} | {response.model_name}"
        })

        # 存储 logprobs（用于热力图）
        if logprobs_data:
            st.session_state.logprobs_cache.append({
                "conv_id": conv_id,
                "user_input": prompt,
                "logprobs": logprobs_data,
            })

        # 预警检测
        alert_cfg = get_alert_config()
        if alert_cfg.get("enabled", True) and alert_cfg.get("auto_scan_on_new_message", True):
            alerts = detect_contradictions(recent_n=20)
            if alerts:
                alert_count = get_unread_alert_count()
                if alert_count > 0:
                    st.warning(f"检测到 {alert_count} 条新预警，请查看「预警中心」")

        st.rerun()

    # 历史对话回顾
    with st.expander("历史对话记录", expanded=False):
        project_filter = st.selectbox(
            "按项目筛选",
            ["全部"] + get_all_project_tags(),
            key="history_filter"
        )
        tag_filter = "" if project_filter == "全部" else project_filter
        history = get_conversations(tag_filter, limit=30)

        for conv in history:
            with st.container():
                st.markdown(f"**{format_time(conv.get('timestamp', ''))}** "
                            f"`{conv.get('project_tag', '') or '无标签'}` "
                            f"[Token: {conv.get('total_tokens', 0)}]")
                user_preview = (conv.get("user_input", "") or "")[:80]
                st.caption(f"Q: {user_preview}...")
                ai_preview = (conv.get("ai_output", "") or "")[:80]
                st.caption(f"A: {ai_preview}...")
            st.divider()
