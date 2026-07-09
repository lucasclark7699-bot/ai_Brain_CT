"""
预警系统面板：红色警告条 + 预警列表 + 已读标记
"""
import streamlit as st
import plotly.graph_objects as go

from src.database import get_alerts, save_alert, acknowledge_alert, get_unread_alert_count
from src.analyzer import detect_contradictions
from src.config import get_alert_config
from utils.helpers import format_time, get_severity_color, get_severity_icon


def render_alerts_panel():
    """渲染预警面板"""
    st.header("预警中心")
    st.caption("AI 行为异常检测 —— 引用错误、逻辑矛盾、话题切换")

    alert_cfg = get_alert_config()

    # 手动触发扫描
    col_scan, col_ackall = st.columns([3, 1])
    with col_scan:
        if st.button("立即扫描矛盾", type="primary"):
            with st.spinner("正在扫描..."):
                detected = detect_contradictions(recent_n=30)
                new_count = 0
                for alert in detected:
                    save_alert(
                        conversation_id=alert.get("conv_id"),
                        alert_type=alert["type"],
                        description=alert["description"],
                        severity=alert["severity"],
                    )
                    new_count += 1
                if new_count > 0:
                    st.success(f"扫描完成！发现 {new_count} 条新预警。")
                    st.rerun()
                else:
                    st.info("扫描完成，未发现异常。")

    with col_ackall:
        unread = get_unread_alert_count()
        if unread > 0:
            if st.button(f"全部已读 ({unread})"):
                all_alerts = get_alerts(acknowledged=0)
                for a in all_alerts:
                    acknowledge_alert(a["id"])
                st.rerun()

    # 自动扫描状态
    if alert_cfg.get("enabled"):
        st.caption(f"自动预警已启用 | 新消息后自动扫描")
    else:
        st.caption("自动预警已禁用")

    st.divider()

    # 预警列表
    filter_option = st.radio(
        "筛选", ["未读预警", "全部预警", "已读预警"],
        horizontal=True,
        key="alert_filter",
    )

    # 预警列表（按筛选条件一次性取全，避免饼图与列表数对不上 / 重复查库）
    if filter_option == "未读预警":
        alerts = get_alerts(acknowledged=0, limit=1000)
    elif filter_option == "已读预警":
        alerts = get_alerts(acknowledged=1, limit=1000)
    else:
        alerts = get_alerts(limit=1000)

    if not alerts:
        st.success("暂无预警记录，AI 表现正常。")
        return

    # 按严重程度统计（直接基于当前筛选结果，保证与下方列表完全一致）
    severity_stats = {}
    for a in alerts:
        sev = a["severity"]
        severity_stats[sev] = severity_stats.get(sev, 0) + 1

    # 显示统计饼图
    if severity_stats:
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_stats.keys()),
            values=list(severity_stats.values()),
            hole=0.5,
            marker=dict(colors=[get_severity_color(s) for s in severity_stats.keys()]),
            textinfo="label+value",
        )])
        fig.update_layout(
            title="预警类型分布",
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig)

    st.divider()
    st.subheader(f"预警列表 ({len(alerts)} 条)")

    for alert in alerts:
        sev_color = get_severity_color(alert["severity"])
        sev_icon = get_severity_icon(alert["severity"])

        with st.container():
            col_main, col_btn = st.columns([5, 1])

            with col_main:
                ack_badge = "" if alert["acknowledged"] else " 🔔"
                st.markdown(
                    f"### {sev_icon} [{alert['alert_type']}] "
                    f"<span style='color:{sev_color};font-size:0.8em'>{alert['severity'].upper()}</span>"
                    f"{ack_badge}",
                    unsafe_allow_html=True,
                )
                st.markdown(alert["description"])
                st.caption(
                    f"创建时间: {format_time(alert['created_at'])} | "
                    f"关联对话: #{alert['conversation_id'] or 'N/A'} | "
                    f"{'已读' if alert['acknowledged'] else '未读'}"
                )

            with col_btn:
                if not alert["acknowledged"]:
                    if st.button("标记已读", key=f"ack_{alert['id']}"):
                        acknowledge_alert(alert["id"])
                        st.rerun()
                else:
                    st.caption("✓ 已处理")

        st.divider()
