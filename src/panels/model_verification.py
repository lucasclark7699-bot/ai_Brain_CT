"""
模型验证面板：综合判断模型请求返回一致性、能力测试、上下文记忆和服务器指纹。
"""
import re
import streamlit as st
from datetime import datetime
from src.api_client import APIClient
from utils.helpers import format_tokens, sanitize_float


def _extract_number(text: str) -> int | None:
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(nums[0])
    except ValueError:
        return None


def _normalize_model_name(name: str) -> str:
    return (name or "").strip().lower()


def _extract_chinese_number(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    mapping = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
        "十三": 13,
        "十四": 14,
        "十五": 15,
    }
    for word, value in mapping.items():
        if word in text:
            return value
    return None


def _format_server_info(headers) -> str:
    if not headers:
        return "未知"
    values = []
    for key in ["server", "via", "x-powered-by", "x-cache", "cf-ray"]:
        value = headers.get(key)
        if value:
            values.append(f"{key}: {value}")
    return " | ".join(values) if values else "未知"


def _extract_two_numbers(text: str) -> tuple[int | None, int | None]:
    if not text:
        return None, None

    # 优先匹配“火 7 木 6”或“7火 6木”这类格式
    match = re.search(r"火[^0-9]*(\d+).*木[^0-9]*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d+)[^0-9]*火.*?(\d+)[^0-9]*木", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None


def _is_official_provider(provider_name: str, base_url: str) -> bool:
    norm = f"{provider_name or ''} {base_url or ''}".lower()
    return any(token in norm for token in [
        "openai.com",
        "api.openai",
        "api.deepseek.com",
        "deepseek",
        "dashscope.aliyuncs.com",
        "qwen",
        "cloud.tencent",
        "tencent",
        "aliyun",
    ])


def _is_same_provider_model(requested: str, returned: str, provider_name: str, base_url: str) -> bool:
    requested_norm = _normalize_model_name(requested)
    returned_norm = _normalize_model_name(returned)
    provider_norm = (provider_name or "").lower()
    base_norm = (base_url or "").lower()

    if requested_norm == returned_norm:
        return True
    if returned_norm.startswith(requested_norm) or requested_norm.startswith(returned_norm):
        return True

    # 处理 DeepSeek / Qwen 等厂商命名差异
    if "deepseek" in provider_norm or "api.deepseek.com" in base_norm:
        if "deepseek" in requested_norm:
            return bool(returned_norm and (
                returned_norm.startswith("deepseek") or
                returned_norm.startswith("gpt") or
                returned_norm.startswith("text") or
                returned_norm.startswith("chat")
            ))
        return requested_norm == returned_norm
    if "qwen" in provider_norm or "dashscope.aliyuncs.com" in base_norm:
        if "qwen" in requested_norm:
            return bool(returned_norm and (
                returned_norm.startswith("qwen") or
                returned_norm.startswith("gpt") or
                returned_norm.startswith("text") or
                "chat" in returned_norm
            ))
        return requested_norm == returned_norm

    if requested_norm in returned_norm or returned_norm in requested_norm:
        return True

    # 厂商前缀相同则默认认为是同一家族模型
    req_root = requested_norm.split("-")[0]
    ret_root = returned_norm.split("-")[0]
    if req_root == ret_root and req_root not in {"gpt", "o3", "text", "gpt4o"}:
        return True

    return False


def _score_verification(results: dict) -> dict:
    score = 0
    reasons = []

    model_match = results["model_consistency"]["match"]
    if model_match:
        score += 40
    else:
        reasons.append(
            f"请求模型 {results['model_consistency']['requested']}，返回模型 {results['model_consistency']['returned']}，名称不一致。"
        )

    if results["counting_test"]["pass"]:
        score += 30
    else:
        reasons.append(
            f"数数测试失败：期望 {results['counting_test']['expected']}，实际 {results['counting_test']['actual']}。"
        )

    if results["context_test"]["pass"]:
        score += 30
    else:
        reasons.append(
            f"上下文记忆测试失败：期望 {results['context_test']['expected']}，实际 {results['context_test']['actual']}。"
        )

    server_info = results.get("server_info", "") or ""
    server_lower = server_info.lower()
    if _is_official_provider(results.get("provider_name", ""), results.get("base_url", "")):
        if any(token in server_lower for token in ["gunicorn", "kong", "fastly"]):
            reasons.append(f"服务器指纹异常：{server_info}")
    else:
        if any(token in server_lower for token in ["nginx", "cloudflare", "gunicorn", "kong", "fastly"]):
            reasons.append(f"服务器指纹异常：{server_info}")

    response_speed = results.get("response_speed", 0.0)
    if response_speed and response_speed < 0.2:
        reasons.append(f"响应速度过快：{response_speed:.2f}s，可能走缓存或中转。")

    if score >= 80 and len(reasons) == 0:
        verdict = "✅ 真模型（无明显套壳迹象）"
    elif score >= 60:
        verdict = "⚠️ 存疑（建议人工复核）"
    else:
        verdict = "❌ 假模型/套壳（疑似冒充）"

    results["score"] = score
    results["verdict"] = verdict
    results["reasons"] = reasons
    return results


def _chat_with_raw_response(client: APIClient, messages: list[dict], max_tokens: int = 10):
    response, raw = client.chat_with_raw_response(messages, enable_logprobs=False, max_tokens=max_tokens)
    returned = response.model_name or client.model
    content = response.content or ""
    headers = getattr(raw, "headers", {}) if raw else {}
    elapsed = 0.0
    if raw and getattr(raw, "elapsed", None) is not None:
        elapsed = raw.elapsed.total_seconds()
    return {
        "returned_model": returned,
        "content": content,
        "headers": headers,
        "elapsed": elapsed,
        "error": response.error,
    }


def verify_model_endpoint(client: APIClient) -> dict:
    requested_model = client.model

    # 1. 模型名一致性检测
    consistency = _chat_with_raw_response(
        client,
        [{"role": "user", "content": "你好"}],
        max_tokens=5,
    )

    returned_model = consistency["returned_model"]
    model_match = _is_same_provider_model(
        requested_model,
        returned_model,
        client.provider_name,
        client.base_url,
    )

    model_consistency = {
        "requested": requested_model,
        "returned": returned_model,
        "match": model_match,
    }

    # 2. 数数能力测试
    counting_fire = 7
    counting_wood = 6
    counting_string = "火" * counting_fire + "木" * counting_wood
    counting_prompt = (
        "下面文本只包含两个字符：火 和 木。请分别统计它们出现的次数。"
        "直接输出两个数字，先输出火的次数，再输出木的次数，用空格分隔，不要其他内容。\n"
        f"文本：{counting_string}"
    )
    counting = _chat_with_raw_response(
        client,
        [{"role": "user", "content": counting_prompt}],
        max_tokens=15,
    )
    first_counting_answer = counting["content"].strip()
    fire_actual, wood_actual = _extract_two_numbers(first_counting_answer)
    if fire_actual is None or wood_actual is None:
        fire_actual = fire_actual if fire_actual is not None else _extract_chinese_number(first_counting_answer)
        wood_actual = wood_actual if wood_actual is not None else _extract_chinese_number(first_counting_answer)

    retry_fire_answer = None
    retry_wood_answer = None

    # 如果只错了一个数量，尝试单独重新确认该字符的计数
    if fire_actual != counting_fire or wood_actual != counting_wood:
        if fire_actual != counting_fire:
            retry_fire = _chat_with_raw_response(
                client,
                [{
                    "role": "user",
                    "content": (
                        "下面文本只包含“火”字符，请数一数“火”出现了多少次。"
                        "不要数空格、不要数标点、不要数引号，也不要输出其他文字。\n"
                        f"文本：{'火' * counting_fire}"
                    )
                }],
                max_tokens=10,
            )
            retry_fire_answer = retry_fire["content"].strip()
            retry_fire_actual = _extract_number(retry_fire_answer)
            if retry_fire_actual == counting_fire:
                fire_actual = counting_fire

        if wood_actual != counting_wood:
            retry_wood = _chat_with_raw_response(
                client,
                [{
                    "role": "user",
                    "content": (
                        "下面文本只包含“木”字符，请数一数“木”出现了多少次。"
                        "不要数空格、不要数标点、不要数引号，也不要输出其他文字。\n"
                        f"文本：{'木' * counting_wood}"
                    )
                }],
                max_tokens=10,
            )
            retry_wood_answer = retry_wood["content"].strip()
            retry_wood_actual = _extract_number(retry_wood_answer)
            if retry_wood_actual == counting_wood:
                wood_actual = counting_wood

    counting_pass = fire_actual == counting_fire and wood_actual == counting_wood
    counting_test = {
        "expected": f"{counting_fire}/{counting_wood}",
        "actual": f"{fire_actual}/{wood_actual}",
        "pass": counting_pass,
        "raw": (
            f"第一次: {first_counting_answer}"
            + (f" | retry_fire: {retry_fire_answer}" if retry_fire_answer is not None else "")
            + (f" | retry_wood: {retry_wood_answer}" if retry_wood_answer is not None else "")
        ),
    }

    # 3. 上下文记忆测试
    context_messages = [
        {"role": "user", "content": "记住数字 7"},
        {"role": "user", "content": "记住数字 3"},
    ]
    for message in context_messages:
        _chat_with_raw_response(client, message if isinstance(message, list) else [message], max_tokens=5)

    context_prompt = [
        {"role": "user", "content": "记住数字 7"},
        {"role": "user", "content": "记住数字 3"},
        {"role": "user", "content": "我刚才让你记住的两个数字之和是多少？只输出数字。"},
    ]
    context_result = _chat_with_raw_response(client, context_prompt, max_tokens=10)
    context_answer = context_result["content"].strip()
    context_actual = _extract_number(context_answer)
    context_pass = context_actual == 10
    context_test = {
        "expected": 10,
        "actual": context_actual,
        "pass": context_pass,
        "raw": context_answer,
    }

    server_info = _format_server_info(consistency["headers"])
    response_speed = sanitize_float(consistency["elapsed"], 0.0)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model": requested_model,
        "returned_model": returned_model,
        "provider_name": client.provider_name,
        "base_url": client.base_url,
        "model_consistency": model_consistency,
        "counting_test": counting_test,
        "context_test": context_test,
        "server_info": server_info,
        "response_speed": response_speed,
        "raw_headers": dict(consistency["headers"] or {}),
    }

    return _score_verification(results)


def render_model_verification():
    st.header("模型真伪验证")
    st.caption("通过请求/返回一致性、能力测试、上下文记忆和响应指纹，给出综合判定。")

    if "verification_result" not in st.session_state:
        st.session_state.verification_result = None

    if st.button("开始执行模型验证", type="primary"):
        if not hasattr(st.session_state, "api_client"):
            st.warning("请先在侧边栏选择并配置 API 供应商，然后再次执行验证。")
        else:
            client = st.session_state.api_client
            with st.spinner("正在执行验证测试，请稍候..."):
                st.session_state.verification_result = verify_model_endpoint(client)
            st.success("验证完成，已生成结果。")
            st.rerun()

    result = st.session_state.get("verification_result")
    if result:
        st.metric("综合得分", f"{result['score']}/100", delta=result["verdict"])

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("模型一致性")
            st.write(f"请求模型: {result['model_consistency']['requested']}")
            st.write(f"返回模型: {result['model_consistency']['returned']}")
            st.write("✅ 一致" if result['model_consistency']['match'] else "❌ 不一致")

        with col2:
            st.subheader("能力测试")
            st.write(f"数数测试: {'通过' if result['counting_test']['pass'] else '失败'}")
            st.write(f"回答: {result['counting_test']['raw']}")
            st.write(f"上下文记忆: {'通过' if result['context_test']['pass'] else '失败'}")
            st.write(f"回答: {result['context_test']['raw']}")

        with col3:
            st.subheader("响应指纹")
            st.write(f"响应时延: {result['response_speed']:.2f}s")
            st.write(f"服务器信息: {result['server_info']} ")
            st.write(f"时间戳: {result['timestamp']}")

        if result['reasons']:
            st.divider()
            st.subheader("可疑点说明")
            for reason in result['reasons']:
                st.write(f"- {reason}")

        st.divider()
        st.subheader("原始 HTTP 头信息")
        st.code(result['raw_headers'])
    else:
        st.info("点击上方按钮开始执行模型真实度检测。")
