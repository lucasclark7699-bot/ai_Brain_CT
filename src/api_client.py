"""
可配置 API 客户端：封装 OpenAI SDK，支持多供应商切换
"""
import json
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from utils.helpers import sanitize_text


@dataclass
class ChatResponse:
    content: str
    total_tokens: int = 0
    model_name: str = ""
    request_model: str = ""
    provider_name: str = ""
    response_time_ms: int = 0
    logprobs: Optional[list] = None
    finish_reason: str = ""
    error: Optional[str] = None


class APIClient:
    """统一的 API 调用封装"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 supports_logprobs: bool = True, provider_name: str = "Unknown"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.supports_logprobs = supports_logprobs
        self.provider_name = provider_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=15,          # 单次请求最长等待 15 秒，超时即报错不再干等
            max_retries=1,       # 失败仅重试 1 次，避免叠加超时拖垮体验
        )

    def chat(self, messages: list[dict],
             enable_logprobs: bool = True,
             temperature: float = 0.7,
             max_tokens: int = 4096) -> ChatResponse:
        """
        发送聊天请求
        messages: [{"role": "user/assistant/system", "content": "..."}, ...]
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if enable_logprobs and self.supports_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5

            kwargs["messages"] = [
                {"role": m["role"], "content": sanitize_text(m.get("content", "") or "")}
                for m in messages
            ]
            start = time.perf_counter()
            resp = self.client.chat.completions.create(**kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            choice = resp.choices[0]
            content = sanitize_text(choice.message.content or "")

            logprobs_data = None
            if enable_logprobs and self.supports_logprobs and getattr(choice, "logprobs", None):
                logprobs_data = self._parse_logprobs(choice.logprobs)

            return ChatResponse(
                content=content,
                total_tokens=resp.usage.total_tokens if getattr(resp, 'usage', None) else 0,
                model_name=getattr(resp, 'model', self.model) or self.model,
                request_model=self.model,
                provider_name=self.provider_name,
                response_time_ms=elapsed_ms,
                logprobs=logprobs_data,
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            return ChatResponse(
                content="",
                error=str(e),
                request_model=self.model,
                provider_name=self.provider_name,
            )

    def chat_with_raw_response(self, messages: list[dict],
                               enable_logprobs: bool = True,
                               temperature: float = 0.7,
                               max_tokens: int = 4096):
        """发送聊天请求并返回解析后的结果及原始 HTTP 响应。"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if enable_logprobs and self.supports_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5

            kwargs["messages"] = [
                {"role": m["role"], "content": sanitize_text(m.get("content", "") or "")}
                for m in messages
            ]
            start = time.perf_counter()
            raw_resp = self.client.chat.completions.with_raw_response.create(**kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            parsed = None
            if getattr(raw_resp, 'text', None):
                try:
                    parsed = json.loads(raw_resp.text)
                except Exception:
                    parsed = None
            if parsed is None and hasattr(raw_resp, 'parse'):
                try:
                    parsed = raw_resp.parse(to=dict)
                except Exception:
                    parsed = None

            choice = None
            if parsed and isinstance(parsed, dict):
                choices = parsed.get("choices", [])
                if choices:
                    choice = choices[0]

            content = ""
            reasoning = ""
            if choice and isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "") or ""
                    # DeepSeek 等推理模型把思考过程放在 reasoning_content，
                    # 正文可能为空，此时拿推理内容兜底，避免能力测试误判失败
                    reasoning = message.get("reasoning_content", "") or ""

            if not content and reasoning:
                content = reasoning

            logprobs_data = None
            if enable_logprobs and self.supports_logprobs and choice and choice.get("logprobs"):
                logprobs_data = self._parse_logprobs(choice["logprobs"])

            return ChatResponse(
                content=sanitize_text(content),
                total_tokens=(parsed.get("usage", {}).get("total_tokens", 0) if isinstance(parsed, dict) else 0),
                model_name=(parsed.get("model", self.model) or self.model) if isinstance(parsed, dict) else self.model,
                request_model=self.model,
                provider_name=self.provider_name,
                response_time_ms=elapsed_ms,
                logprobs=logprobs_data,
                finish_reason=(choice.get("finish_reason", "") if isinstance(choice, dict) else ""),
            ), raw_resp
        except Exception as e:
            return ChatResponse(
                content="",
                error=str(e),
                request_model=self.model,
                provider_name=self.provider_name,
            ), None

    def _parse_logprobs(self, logprobs_obj) -> list[dict]:
        """解析 logprobs 数据为统一格式"""
        result = []
        try:
            for item in logprobs_obj.content or []:
                result.append({
                    "token": item.token,
                    "logprob": item.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (item.top_logprobs or [])
                    ]
                })
        except Exception:
            pass
        return result

    def test_connection(self) -> tuple[bool, str]:
        """测试 API 连接"""
        try:
            start = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return True, f"连接成功 (模型: {getattr(resp, 'model', self.model)}，时延: {elapsed_ms}ms)"
        except Exception as e:
            return False, str(e)

    def is_official_endpoint(self) -> bool:
        return "openai.com" in self.base_url.lower()

    def detect_model_origin(self) -> tuple[str, str]:
        """基于接口域名和模型名判断是否可能为官方模型"""
        normalized = self.model.lower()
        official_prefixes = ["gpt-", "gpt4", "gpt4o", "gpt-4", "gpt-3.5", "text-davinci", "text-curie"]
        if self.is_official_endpoint():
            if any(normalized.startswith(prefix) for prefix in official_prefixes):
                return "官方模型", "当前接口为 OpenAI 官方域名，模型名称符合常见官方命名格式。"
            return "疑似非官方模型", "当前接口域名为 OpenAI 官方域名，但模型名称不符合常见官方命名。"

        if any(normalized.startswith(prefix) for prefix in official_prefixes):
            return "疑似套壳模型", "当前 base_url 不是 OpenAI 官方域名，模型名看起来像官方名称，但可能是兼容接口或套壳服务。"
        return "非官方模型", "当前接口地址不是 OpenAI 官方域名，且模型名称也不属于常见官方模型。"


class APIClientFactory:
    """API 客户端工厂"""

    @staticmethod
    def create_from_config(config: dict) -> APIClient:
        """从配置字典创建客户端"""
        return APIClient(
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            api_key=config.get("api_key", ""),
            model=config.get("model", "gpt-3.5-turbo"),
            supports_logprobs=config.get("supports_logprobs", True),
            provider_name=config.get("name", "Unknown"),
        )

    @staticmethod
    def create_from_name(providers: list[dict], provider_name: str) -> Optional[APIClient]:
        """按名称从供应商列表中创建"""
        for p in providers:
            if p.get("name") == provider_name:
                return APIClientFactory.create_from_config(p)
        return None
