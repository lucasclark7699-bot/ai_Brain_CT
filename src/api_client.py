"""
可配置 API 客户端：封装 OpenAI SDK，支持多供应商切换
"""
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


@dataclass
class ChatResponse:
    content: str
    total_tokens: int = 0
    model_name: str = ""
    logprobs: Optional[list] = None
    finish_reason: str = ""
    error: Optional[str] = None


class APIClient:
    """统一的 API 调用封装"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 supports_logprobs: bool = True):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.supports_logprobs = supports_logprobs
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
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

            resp = self.client.chat.completions.create(**kwargs)

            choice = resp.choices[0]
            content = choice.message.content or ""

            logprobs_data = None
            if enable_logprobs and self.supports_logprobs and choice.logprobs:
                logprobs_data = self._parse_logprobs(choice.logprobs)

            return ChatResponse(
                content=content,
                total_tokens=resp.usage.total_tokens if resp.usage else 0,
                model_name=resp.model,
                logprobs=logprobs_data,
                finish_reason=choice.finish_reason or "",
            )

        except Exception as e:
            return ChatResponse(
                content="",
                error=str(e),
            )

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
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True, f"连接成功 (模型: {resp.model})"
        except Exception as e:
            return False, str(e)


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
        )

    @staticmethod
    def create_from_name(providers: list[dict], provider_name: str) -> Optional[APIClient]:
        """按名称从供应商列表中创建"""
        for p in providers:
            if p.get("name") == provider_name:
                return APIClientFactory.create_from_config(p)
        return None
