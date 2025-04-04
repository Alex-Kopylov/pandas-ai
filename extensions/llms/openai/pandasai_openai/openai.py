import os
from typing import Any, Dict, Optional

import openai

from pandasai.exceptions import APIKeyNotFoundError, UnsupportedModelError
from pandasai.helpers import load_dotenv

from .base import BaseOpenAI

load_dotenv()


class OpenAI(BaseOpenAI):
    """OpenAI LLM using BaseOpenAI Class.

    An API call to OpenAI API is sent and response is recorded and returned.
    The default chat model is **gpt-4o-mini**.
    """

    model: str = "gpt-4o-mini"

    def __init__(
        self,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        """
        __init__ method of OpenAI Class

        Args:
            api_token (str): API Token for OpenAI platform.
            **kwargs: Extended Parameters inferred from BaseOpenAI class

        """
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None

        if not self.api_token:
            raise APIKeyNotFoundError("OpenAI API key is required")

        self.api_base = (
            kwargs.get("api_base") or os.getenv("OPENAI_API_BASE") or self.api_base
        )
        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")
        if self.openai_proxy:
            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}

        self._set_params(**kwargs)
        # set the client
        self._is_chat_model = True
        self.client = openai.OpenAI(**self._client_params).chat.completions
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API"""
        return {
            **super()._default_params,
            "model": self.model,
        }

    @property
    def type(self) -> str:
        return "openai"
