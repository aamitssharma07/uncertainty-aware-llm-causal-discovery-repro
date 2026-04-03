import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from prompt_loader import build_prompt
from parser import build_batch_response_model, normalize_batch_query_output, parse_batch_query_text

QueryMode = str  # "edge" | "no_edge"


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return str(content)


class OpenRouterLLM:
    """
    LLM client using LangChain on top of OpenRouter's OpenAI-compatible API.

    Required env var: OPENROUTER_API_KEY
    Optional env var: OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
    """

    def __init__(self):
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment / .env")
        self.api_key = api_key
        self.base_url = base_url

    def label_all_pairs(
        self,
        *,
        model: str,
        schema: Dict[str, Any],
        variables: List[str],
        query_mode: QueryMode,
        prompt_family: str,
        prompt_style: str,
        input_csv: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Send one prompt per ordered pair and return the parsed label dict.

        Returns:
            { "A->B": {"confidence": float, ...}, ... }
        """
        llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        batch_response_model = build_batch_response_model(variables)
        prompt = build_prompt(
            query_mode=query_mode,
            schema=schema,
            prompt_family=prompt_family,
            prompt_style=prompt_style,
            attribute_a="",
            attribute_b="",
            input_csv=input_csv,
        )
        chain = prompt | llm.with_structured_output(batch_response_model)
        try:
            payload = chain.invoke({})
            return normalize_batch_query_output(payload, variables)
        except Exception as structured_error:
            raw_chain = prompt | llm
            raw_response = raw_chain.invoke({})
            raw_text = _response_to_text(raw_response)
            try:
                return parse_batch_query_text(raw_text, variables)
            except Exception:
                raise structured_error
