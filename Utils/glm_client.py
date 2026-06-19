"""Cliente GLM-5 (Zhipu AI / Z.AI) para geração de texto."""

import os
import logging
import threading

import httpx

logger = logging.getLogger(__name__)

_glm_client = None
_http_client = None
_glm_lock = threading.Lock()

DEFAULT_MODEL = os.getenv('GLM_MODEL', 'glm-5')
DEFAULT_BASE_URL = os.getenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')


def _get_api_key() -> str:
    return (os.getenv('ZAI_API_KEY') or os.getenv('ZHIPU_API_KEY') or '').strip()


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is not None:
        return _http_client
    with _glm_lock:
        if _http_client is None:
            client_kwargs = dict(
                limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=30),
                timeout=httpx.Timeout(60.0),
            )
            try:
                _http_client = httpx.Client(http2=True, **client_kwargs)
            except Exception:
                _http_client = httpx.Client(**client_kwargs)
            logger.info('✅ GLM HTTP client persistente (keep-alive) inicializado')
    return _http_client


def glm_disponivel() -> bool:
    return bool(_get_api_key())


def _criar_cliente_sdk(api_key: str, base_url: str):
    http_client = _get_http_client()
    kwargs = {
        'api_key': api_key,
        'base_url': base_url,
        'http_client': http_client,
    }
    if 'z.ai' in base_url:
        from zai import ZaiClient
        try:
            return ZaiClient(**kwargs)
        except TypeError:
            return ZaiClient(api_key=api_key, base_url=base_url)
    from zai import ZhipuAiClient
    try:
        return ZhipuAiClient(**kwargs)
    except TypeError:
        return ZhipuAiClient(api_key=api_key, base_url=base_url)


def get_glm_client():
    global _glm_client
    if _glm_client is not None:
        return _glm_client
    with _glm_lock:
        if _glm_client is not None:
            return _glm_client
        api_key = _get_api_key()
        if not api_key:
            return None
        base_url = os.getenv('GLM_BASE_URL', DEFAULT_BASE_URL).strip()
        try:
            _glm_client = _criar_cliente_sdk(api_key, base_url)
            logger.info('✅ GLM client inicializado (modelo: %s)', DEFAULT_MODEL)
        except Exception as e:
            logger.error('❌ Erro ao inicializar GLM: %s', e)
    return _glm_client


def chat(messages, model=None, max_tokens=200, temperature=0.7) -> str:
    client = get_glm_client()
    if not client:
        raise ValueError('ZAI_API_KEY não configurada')

    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
