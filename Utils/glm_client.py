"""Cliente GLM-5 (Zhipu AI / Z.AI) para geração de texto."""

import os
import logging
import threading

logger = logging.getLogger(__name__)

_glm_client = None
_glm_lock = threading.Lock()

DEFAULT_MODEL = os.getenv('GLM_MODEL', 'glm-5')
DEFAULT_BASE_URL = os.getenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')


def _get_api_key() -> str:
    return (os.getenv('ZAI_API_KEY') or os.getenv('ZHIPU_API_KEY') or '').strip()


def glm_disponivel() -> bool:
    return bool(_get_api_key())


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
            if 'z.ai' in base_url:
                from zai import ZaiClient
                _glm_client = ZaiClient(api_key=api_key, base_url=base_url)
            else:
                from zai import ZhipuAiClient
                _glm_client = ZhipuAiClient(api_key=api_key, base_url=base_url)
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
    if not response.choices:
        logger.warning('GLM retornou choices vazio')
        return ''
    content = response.choices[0].message.content
    return (content or '').strip()
