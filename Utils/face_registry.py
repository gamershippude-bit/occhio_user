"""
Reconhecimento facial, alertas e operações de rosto.
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONFIRMACOES = (
    'sim', 's', 'pode', 'pode ser', 'quero', 'vai', 'claro',
    'beleza', 'tá', 'ta', 'ok', 'isso', 'isso mesmo', 'exato',
    'por favor', 'pfv', 'pf', 'vamo', 'vamos',
    'si', 'com certeza', 'positivo', 'confirmo', 'uhum', 'aham',
)
NEGACOES = (
    'não', 'nao', 'n', 'negativo', 'deixa', 'deixa pra la',
    'esquece', 'tanto faz', 'sem querer', 'não quero', 'nao quero',
    'cancela', 'dispensa', 'sem aviso', 'de jeito nenhum',
)
PALAVRAS_SIM = CONFIRMACOES
PALAVRAS_NAO = NEGACOES
RESPOSTAS_SIM_CURTAS = frozenset({'s', 'si', 'sim', 'n', 'nao', 'não'})


def detectar_sim(texto: str) -> Optional[bool]:
    t = re.sub(r'[^\w\sáàâãéêíóôõúüç]', ' ', texto.lower().strip())
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return None

    if t in RESPOSTAS_SIM_CURTAS:
        return t in ('s', 'si', 'sim')

    tokens = set(t.split())
    if tokens & {'sim', 'si', 's'} and not tokens & {'nao', 'não', 'n'}:
        return True
    if tokens & {'nao', 'não', 'n'} and not tokens & {'sim', 'si'}:
        return False

    if re.search(r'\bn[aã]o\b', t):
        return False
    if re.search(r'\bsim\b', t):
        return True

    for p in PALAVRAS_NAO:
        if p in t and len(p) >= 3:
            return False
    for p in PALAVRAS_SIM:
        if p in t and len(p) >= 2:
            return True
    return None


class FaceRegistry:
    """Gerencia reconhecimento facial e alertas de rostos conhecidos."""

    def __init__(self, face_detector=None, face_store=None):
        self.face_detector = face_detector
        self.face_store = face_store
        self._alertas_recentes: Dict[str, float] = {}
        self._catalog: Dict[str, dict] = {}
        self.alerta_cooldown = 25.0

    def recarregar_rostos(self) -> int:
        if not self.face_detector or not self.face_store:
            return 0
        try:
            encodings, nomes, _ = self.face_store.load_face_encodings()
            self.face_detector.carregar_encodings(encodings, nomes)
            if hasattr(self.face_store, 'get_faces_catalog'):
                self._catalog = self.face_store.get_faces_catalog()
            self._alertas_recentes.clear()
            logger.info(f'📥 {len(nomes)} rosto(s) do banco: {list(self._catalog.values())}')
            return len(nomes)
        except Exception as e:
            logger.error(f'Erro ao recarregar rostos: {e}')
            return 0

    def _meta_rosto(self, nome: str) -> Optional[dict]:
        if not nome:
            return None
        meta = self._catalog.get(nome.lower())
        if meta:
            return meta
        if self.face_store:
            rel = self.face_store.get_relacao(nome)
            if rel:
                return {'nome': nome, 'relacao': rel}
        return None

    def capturar_encoding_do_frame_atual(self, occhio, timeout: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Solicita captura de encoding via loop de stream (evita dlib na thread de voz)."""
        if not occhio or not hasattr(occhio, '_encoding_request'):
            return None, 'Captura de rosto indisponível no momento.'

        with occhio._encoding_lock:
            occhio._encoding_result = None
            occhio._encoding_erro = None
        occhio._encoding_request.set()

        inicio = time.time()
        while time.time() - inicio < timeout:
            with occhio._encoding_lock:
                if occhio._encoding_result is not None:
                    return occhio._encoding_result, None
                if occhio._encoding_erro is not None:
                    return None, occhio._encoding_erro
            time.sleep(0.05)

        occhio._encoding_request.clear()
        logger.warning('⚠️ Timeout ao capturar encoding — tente novamente')
        return None, 'Timeout ao capturar rosto da câmera. Tente novamente.'

    def capturar_encoding_async(self, occhio, timeout: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[str]]:
        return self.capturar_encoding_do_frame_atual(occhio, timeout)

    def remover_por_nome(self, nome: str) -> bool:
        """Remove rosto pelo nome (case-insensitive, busca parcial)."""
        if not self.face_store or not hasattr(self.face_store, 'list_faces'):
            return False
        nome_busca = nome.strip().lower().rstrip('?.!')
        if not nome_busca:
            return False
        faces = self.face_store.list_faces()
        removidos = 0
        for face in faces:
            fn = face.get('nome', '').lower()
            if nome_busca in fn or fn in nome_busca:
                if hasattr(self.face_store, 'delete_face_by_id'):
                    if self.face_store.delete_face_by_id(face['id']):
                        removidos += 1
                elif hasattr(self.face_store, 'delete_face'):
                    if self.face_store.delete_face(face.get('nome', '')):
                        removidos += 1
        if removidos:
            self.recarregar_rostos()
            return True
        return False

    def detectar_faces_stream(self, frame) -> List[Dict[str, Any]]:
        if not self.face_detector or not hasattr(self.face_detector, 'detectar_faces_bbox'):
            return []
        rostos = self.face_detector.detectar_faces_bbox(frame)
        for rosto in rostos:
            if not rosto.get('conhecido'):
                continue
            meta = self._meta_rosto(rosto.get('nome', ''))
            if meta:
                rosto['relacao'] = meta.get('relacao')
                rosto['avisar'] = meta.get('avisar')
        return rostos

    def verificar_alertas(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.face_store:
            return []
        agora = time.time()
        alertas = []
        nomes_aviso = self.face_store.get_nomes_com_aviso()

        for face in faces:
            nome = face.get('nome', '')
            if not nome or nome.lower() in ('desconhecido', 'unknown'):
                continue
            if nome not in nomes_aviso:
                continue
            ultimo = self._alertas_recentes.get(nome, 0)
            if agora - ultimo < self.alerta_cooldown:
                continue
            self._alertas_recentes[nome] = agora
            alertas.append({
                'nome': nome,
                'mensagem': f'{nome} está aqui.',
            })
        return alertas

    def get_catalogo(self) -> Dict[str, dict]:
        """Retorna o catálogo completo de rostos cadastrados."""
        if self.face_store and hasattr(self.face_store, 'get_faces_catalog'):
            catalog = self.face_store.get_faces_catalog() or {}
            if catalog:
                self._catalog = catalog
            return catalog
        return dict(self._catalog)


def recarregar_estado_facial(occhio) -> bool:
    """
    Recarrega completamente o estado de reconhecimento facial a partir do banco.
    Deve ser chamada após qualquer operação de escrita (salvar, deletar, renomear, atualizar).
    """
    try:
        if not occhio or not occhio.face_registry:
            return False

        qtd = occhio.face_registry.recarregar_rostos()
        occhio._last_rostos = []

        logger.info(f'✅ Estado facial recarregado: {qtd} rosto(s)')
        return True
    except Exception as e:
        logger.error(f'❌ Erro ao recarregar estado facial: {e}')
        return False
