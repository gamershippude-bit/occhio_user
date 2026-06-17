"""
Armazenamento de rostos — MySQL ou memória local (fallback).
"""
import logging
import pickle
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InMemoryFaceStore:
    """Fallback quando MySQL não está configurado."""

    def __init__(self):
        self._rostos: List[dict] = []
        self._lock = threading.Lock()
        self._next_id = 1

    def load_face_encodings(self) -> Tuple[List[np.ndarray], List[str], List[int]]:
        with self._lock:
            encodings, nomes, ids = [], [], []
            for r in self._rostos:
                encodings.append(r['encoding'])
                nomes.append(r['nome'])
                ids.append(r['id'])
            return encodings, nomes, ids

    def face_exists(self, face_encoding: np.ndarray, threshold: float = 0.6) -> bool:
        with self._lock:
            for r in self._rostos:
                dist = np.linalg.norm(np.array(face_encoding) - np.array(r['encoding']))
                if dist < threshold:
                    return True
        return False

    def save_face(
        self,
        face_encoding: np.ndarray,
        nome: str,
        relacao: str = 'conhecido',
        user: Optional[int] = None,
        label: Optional[str] = None,
        avisar: bool = True,
    ) -> bool:
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            self._rostos.append({
                'id': rid,
                'encoding': face_encoding,
                'nome': nome,
                'relacao': relacao,
                'label': label or nome,
                'avisar': avisar,
            })
        logger.info(f'✅ Rosto salvo em memória: {nome}')
        return True

    def get_nomes_com_aviso(self) -> List[str]:
        with self._lock:
            return [r['nome'] for r in self._rostos if r.get('avisar')]

    def get_relacao(self, nome: str) -> Optional[str]:
        with self._lock:
            for r in self._rostos:
                if r['nome'].lower() == nome.lower():
                    return r.get('relacao')
        return None

    def list_faces(self) -> List[dict]:
        with self._lock:
            return [
                {'id': r['id'], 'nome': r['nome'], 'label': r['label'], 'relacao': r['relacao'], 'avisar': r['avisar']}
                for r in self._rostos
            ]


def criar_face_store():
    """Tenta MySQL; se falhar, usa memória."""
    if not __import__('os').getenv('DB_HOST'):
        logger.warning('⚠️ DB_HOST não configurado — rostos salvos em memória (perdem-se ao reiniciar)')
        return InMemoryFaceStore()
    try:
        from db.database import DatabaseManager
        return DatabaseManager()
    except Exception as e:
        logger.error(f'❌ MySQL indisponível ({e}) — usando memória')
        return InMemoryFaceStore()
