"""
Armazenamento de rostos — MySQL (user_rec_facial) ou memória local (fallback).
"""
import logging
import os
import threading
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

    def find_existing_name(self, face_encoding: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        try:
            import face_recognition
            query = np.asarray(face_encoding, dtype=np.float64)
            with self._lock:
                for r in self._rostos:
                    dist = float(face_recognition.face_distance([r['encoding']], query)[0])
                    if dist < threshold:
                        return r['nome']
        except Exception:
            pass
        return None

    def face_exists(self, face_encoding: np.ndarray, threshold: float = 0.6) -> bool:
        return self.find_existing_name(face_encoding, threshold) is not None

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

    def get_faces_catalog(self) -> Dict[str, dict]:
        with self._lock:
            catalog = {}
            for r in self._rostos:
                chave = r['nome'].lower()
                catalog[chave] = {
                    'id': r['id'],
                    'nome': r['nome'],
                    'relacao': r.get('relacao', 'conhecido'),
                    'label': r.get('label', r['nome']),
                    'user': 0,
                    'avisar': r.get('avisar', True),
                    'data_cadastro': None,
                }
            return catalog

    def is_mysql(self) -> bool:
        return False

    def list_faces(self) -> List[dict]:
        with self._lock:
            return [
                {'id': r['id'], 'nome': r['nome'], 'label': r['label'], 'relacao': r['relacao'], 'avisar': r['avisar']}
                for r in self._rostos
            ]


def criar_face_store():
    """Conecta ao MySQL (tabela user_rec_facial) ou usa memória se DB_HOST ausente."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if not os.getenv('DB_HOST'):
        logger.warning(
            '⚠️ DB_HOST não configurado — rostos só em memória. '
            'Configure DB_* no .env para usar MySQL (user_rec_facial).'
        )
        return InMemoryFaceStore()
    try:
        from db.database import DatabaseManager
        store = DatabaseManager()
        catalog = store.get_faces_catalog()
        logger.info(f'✅ MySQL conectado — {len(catalog)} rosto(s) em user_rec_facial')
        return store
    except Exception as e:
        logger.error(f'❌ MySQL indisponível ({e}) — usando memória')
        return InMemoryFaceStore()
