"""
Módulo de Gerenciamento de Banco de Dados para Cloud
"""

import logging
import pickle
import numpy as np
import pymysql
from datetime import datetime
from typing import List, Tuple, Optional, Any
import time
import threading

from db.config import get_cloud_db_config

logger = logging.getLogger("occhio.db.database")

class DatabaseManager:
    """Classe utilitária para interagir com o banco de dados MySQL em cloud."""

    def __init__(self, max_retries=3, retry_delay=2) -> None:
        self.config = get_cloud_db_config()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn = None
        self._lock = threading.Lock()
        
        self._connect_with_retry()

    def _connect_with_retry(self) -> None:
        """Tenta conectar com retry em caso de falha."""
        for attempt in range(self.max_retries):
            try:
                self.conn = pymysql.connect(**self.config)
                logger.info("✅ Conexão com banco de dados estabelecida.")

                # Confirmar banco usado
                with self.conn.cursor() as cursor:
                    cursor.execute("SELECT DATABASE();")
                    db_name = cursor.fetchone()[0]
                    logger.info(f"📊 Conectado ao banco: {db_name}")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Tentativa {attempt + 1}/{self.max_retries} falhou: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ Falha após {self.max_retries} tentativas: {e}")
                    raise

    def _get_cursor(self):
        """Obtém cursor com verificação de conexão."""
        try:
            # Verificar se conexão ainda está viva
            if self.conn is None or not self.conn.open:
                logger.warning("🔁 Reconectando ao banco...")
                self._connect_with_retry()
            
            return self.conn.cursor()
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter cursor: {e}")
            raise

    def close(self) -> None:
        """Fecha a conexão."""
        try:
            if self.conn and self.conn.open:
                self.conn.close()
                logger.info("🔒 Conexão com banco encerrada.")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao fechar conexão: {e}")

    # -------------------
    # Operações com rostos
    # -------------------

    def load_face_encodings(self) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """Carrega encodings de rostos existentes no banco de dados."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute("SELECT imgVetor, imgLabel, imgID FROM user_rec_facial")
                resultados = cursor.fetchall()
                cursor.close()

                encodings, nomes, ids = [], [], []
                for encoding_bytes, label, face_id in resultados:
                    try:
                        if not isinstance(encoding_bytes, bytes):
                            logger.warning(f"Encoding inválido para {label}: não é bytes")
                            continue

                        encoding = pickle.loads(encoding_bytes)
                        if encoding is not None:
                            encodings.append(encoding)
                            nomes.append(label if label else "desconhecido")
                            ids.append(face_id)
                    except Exception as e:
                        logger.warning(f"Erro ao carregar encoding para {label}: {e}")
                        continue

                logger.info(f"📥 Carregados {len(encodings)} encodings de faces.")
                return encodings, nomes, ids

            except Exception as e:
                logger.error(f"❌ Erro ao carregar encodings: {e}")
                return [], [], []

    def face_exists(self, face_encoding: np.ndarray, threshold: float = 0.6) -> bool:
        """Verifica se um rosto já está cadastrado no banco."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute("SELECT imgVetor FROM user_rec_facial")
                resultados = cursor.fetchall()
                cursor.close()

                for (encoding_bytes,) in resultados:
                    try:
                        encoding_db = pickle.loads(encoding_bytes)
                        distancia = np.linalg.norm(
                            np.array(face_encoding) - np.array(encoding_db)
                        )
                        if distancia < threshold:
                            return True
                    except Exception as e:
                        logger.warning(f"Erro ao comparar encodings: {e}")
                        continue

                return False
            except Exception as e:
                logger.error(f"❌ Erro ao verificar rosto cadastrado: {e}")
                return False

    def save_face(
        self,
        face_encoding: np.ndarray,
        nome: str,
        relacao: str = "conhecido",
        user: Optional[int] = None,
        label: Optional[str] = None,
    ) -> bool:
        """Salva um rosto detectado no banco de dados."""
        with self._lock:
            try:
                encoding_bytes = pickle.dumps(face_encoding)
                sql = """
                INSERT INTO user_rec_facial 
                    (imgVetor, imgNome, imgData, imgRelacao, imgUser, imgLabel)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                valores = (
                    encoding_bytes,
                    nome,
                    datetime.now(),
                    relacao,
                    user if user is not None else 0,
                    label if label else nome,  # Usa nome como label se não fornecido
                )

                cursor = self._get_cursor()
                cursor.execute(sql, valores)
                self.conn.commit()
                cursor.close()

                logger.info(f"✅ Rosto salvo com sucesso: Nome='{nome}'")
                return True

            except Exception as e:
                logger.error(f"❌ Erro ao salvar rosto: {e}")
                if self.conn:
                    self.conn.rollback()
                return False

    def load_all_faces(self) -> List[bytes]:
        """Carrega todos os rostos salvos no banco."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute("SELECT imgVetor FROM user_rec_facial")
                resultados = [row[0] for row in cursor.fetchall()]
                cursor.close()
                return resultados
            except Exception as e:
                logger.error(f"❌ Erro ao carregar rostos salvos: {e}")
                return []

    # NOVO: Endpoints para API de gerenciamento de rostos
    def list_faces(self) -> List[dict]:
        """Lista todas as faces cadastradas."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute("SELECT imgID, imgNome, imgLabel, imgData FROM user_rec_facial")
                resultados = cursor.fetchall()
                cursor.close()

                faces = []
                for face_id, nome, label, data in resultados:
                    faces.append({
                        "id": face_id,
                        "nome": nome,
                        "label": label,
                        "data_cadastro": data.isoformat() if data else None
                    })
                
                return faces
            except Exception as e:
                logger.error(f"❌ Erro ao listar faces: {e}")
                return []

    def delete_face(self, face_id: int) -> bool:
        """Remove uma face do banco."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute("DELETE FROM user_rec_facial WHERE imgID = %s", (face_id,))
                affected_rows = cursor.rowcount
                self.conn.commit()
                cursor.close()

                if affected_rows > 0:
                    logger.info(f"🗑️ Face {face_id} removida com sucesso")
                    return True
                else:
                    logger.warning(f"⚠️ Face {face_id} não encontrada")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Erro ao remover face: {e}")
                if self.conn:
                    self.conn.rollback()
                return False