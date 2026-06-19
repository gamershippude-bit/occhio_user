"""
Módulo de Gerenciamento de Banco de Dados para Cloud
Tabela: user_rec_facial (imgID, imgVetor, imgNome, imgData, imgRelacao, imgUser, imgLabel, avisar)
"""

import logging
import pickle
import numpy as np
import pymysql
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time
import threading

from db.config import get_cloud_db_config

logger = logging.getLogger("occhio.db.database")

SQL_SELECT_FACES = """
    SELECT imgID, imgVetor, imgNome, imgLabel, imgRelacao, imgUser, avisar, imgData
    FROM user_rec_facial
    ORDER BY imgID
"""

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

    def _parse_encoding(self, encoding_bytes, nome_ref: str) -> Optional[np.ndarray]:
        if not isinstance(encoding_bytes, bytes):
            logger.warning(f"Encoding inválido para {nome_ref}: não é bytes")
            return None
        try:
            encoding = pickle.loads(encoding_bytes)
            if encoding is None:
                return None
            arr = np.asarray(encoding, dtype=np.float64)
            if arr.shape != (128,):
                logger.warning(f"Encoding inválido para {nome_ref}: shape {arr.shape}")
                return None
            return arr
        except Exception as e:
            logger.warning(f"Erro ao decodificar vetor de {nome_ref}: {e}")
            return None

    @staticmethod
    def _nome_exibicao(img_nome: str, img_label: Optional[str]) -> str:
        nome = (img_nome or img_label or '').strip()
        return nome if nome else 'desconhecido'

    def _fetch_all_faces(self):
        cursor = self._get_cursor()
        cursor.execute(SQL_SELECT_FACES)
        resultados = cursor.fetchall()
        cursor.close()
        return resultados

    def load_face_encodings(self) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """Carrega imgVetor + imgNome de user_rec_facial para reconhecimento."""
        with self._lock:
            try:
                encodings, nomes, ids = [], [], []
                for row in self._fetch_all_faces():
                    img_id, encoding_bytes, img_nome, img_label, *_ = row
                    nome = self._nome_exibicao(img_nome, img_label)
                    if nome.lower() in ('desconhecido', 'unknown'):
                        continue
                    encoding = self._parse_encoding(encoding_bytes, nome)
                    if encoding is not None:
                        encodings.append(encoding)
                        nomes.append(nome)
                        ids.append(img_id)

                logger.info(f"📥 MySQL: {len(encodings)} rosto(s) carregado(s) — {nomes}")
                return encodings, nomes, ids

            except Exception as e:
                logger.error(f"❌ Erro ao carregar encodings: {e}")
                return [], [], []

    def get_faces_catalog(self) -> Dict[str, dict]:
        """Mapa nome → metadados (parentesco, avisar) para uso na IA."""
        with self._lock:
            try:
                catalog: Dict[str, dict] = {}
                for row in self._fetch_all_faces():
                    img_id, _, img_nome, img_label, relacao, img_user, avisar, img_data = row
                    nome = self._nome_exibicao(img_nome, img_label)
                    if nome.lower() in ('desconhecido', 'unknown'):
                        continue
                    chave = nome.lower()
                    catalog[chave] = {
                        'id': img_id,
                        'nome': nome,
                        'relacao': (relacao or 'conhecido').strip(),
                        'label': (img_label or nome).strip(),
                        'user': img_user or 0,
                        'avisar': bool(avisar),
                        'data_cadastro': img_data.isoformat() if img_data else None,
                    }
                return catalog
            except Exception as e:
                logger.error(f"❌ Erro ao montar catálogo de rostos: {e}")
                return {}

    def face_exists(self, face_encoding: np.ndarray, threshold: float = 0.6) -> bool:
        """Verifica se imgVetor já existe na tabela (evita cadastro duplicado)."""
        return self.find_existing_name(face_encoding, threshold) is not None

    def find_existing_name(self, face_encoding: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        with self._lock:
            try:
                import face_recognition
                query_vec = np.asarray(face_encoding, dtype=np.float64)
                for row in self._fetch_all_faces():
                    img_id, encoding_bytes, img_nome, img_label, *_ = row
                    nome = self._nome_exibicao(img_nome, img_label)
                    encoding_db = self._parse_encoding(encoding_bytes, nome)
                    if encoding_db is None:
                        continue
                    dist = float(face_recognition.face_distance([encoding_db], query_vec)[0])
                    if dist < threshold:
                        return nome
                return None
            except Exception as e:
                logger.error(f"❌ Erro ao verificar rosto cadastrado: {e}")
                return None

    def save_face(
        self,
        face_encoding: np.ndarray,
        nome: str,
        relacao: str = "conhecido",
        user: Optional[int] = None,
        label: Optional[str] = None,
        avisar: bool = True,
    ) -> bool:
        """Salva um rosto detectado no banco de dados."""
        with self._lock:
            try:
                encoding_bytes = pickle.dumps(face_encoding)
                sql = """
                INSERT INTO user_rec_facial 
                    (imgVetor, imgNome, imgData, imgRelacao, imgUser, imgLabel, avisar)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                valores = (
                    encoding_bytes,
                    nome,
                    datetime.now(),
                    relacao,
                    user if user is not None else 0,
                    label if label else nome,
                    1 if avisar else 0,
                )

                cursor = self._get_cursor()
                cursor.execute(sql, valores)
                self.conn.commit()
                cursor.close()

                logger.info(f"✅ Rosto salvo com sucesso: Nome='{nome}' avisar={avisar}")
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

    def list_faces(self) -> List[dict]:
        """Lista rostos cadastrados (sem imgVetor — só metadados)."""
        with self._lock:
            try:
                faces = []
                for row in self._fetch_all_faces():
                    img_id, _, img_nome, img_label, relacao, img_user, avisar, img_data = row
                    nome = self._nome_exibicao(img_nome, img_label)
                    faces.append({
                        "id": img_id,
                        "nome": nome,
                        "label": (img_label or nome).strip(),
                        "relacao": (relacao or 'conhecido').strip(),
                        "user": img_user or 0,
                        "avisar": bool(avisar),
                        "data_cadastro": img_data.isoformat() if img_data else None,
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

    def get_nomes_com_aviso(self) -> List[str]:
        """Nomes (imgNome) com avisar=1."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute(
                    "SELECT DISTINCT imgNome FROM user_rec_facial WHERE avisar = 1"
                )
                nomes = [row[0] for row in cursor.fetchall() if row[0]]
                cursor.close()
                return nomes
            except Exception as e:
                logger.error(f"❌ Erro ao listar avisos: {e}")
                return []

    def get_relacao(self, nome: str) -> Optional[str]:
        """Parentesco (imgRelacao) do rosto pelo imgNome."""
        with self._lock:
            try:
                cursor = self._get_cursor()
                cursor.execute(
                    """
                    SELECT imgRelacao FROM user_rec_facial
                    WHERE LOWER(imgNome) = LOWER(%s)
                    ORDER BY imgID DESC LIMIT 1
                    """,
                    (nome.strip(),),
                )
                row = cursor.fetchone()
                cursor.close()
                return row[0].strip() if row and row[0] else None
            except Exception as e:
                logger.error(f"❌ Erro ao buscar relação: {e}")
                return None

    def is_mysql(self) -> bool:
        return True