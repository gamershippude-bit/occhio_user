"""
Módulo de Gerenciamento de Banco de Dados
Este módulo implementa as operações de banco de dados para:
- Conexão com o banco MySQL
- Carregamento de encodings faciais
- Salvamento de rostos detectados
- Geração de relatórios PDF
"""

import pymysql
import os
import cv2
from datetime import datetime
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

def conectar_db():
    """Estabelece conexão com o banco de dados"""
    try:
        # Conecta ao MySQL
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='master',
            database='tcc_db'
        )
        
        cursor = conn.cursor()
        logger.info("Conexão com banco de dados estabelecida")
        return conn, cursor
        
    except Exception as e:
        logger.error(f"Erro ao conectar ao banco de dados: {e}")
        return None, None

def carregar_encodings_existentes(cursor):
    """Carrega encodings de rostos existentes no banco de dados"""
    try:
        cursor.execute("SELECT imgVetor, imgNome FROM user_rec_facial")
        resultados = cursor.fetchall()
        
        encodings = []
        nomes = []
        
        for encoding_bytes, nome in resultados:
            try:
                # Verifica se o encoding_bytes é válido
                if not isinstance(encoding_bytes, bytes):
                    logger.warning(f"Encoding inválido para {nome}: não é bytes")
                    continue
                    
                encoding = pickle.loads(encoding_bytes)
                if encoding is not None:
                    encodings.append(encoding)
                    nomes.append(nome)
                else:
                    logger.warning(f"Encoding nulo para {nome}")
            except Exception as e:
                logger.warning(f"Erro ao carregar encoding para {nome}: {e}")
                continue
            
        logger.info(f"Carregados {len(encodings)} encodings de faces")
        return encodings, nomes
        
    except Exception as e:
        logger.error(f"Erro ao carregar encodings: {e}")
        return [], []

def salvar_rosto(face_id, face_encoding, nome, cursor, conn):
    """Salva um rosto detectado no banco de dados"""
    try:
        # Codifica o encoding para salvar no banco
        encoding_bytes = pickle.dumps(face_encoding)
        
        # Salva no banco de dados
        cursor.execute(
            "INSERT INTO user_rec_facial (imgVetor, imgNome, imgData, imgRelacao) VALUES (%s, %s, %s, %s)",
            (encoding_bytes, nome, datetime.now(), "desconhecido")  # Adicionando valor padrão para imgRelacao
        )
        conn.commit()
        logger.info(f"Rosto salvo com sucesso: {face_id}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar rosto: {e}")
        if conn:
            conn.rollback()
        return False

def carregar_rostos_salvos(cursor):
    """
    Carrega todos os rostos salvos do banco de dados.
    
    Args:
        cursor: Cursor do banco de dados
        
    Returns:
        list: Lista de rostos salvos
    """
    try:
        cursor.execute("SELECT imgVetor FROM user_rec_facial")
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Erro ao carregar rostos salvos: {e}")
        return []

def salvar_rosto_no_banco(cursor, conn, img_blob, img_nome, img_relacao, img_data, img_user):
    """
    Salva um rosto diretamente no banco de dados.
    
    Args:
        cursor: Cursor do banco de dados
        conn: Conexão com o banco
        img_blob: Imagem em formato blob
        img_nome: Nome do rosto
        img_relacao: Relação com o usuário
        img_data: Data de captura
        img_user: ID do usuário
    """
    try:
        sql = """
        INSERT INTO user_rec_facial (imgVetor, imgNome, imgRelacao, imgData, imgUser)
        VALUES (%s, %s, %s, %s, %s)
        """
        valores = (img_blob, img_nome, img_relacao, img_data, img_user)
        cursor.execute(sql, valores)
        conn.commit()
        logger.info(f"Rosto {img_nome} salvo no banco de dados")
    except Exception as e:
        logger.error(f"Erro ao salvar rosto no banco: {e}")
        if conn:
            conn.rollback()
