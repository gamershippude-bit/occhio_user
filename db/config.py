"""
Configurações do Banco de Dados (Produção/Cloud)

Configurações para ambiente de produção na nuvem.
"""

from typing import Dict
import os

def get_test_db_config() -> Dict[str, str]:
    """
    Retorna as configurações de conexão para o banco MySQL.
    Usa variáveis de ambiente para segurança.
    """
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "master"),
        "database": os.getenv("DB_NAME", "tcc_db"),
        "charset": "utf8mb4",
        "port": int(os.getenv("DB_PORT", "3306")),
        "connect_timeout": 30,
        "autocommit": True
    }