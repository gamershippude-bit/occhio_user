"""
Configurações do Banco de Dados
Este módulo contém as configurações de conexão com o banco de dados MySQL.
"""

# Configurações do banco de dados MySQL
db_config = {
    "host": "localhost",      # Endereço do servidor MySQL
    "user": "root",          # Nome de usuário
    "password": "master",    # Senha do usuário
    "database": "tcc_db",    # Nome do banco de dados
    "charset": "utf8mb4"     # Codificação de caracteres
}

# Configurações adicionais podem ser adicionadas aqui
# Exemplo:
# - Timeout de conexão
# - Pool de conexões
# - Configurações de SSL 