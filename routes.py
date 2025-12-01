"""
routes.py - Configuração de rotas da API Occhio
Agora só contém as rotas /processar, /perguntar, /estatistica
As rotas /, /health, /system já estão no main.py
"""

import time
import logging
from flask import request, jsonify

logger = logging.getLogger("Occhio-Cloud-Routes")

def configure_routes(app, get_occhio_instance_func):
    """
    Configura APENAS as rotas específicas que precisam da lógica de negócio
    """
    
    @app.route('/processar', methods=['POST'])
    def processar():
        """Endpoint /processar"""
        try:
            data = request.get_json()
            if not data or 'imagem' not in data:
                return jsonify({
                    "sucesso": False,
                    "error": "Dados inválidos. Envie {'imagem': 'base64_string'}",
                    "timestamp": time.time()
                }), 400
            
            occhio = get_occhio_instance_func()
            resultado = occhio.processar_imagem_seguranca(data['imagem'])
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro em /processar: {e}")
            return jsonify({
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }), 500

    @app.route('/perguntar', methods=['POST'])
    def perguntar():
        """Endpoint /perguntar"""
        try:
            data = request.get_json()
            if not data or 'imagem' not in data or 'pergunta' not in data:
                return jsonify({
                    "sucesso": False,
                    "error": "Dados inválidos. Envie {'imagem': 'base64_string', 'pergunta': 'texto'}",
                    "timestamp": time.time()
                }), 400
            
            occhio = get_occhio_instance_func()
            resultado = occhio.perguntar_sobre_imagem(data['imagem'], data['pergunta'])
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro em /perguntar: {e}")
            return jsonify({
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }), 500

    @app.route('/estatistica', methods=['POST'])
    def estatistica():
        """Endpoint /estatistica"""
        try:
            data = request.get_json()
            if not data or 'imagem' not in data:
                return jsonify({
                    "sucesso": False,
                    "error": "Dados inválidos. Envie {'imagem': 'base64_string'}",
                    "timestamp": time.time()
                }), 400
            
            occhio = get_occhio_instance_func()
            resultado = occhio.obter_estatisticas_detalhadas(data['imagem'])
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro em /estatistica: {e}")
            return jsonify({
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }), 500

    logger.info("✅ Rotas específicas configuradas: /processar, /perguntar, /estatistica")
    return app