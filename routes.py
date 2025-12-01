"""
routes.py - Configuração de rotas da API Occhio
"""

import time
import logging
from flask import request, jsonify

logger = logging.getLogger("Occhio-Cloud")

def configure_routes(app, get_occhio_instance_func):
    """
    Configura todas as rotas no app Flask
    """
    
    # ================== ENDPOINTS PRINCIPAIS ==================

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check simples"""
        return jsonify({
            "service": "Occhio Cloud",
            "status": "healthy", 
            "timestamp": time.time()
        })

    @app.route('/health-completo', methods=['GET'])
    def health_completo():
        """Health check completo do sistema"""
        try:
            occhio = get_occhio_instance_func()
            resultado = occhio.obter_estatisticas_sistema()
            return jsonify(resultado)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "status": "unhealthy"
            }), 500

    @app.route('/ready', methods=['GET'])
    def ready_check():
        """Endpoint de readiness"""
        from main import _occhio_instance
        if _occhio_instance is not None:
            return jsonify({"status": "ready", "initialized": True})
        else:
            return jsonify({"status": "initializing", "initialized": False}), 503

    # ================== ENDPOINTS ATUALIZADOS ==================

    @app.route('/processar', methods=['POST'])
    def processar():
        """
        Endpoint /processar - Processa imagem para segurança com descrição natural
        """
        try:
            occhio = get_occhio_instance_func()
            
            if 'image' not in request.files and 'image_data' not in request.json:
                return jsonify({"sucesso": False, "error": "Nenhuma imagem fornecida"}), 400
            
            if 'image' in request.files:
                image_data = request.files['image'].read()
            else:
                image_data = request.json['image_data']
            
            logger.info("🛡️ Processando imagem para segurança (/processar)")
            resultado = occhio.processar_imagem_seguranca(image_data)
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint /processar: {e}")
            return jsonify({"sucesso": False, "error": str(e)}), 500

    @app.route('/perguntar', methods=['POST'])
    def perguntar():
        """
        Endpoint /perguntar - Responde pergunta sobre a imagem
        """
        try:
            occhio = get_occhio_instance_func()
            
            data = request.json
            if 'pergunta' not in data:
                return jsonify({"sucesso": False, "error": "Pergunta não fornecida"}), 400
            
            pergunta = data['pergunta']
            image_data = data.get('image_data')
            
            if not image_data:
                return jsonify({"sucesso": False, "error": "Forneça image_data"}), 400
            
            logger.info(f"💬 Nova pergunta: '{pergunta}'")
            resultado = occhio.perguntar_sobre_imagem(image_data, pergunta)
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint /perguntar: {e}")
            return jsonify({"sucesso": False, "error": str(e)}), 500

    @app.route('/estatistica', methods=['POST'])
    def estatistica():
        """
        Endpoint /estatistica - Retorna estatísticas técnicas
        """
        try:
            occhio = get_occhio_instance_func()
            
            if 'image' not in request.files and 'image_data' not in request.json:
                return jsonify({"sucesso": False, "error": "Nenhuma imagem fornecida"}), 400
            
            if 'image' in request.files:
                image_data = request.files['image'].read()
            else:
                image_data = request.json['image_data']
            
            logger.info("📊 Gerando estatísticas técnicas (/estatistica)")
            resultado = occhio.obter_estatisticas_detalhadas(image_data)
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint /estatistica: {e}")
            return jsonify({"sucesso": False, "error": str(e)}), 500

    @app.route('/estatisticas-sistema', methods=['GET'])
    def estatisticas_sistema():
        """
        Retorna estatísticas do sistema
        """
        try:
            occhio = get_occhio_instance_func()
            resultado = occhio.obter_estatisticas_sistema()
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint /estatisticas-sistema: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # ================== ENDPOINTS DE COMPATIBILIDADE ==================

    @app.route('/analise-rapida', methods=['POST'])
    def analise_rapida():
        """Endpoint legado para compatibilidade"""
        try:
            # Redirecionar para /perguntar com pergunta padrão
            data = request.json if request.json else {}
            if 'image_data' not in data:
                return jsonify({"sucesso": False, "error": "Nenhuma imagem fornecida"}), 400
            
            data['pergunta'] = "Descreva o que você vê nesta imagem"
            
            # Criar uma requisição fake para o endpoint perguntar
            request._cached_json = data
            return perguntar()
                
        except Exception as e:
            logger.error(f"❌ Erro endpoint /analise-rapida: {e}")
            return jsonify({"sucesso": False, "error": str(e)}), 500

    # Middleware para inicialização
    @app.before_request
    def initialize_occhio():
        """Inicializa o Occhio Cloud na primeira requisição"""
        try:
            get_occhio_instance_func()
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
    
    logger.info("✅ Rotas configuradas com sucesso")
    return app