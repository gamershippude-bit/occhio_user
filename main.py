"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO CLOUD/API - Endpoints completos baseados no novo interpreter
"""

import cv2
import logging
import time
import os
import numpy as np
import threading
import face_recognition
import pickle
import traceback
import base64

# Flask
from flask import Flask, request, jsonify

# Utils
from Detectors.yolo_detector import YOLODetector
from Detectors.face_detector import FaceDetector
from db.database import DatabaseManager
from Utils.interpreter import Interpreter

# ================== CONFIGURAÇÃO DE LOG ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('occhio_cloud.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("Occhio-Cloud")

app = Flask(__name__)

# Cache para evitar inicialização múltipla
_occhio_instance = None
_initialization_lock = threading.Lock()

class OcchioCloud:
    """Classe otimizada para cloud - com novos endpoints baseados no interpreter"""

    def __init__(self, api_key=None):
        try:
            logger.info("🚀 Iniciando Occhio Cloud Backend")
            self.api_key = api_key
            
            # Inicializar componentes
            logger.info("🔧 Inicializando detectores...")
            
            # YOLO
            try:
                self.detector_objetos = YOLODetector()
                logger.info("✅ YOLO inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro YOLO: {e}")
                self.detector_objetos = None
                
            # Face Detector
            try:
                self.detector_faces = FaceDetector()
                logger.info("✅ Face detector inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro Face Detector: {e}")
                self.detector_faces = None

            # Banco de dados
            try:
                self.db = DatabaseManager()
                self.carregar_faces_do_banco()
                logger.info("✅ Banco inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro Banco: {e}")
                self.db = None

            # Interpreter - NOVA VERSÃO
            try:
                self.interpreter = Interpreter(api_key=api_key)
                logger.info("✅ Interpreter OK - Nova Versão com Endpoints")
            except Exception as e:
                logger.error(f"❌ Erro Interpreter: {e}")
                self.interpreter = None

            logger.info("🎉 Occhio Cloud inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            raise

    def carregar_faces_do_banco(self):
        """Carrega faces do banco - versão simplificada"""
        try:
            if not self.db or not self.detector_faces:
                return
            
            conn = self.db.conn
            cursor = conn.cursor()
            cursor.execute("SELECT imgVetor, imgNome FROM user_rec_facial")
            resultados = cursor.fetchall()
            
            known_face_encodings = []
            known_face_names = []
            
            for encoding_data, nome in resultados:
                nome_str = str(nome).strip()
                
                if (encoding_data and nome_str and 
                    nome_str.lower() not in ['desconhecido', 'unknown', '']):
                    
                    try:
                        if isinstance(encoding_data, (bytes, bytearray)):
                            encoding_list = pickle.loads(encoding_data)
                            encoding_array = np.array(encoding_list)
                            
                            if encoding_array.shape == (128,) and np.any(encoding_array):
                                known_face_encodings.append(encoding_array)
                                known_face_names.append(nome_str)
                    except Exception:
                        continue
            
            cursor.close()
            
            if known_face_encodings:
                self.detector_faces.carregar_encodings(known_face_encodings, known_face_names)
                logger.info(f"✅ {len(known_face_encodings)} encodings carregados")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar faces: {e}")

    def _decode_image(self, image_data):
        """Decodifica imagem de forma robusta"""
        try:
            # Se for string (base64)
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
            # Se já for bytes
            elif isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                raise ValueError(f"Tipo de imagem não suportado: {type(image_data)}")
            
            if frame is None:
                raise ValueError("Falha ao decodificar imagem")
            
            # Redimensionar se for muito grande (otimização)
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                scale = min(1280/w, 720/h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar imagem: {e}")
            raise

    def _obter_deteccoes_detalhadas(self, frame):
        """Obtém detecções detalhadas para os novos endpoints"""
        deteccoes = {
            "objetos": [],
            "faces": []
        }
        
        # Detecção de objetos com YOLO
        if self.detector_objetos:
            try:
                # Usar método que retorna bounding boxes
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    objetos_com_bbox = self.detector_objetos.detectar_com_bbox(frame)
                    for obj in objetos_com_bbox:
                        deteccoes["objetos"].append({
                            'name': obj.get('class', 'desconhecido'),
                            'confidence': obj.get('confidence', 0.5),
                            'bbox': obj.get('bbox', {})
                        })
                else:
                    # Fallback para método rápido
                    objetos, _ = self.detector_objetos.detectar_objetos_rapido(frame)
                    for obj in objetos:
                        deteccoes["objetos"].append({
                            'name': obj,
                            'confidence': 0.7,
                            'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100}
                        })
            except Exception as e:
                logger.error(f"❌ Erro detecção objetos: {e}")

        # Detecção de faces
        if self.detector_faces:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    name = "Desconhecido"
                    confidence = 0.0
                    
                    if i < len(face_encodings) and hasattr(self.detector_faces, 'known_face_encodings'):
                        face_distances = face_recognition.face_distance(
                            self.detector_faces.known_face_encodings, face_encodings[i]
                        )
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            best_distance = face_distances[best_match_index]
                            confidence = max(0, 1 - best_distance)
                            
                            if best_distance <= 0.6:
                                name = self.detector_faces.known_face_names[best_match_index]
                    
                    deteccoes["faces"].append({
                        'name': name,
                        'confidence': float(confidence),
                        'bbox': {
                            'x': int(left),
                            'y': int(top),
                            'width': int(right - left),
                            'height': int(bottom - top)
                        }
                    })
                    
            except Exception as e:
                logger.error(f"❌ Erro detecção faces: {e}")

        return deteccoes

    # ========== NOVOS ENDPOINTS BASEADOS NO INTERPRETER ==========

    def endpoint_processar(self, image_data):
        """
        Endpoint /processar - Processa imagem e retorna detecções com coordenadas
        """
        try:
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "success": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Usar o novo método do interpreter
            resultado = self.interpreter.processar_deteccoes(
                objetos_detectados=deteccoes["objetos"],
                faces_detectadas=deteccoes["faces"]
            )
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint_processar: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def endpoint_perguntar(self, image_data, pergunta):
        """
        Endpoint /perguntar - Responde pergunta com correlação com imagem
        """
        try:
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "success": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Extrair nomes das faces para o interpreter
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Usar o novo método do interpreter
            resultado = self.interpreter.perguntar_sobre_imagem(
                pergunta=pergunta,
                objetos_detectados=deteccoes["objetos"],
                faces_nomes=faces_nomes
            )
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint_perguntar: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def endpoint_estatistica(self, image_data):
        """
        Endpoint /estatistica - Retorna estatísticas detalhadas
        """
        try:
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "success": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Usar o novo método do interpreter
            resultado = self.interpreter.obter_estatisticas(
                objetos_detectados=deteccoes["objetos"],
                faces_detectadas=deteccoes["faces"]
            )
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint_estatistica: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def endpoint_completo(self, image_data, pergunta=None):
        """
        Endpoint /completo - Junta todos os processamentos
        """
        try:
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "success": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Extrair nomes das faces
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Usar o novo método do interpreter
            resultado = self.interpreter.processamento_completo(
                objetos_detectados=deteccoes["objetos"],
                faces_detectadas=deteccoes["faces"],
                pergunta=pergunta
            )
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro endpoint_completo: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def obter_estatisticas_sistema(self):
        """Retorna estatísticas do sistema"""
        try:
            estatisticas = {
                "faces_cadastradas": len(self.detector_faces.known_face_names) if self.detector_faces else 0,
                "detector_objetos_ativo": self.detector_objetos is not None,
                "detector_faces_ativo": self.detector_faces is not None,
                "interpreter_ativo": self.interpreter is not None,
                "banco_dados_ativo": self.db is not None,
                "timestamp": time.time()
            }
            
            return {
                "success": True,
                "estatisticas": estatisticas
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Singleton para a instância
def get_occhio_instance():
    global _occhio_instance
    if _occhio_instance is None:
        with _initialization_lock:
            if _occhio_instance is None:
                try:
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY não configurada")
                    
                    _occhio_instance = OcchioCloud(api_key=api_key)
                except Exception as e:
                    logger.error(f"❌ Erro ao criar instância: {e}")
                    raise
    return _occhio_instance

@app.before_request
def initialize_occhio():
    """Inicializa o Occhio Cloud na primeira requisição"""
    try:
        get_occhio_instance()
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")

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
        occhio = get_occhio_instance()
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
    global _occhio_instance
    if _occhio_instance is not None:
        return jsonify({"status": "ready", "initialized": True})
    else:
        return jsonify({"status": "initializing", "initialized": False}), 503

# ================== NOVOS ENDPOINTS BASEADOS NO INTERPRETER ==================

@app.route('/processar', methods=['POST'])
def processar():
    """
    Endpoint /processar - Processa imagem e retorna detecções com coordenadas
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        resultado = occhio.endpoint_processar(image_data)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /processar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/perguntar', methods=['POST'])
def perguntar():
    """
    Endpoint /perguntar - Responde pergunta com correlação com imagem
    """
    try:
        occhio = get_occhio_instance()
        
        data = request.json
        if 'pergunta' not in data:
            return jsonify({"success": False, "error": "Pergunta não fornecida"}), 400
        
        pergunta = data['pergunta']
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"success": False, "error": "Forneça image_data"}), 400
        
        logger.info(f"❓ Nova pergunta: '{pergunta}'")
        resultado = occhio.endpoint_perguntar(image_data, pergunta)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /perguntar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/estatistica', methods=['POST'])
def estatistica():
    """
    Endpoint /estatistica - Retorna estatísticas detalhadas da imagem
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        resultado = occhio.endpoint_estatistica(image_data)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /estatistica: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/completo', methods=['POST'])
def completo():
    """
    Endpoint /completo - Junta todos os processamentos
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        # Obter dados
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        pergunta = request.json.get('pergunta')
        
        logger.info(f"🎯 Endpoint COMPLETO - Pergunta: {pergunta}")
        
        resultado = occhio.endpoint_completo(image_data, pergunta)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /completo: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/estatisticas-sistema', methods=['GET'])
def estatisticas_sistema():
    """
    Retorna estatísticas do sistema (legado - mantido para compatibilidade)
    """
    try:
        occhio = get_occhio_instance()
        resultado = occhio.obter_estatisticas_sistema()
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /estatisticas-sistema: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ================== ENDPOINTS DE COMPATIBILIDADE (opcionais) ==================

@app.route('/analise-rapida', methods=['POST'])
def analise_rapida():
    """Endpoint legado para compatibilidade"""
    try:
        # Redirecionar para /perguntar com pergunta padrão
        data = request.json if request.json else {}
        data['pergunta'] = "Descreva o que você vê nesta imagem"
        
        # Criar uma requisição fake para o endpoint perguntar
        from flask import copy_current_request_context
        with copy_current_request_context():
            return perguntar()
            
    except Exception as e:
        logger.error(f"❌ Erro endpoint /analise-rapida: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def iniciar_servidor():
    """Inicia o servidor para nuvem"""
    port = int(os.getenv('PORT', '8080'))
    
    # Verificar API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("❌ OPENAI_API_KEY não configurada")
        exit(1)
    
    try:
        # Tentar usar waitress para produção
        from waitress import serve
        logger.info(f"🚀 Iniciando Occhio Cloud na porta {port}")
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        # Fallback para Flask dev server
        logger.info(f"🚀 Iniciando Occhio Cloud (Flask) na porta {port}")
        app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    iniciar_servidor()