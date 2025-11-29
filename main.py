"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO CLOUD/API - Endpoints completos para processamento de imagens
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

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de saúde MUITO simples"""
    return jsonify({
        "status": "healthy", 
        "service": "Occhio Cloud",
        "timestamp": time.time()
    })

@app.route('/health-simple', methods=['GET'])
def health_check_simple():
    """Endpoint de saúde SUPER simples"""
    return "OK", 200

@app.route('/ready', methods=['GET'])
def ready_check():
    """Endpoint de readiness - verifica se sistema está inicializado"""
    global _occhio_instance
    if _occhio_instance is not None:
        return jsonify({"status": "ready", "initialized": True})
    else:
        return jsonify({"status": "initializing", "initialized": False}), 503

class OcchioCloud:
    """Classe otimizada para cloud - apenas processamento de imagens via API"""

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

            # Interpreter - COM A CORREÇÃO
            try:
                self.interpreter = Interpreter(api_key=api_key)
                logger.info("✅ Interpreter OK - Versão Corrigida")
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

    def processar_analise_rapida(self, image_data):
        """
        FASE 1: Processamento rápido para análise textual
        """
        try:
            frame = self._decode_image(image_data)
            deteccoes = self._processar_deteccoes_rapidas(frame)
            
            # Gerar resposta textual
            resposta_texto = self._gerar_resposta_textual_rapida(deteccoes)
            
            return {
                "success": True,
                "resposta": resposta_texto,
                "deteccoes_basicas": {
                    "total_faces": deteccoes["total_faces"],
                    "total_objetos": deteccoes["total_objetos"],
                    "faces_conhecidas": deteccoes["faces_conhecidas"]
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro análise rápida: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def processar_deteccoes_detalhadas(self, image_data):
        """
        FASE 2: Processamento detalhado para coordenadas
        """
        try:
            frame = self._decode_image(image_data)
            coordenadas = self._obter_coordenadas_detalhadas(frame)
            
            return {
                "success": True,
                "coordenadas": coordenadas,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro detecções detalhadas: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def processar_completo(self, image_data):
        """
        Processamento completo: análise rápida + coordenadas
        """
        try:
            frame = self._decode_image(image_data)
            
            # Processar ambas as fases
            deteccoes_rapidas = self._processar_deteccoes_rapidas(frame)
            coordenadas = self._obter_coordenadas_detalhadas(frame)
            resposta_texto = self._gerar_resposta_textual_rapida(deteccoes_rapidas)
            
            return {
                "success": True,
                "analise_rapida": {
                    "resposta": resposta_texto,
                    "deteccoes_basicas": {
                        "total_faces": deteccoes_rapidas["total_faces"],
                        "total_objetos": deteccoes_rapidas["total_objetos"],
                        "faces_conhecidas": deteccoes_rapidas["faces_conhecidas"]
                    }
                },
                "deteccoes_detalhadas": {
                    "coordenadas": coordenadas
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro processamento completo: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def responder_pergunta(self, image_data, pergunta):
        """
        Responde uma pergunta específica sobre a imagem usando o interpreter CORRIGIDO
        """
        try:
            frame = self._decode_image(image_data)
            
            # Obter detecções detalhadas para o interpreter
            deteccoes_rapidas = self._processar_deteccoes_rapidas(frame)
            coordenadas = self._obter_coordenadas_detalhadas(frame)
            
            # Preparar dados para o interpreter CORRIGIDO
            faces_nomes = []
            objetos_detectados = []
            
            # Extrair nomes das faces
            for face in coordenadas.get("faces", []):
                if face["nome"] != "Desconhecido":
                    faces_nomes.append(face["nome"])
                else:
                    faces_nomes.append("Desconhecido")
            
            # Extrair objetos (usando YOLO)
            if self.detector_objetos:
                objetos_detectados, _ = self.detector_objetos.detectar_objetos_rapido(frame)
            
            # Usar interpreter CORRIGIDO para resposta inteligente
            if self.interpreter:
                resposta = self.interpreter.responder_pergunta(pergunta, objetos_detectados, faces_nomes)
            else:
                # Fallback para resposta básica
                resposta = self._gerar_resposta_textual_rapida(deteccoes_rapidas)
            
            return {
                "success": True,
                "pergunta": pergunta,
                "resposta": resposta,
                "deteccoes": {
                    "faces": len(faces_nomes),
                    "objetos": len(objetos_detectados),
                    "faces_conhecidas": deteccoes_rapidas["faces_conhecidas"]
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta: {e}")
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

    def _processar_deteccoes_rapidas(self, frame):
        """Processa detecções de forma otimizada para velocidade"""
        deteccoes = {
            "total_faces": 0,
            "total_objetos": 0,
            "faces_conhecidas": 0
        }
        
        # Detecção rápida de faces
        if self.detector_faces and hasattr(self.detector_faces, 'known_face_encodings'):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=0)
                
                conhecidos = 0
                if face_locations and self.detector_faces.known_face_encodings:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        face_distances = face_recognition.face_distance(
                            self.detector_faces.known_face_encodings, face_encoding
                        )
                        
                        if len(face_distances) > 0:
                            best_distance = np.min(face_distances)
                            if best_distance <= 0.6:
                                conhecidos += 1
                
                deteccoes["total_faces"] = len(face_locations)
                deteccoes["faces_conhecidas"] = conhecidos
                
            except Exception as e:
                logger.error(f"❌ Erro detecção rápida faces: {e}")

        # Detecção rápida de objetos
        if self.detector_objetos:
            try:
                objetos_detectados, _ = self.detector_objetos.detectar_objetos_rapido(frame)
                deteccoes["total_objetos"] = len(objetos_detectados)
            except Exception as e:
                logger.error(f"❌ Erro detecção rápida objetos: {e}")

        return deteccoes

    def _obter_coordenadas_detalhadas(self, frame):
        """Obtém coordenadas detalhadas para desenhar caixas"""
        coordenadas = {
            "faces": [],
            "objetos": []
        }
        
        # Coordenadas das faces
        if self.detector_faces and hasattr(self.detector_faces, 'known_face_encodings'):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    name = "Desconhecido"
                    confidence = 0.0
                    
                    if i < len(face_encodings) and self.detector_faces.known_face_encodings:
                        face_distances = face_recognition.face_distance(
                            self.detector_faces.known_face_encodings, face_encodings[i]
                        )
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            best_distance = face_distances[best_match_index]
                            confidence = max(0, 1 - best_distance)
                            
                            if best_distance <= 0.6:
                                name = self.detector_faces.known_face_names[best_match_index]
                    
                    coordenadas["faces"].append({
                        "nome": name,
                        "confianca": float(confidence),
                        "caixa": {
                            "x1": int(left),
                            "y1": int(top),
                            "x2": int(right),
                            "y2": int(bottom)
                        }
                    })
                    
            except Exception as e:
                logger.error(f"❌ Erro coordenadas faces: {e}")

        # Coordenadas dos objetos
        if self.detector_objetos:
            try:
                objetos_coords = self.detector_objetos.obter_coordenadas_objetos(frame)
                coordenadas["objetos"] = objetos_coords
            except Exception as e:
                logger.error(f"❌ Erro coordenadas objetos: {e}")

        return coordenadas

    def _gerar_resposta_textual_rapida(self, deteccoes):
        """Gera resposta textual rápida"""
        total_faces = deteccoes["total_faces"]
        total_objetos = deteccoes["total_objetos"]
        faces_conhecidas = deteccoes["faces_conhecidas"]
        
        if total_faces == 0 and total_objetos == 0:
            return "Não detectei pessoas ou objetos significativos no ambiente."
        
        partes = []
        
        if total_faces > 0:
            if faces_conhecidas > 0:
                partes.append(f"{faces_conhecidas} pessoa(s) conhecida(s)")
            desconhecidos = total_faces - faces_conhecidas
            if desconhecidos > 0:
                partes.append(f"{desconhecidos} pessoa(s) não identificada(s)")
        
        if total_objetos > 0:
            partes.append(f"{total_objetos} objeto(s) detectado(s)")
        
        return "Vejo " + ", ".join(partes) + "."

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

# ================== ENDPOINTS ==================

@app.route('/perguntar', methods=['POST'])
def perguntar():
    """
    Endpoint para fazer pergunta sobre uma imagem
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
        
        resultado = occhio.responder_pergunta(image_data, pergunta)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /perguntar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/processar', methods=['POST'])
def processar():
    """
    Processa imagem e retorna análise completa
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        resultado = occhio.processar_completo(image_data)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /processar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analise-rapida', methods=['POST'])
def analise_rapida():
    """
    FASE 1: Retorna análise textual rapidamente
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        resultado = occhio.processar_analise_rapida(image_data)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /analise-rapida: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/deteccoes-detalhadas', methods=['POST'])
def deteccoes_detalhadas():
    """
    FASE 2: Retorna coordenadas para desenhar caixas
    """
    try:
        occhio = get_occhio_instance()
        
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({"success": False, "error": "Nenhuma imagem fornecida"}), 400
        
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            image_data = request.json['image_data']
        
        resultado = occhio.processar_deteccoes_detalhadas(image_data)
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /deteccoes-detalhadas: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/estatisticas', methods=['GET'])
def estatisticas():
    """
    Retorna estatísticas do sistema
    """
    try:
        occhio = get_occhio_instance()
        resultado = occhio.obter_estatisticas_sistema()
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /estatisticas: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/completo', methods=['POST'])
def completo():
    """
    Endpoint que engloba tudo: análise rápida + coordenadas + resposta inteligente CORRIGIDA
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
        
        pergunta = request.json.get('pergunta', 'Descreva o que você vê nesta imagem')
        
        # Processar tudo
        resultado_completo = occhio.processar_completo(image_data)
        if not resultado_completo['success']:
            return jsonify(resultado_completo)
        
        # Usar interpreter CORRIGIDO para resposta inteligente
        resposta_inteligente = resultado_completo['analise_rapida']['resposta']
        if occhio.interpreter:
            frame = occhio._decode_image(image_data)
            deteccoes_rapidas = occhio._processar_deteccoes_rapidas(frame)
            coordenadas = occhio._obter_coordenadas_detalhadas(frame)
            
            faces_nomes = [face["nome"] for face in coordenadas.get("faces", [])]
            objetos_detectados, _ = occhio.detector_objetos.detectar_objetos_rapido(frame) if occhio.detector_objetos else ([], {})
            
            # USANDO O INTERPRETER CORRIGIDO
            resposta_inteligente = occhio.interpreter.responder_pergunta(pergunta, objetos_detectados, faces_nomes)
        
        return jsonify({
            "success": True,
            "pergunta": pergunta,
            "resposta_inteligente": resposta_inteligente,
            "analise_rapida": resultado_completo['analise_rapida'],
            "deteccoes_detalhadas": resultado_completo['deteccoes_detalhadas'],
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /completo: {e}")
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