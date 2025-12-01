"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO CLOUD/API - Arquivo principal
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
from flask import Flask

# Utils
from Detectors.yolo_detector import YOLODetector
from Detectors.face_detector import FaceDetector
from db.database import DatabaseManager
from Utils.interpreter import Interpreter

# Importar rotas
from api import *

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
    """Classe principal do sistema Occhio Cloud"""

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

            # Interpreter - Nova versão otimizada
            try:
                self.interpreter = Interpreter(api_key=api_key)
                logger.info("✅ Interpreter OK - Versão Otimizada")
            except Exception as e:
                logger.error(f"❌ Erro Interpreter: {e}")
                self.interpreter = None

            logger.info("🎉 Occhio Cloud inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            raise

    def carregar_faces_do_banco(self):
        """Carrega faces do banco de dados"""
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
        """Obtém detecções detalhadas da imagem"""
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

    # ========== MÉTODOS PARA AS ROTAS PRINCIPAIS ==========

    def processar_imagem_seguranca(self, image_data):
        """
        ROTA /processar - Para análise periódica de segurança
        Retorna identificações + descrição natural com IA generativa
        """
        try:
            logger.info("🛡️ Processando imagem para segurança")
            start_time = time.time()
            
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            # Extrair nomes das faces
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Obter descrição natural usando IA generativa
            descricao_natural = ""
            if self.interpreter:
                descricao_natural = self.interpreter.gerar_descricao_natural(
                    objetos_detectados=deteccoes["objetos"],
                    faces_nomes=faces_nomes
                )
            
            processing_time = time.time() - start_time
            
            # Resposta com IA generativa para descrição natural
            resposta = {
                "sucesso": True,
                "timestamp": time.time(),
                "tempo_processamento": f"{processing_time:.2f}s",
                "tipo": "analise_seguranca",
                "descricao_natural": descricao_natural,
                "resumo": {
                    "total_objetos": len(deteccoes["objetos"]),
                    "total_faces": len(deteccoes["faces"]),
                    "faces_conhecidas": len([f for f in deteccoes["faces"] if f['name'] != 'Desconhecido']),
                    "faces_desconhecidas": len([f for f in deteccoes["faces"] if f['name'] == 'Desconhecido'])
                },
                "alertas": self._gerar_alertas_seguranca(deteccoes),
                "deteccoes": {
                    "objetos": deteccoes["objetos"][:10],  # Limitar para performance
                    "faces": deteccoes["faces"][:5]        # Limitar para performance
                }
            }
            
            logger.info(f"✅ Segurança: {resposta['resumo']['total_objetos']} objetos, {resposta['resumo']['total_faces']} faces")
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro processar_imagem_seguranca: {e}")
            return {
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def perguntar_sobre_imagem(self, image_data, pergunta):
        """
        ROTA /perguntar - Para chat com IA sobre a imagem
        Usa OpenAI para conversa contextual
        """
        try:
            logger.info(f"💬 Processando pergunta: '{pergunta}'")
            start_time = time.time()
            
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "sucesso": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Extrair nomes das faces para o interpreter
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Usar interpreter para resposta contextual
            resultado = self.interpreter.perguntar_sobre_imagem(
                pergunta=pergunta,
                objetos_detectados=deteccoes["objetos"],
                faces_nomes=faces_nomes
            )
            
            processing_time = time.time() - start_time
            resultado["tempo_total"] = f"{processing_time:.2f}s"
            
            logger.info(f"✅ Pergunta respondida - Tempo: {processing_time:.2f}s")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro perguntar_sobre_imagem: {e}")
            return {
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def obter_estatisticas_detalhadas(self, image_data):
        """
        ROTA /estatistica - Para dados técnicos e logs
        Retorna métricas detalhadas para análise
        """
        try:
            logger.info("📊 Gerando estatísticas detalhadas")
            start_time = time.time()
            
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "sucesso": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            # Usar interpreter para estatísticas detalhadas
            resultado = self.interpreter.obter_estatisticas(
                objetos_detectados=deteccoes["objetos"],
                faces_detectadas=deteccoes["faces"]
            )
            
            processing_time = time.time() - start_time
            resultado["tempo_total"] = f"{processing_time:.2f}s"
            resultado["timestamp"] = time.time()
            
            # Adicionar métricas do sistema
            resultado["metricas_sistema"] = {
                "memoria_utilizada": f"{self._obter_uso_memoria()} MB",
                "tempo_resposta": f"{processing_time:.3f}s",
                "qualidade_imagem": f"{frame.shape[1]}x{frame.shape[0]}"
            }
            
            logger.info(f"✅ Estatísticas geradas - {resultado['contagens']['total_objetos']} objetos")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro obter_estatisticas_detalhadas: {e}")
            return {
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def _gerar_alertas_seguranca(self, deteccoes):
        """Gera alertas de segurança baseados nas detecções"""
        alertas = []
        
        # Alertas baseados em faces desconhecidas
        faces_desconhecidas = [f for f in deteccoes["faces"] if f['name'] == 'Desconhecido']
        if faces_desconhecidas:
            alertas.append({
                "tipo": "face_desconhecida",
                "nivel": "medio",
                "mensagem": f"{len(faces_desconhecidas)} face(s) desconhecida(s) detectada(s)",
                "quantidade": len(faces_desconhecidas)
            })
        
        # Alertas baseados em muitos objetos
        if len(deteccoes["objetos"]) > 20:
            alertas.append({
                "tipo": "muitos_objetos",
                "nivel": "baixo", 
                "mensagem": "Muitos objetos detectados no ambiente",
                "quantidade": len(deteccoes["objetos"])
            })
        
        return alertas

    def _obter_uso_memoria(self):
        """Obtém uso de memória aproximado"""
        try:
            import psutil
            process = psutil.Process()
            return f"{process.memory_info().rss / 1024 / 1024:.1f}"
        except:
            return "N/A"

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