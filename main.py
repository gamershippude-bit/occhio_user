"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO BACKEND/CLOUD - Remove interface visual, adiciona endpoints API
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
import io
from PIL import Image

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

class OcchioCloud:
    """Classe principal do sistema de visão computacional para cloud"""

    def __init__(self, api_key=None):
        try:
            logger.info("🚀 Iniciando Occhio Cloud Backend")
            self.api_key = api_key
            
            # === DETECÇÕES ===
            self.face_names_atual = []
            self.objetos_atual = []
            self.contagem_acumulada = {}
            
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

            # Interpreter
            try:
                self.interpreter = Interpreter(api_key=api_key)
                logger.info("✅ Interpreter OK")
            except Exception as e:
                logger.error(f"❌ Erro Interpreter: {e}")
                self.interpreter = None

            logger.info("🎉 Occhio Cloud inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            raise

    def carregar_faces_do_banco(self):
        """Carrega faces do banco com validação melhorada"""
        try:
            logger.info("📂 Carregando faces do banco...")
            
            if not self.db or not self.detector_faces:
                logger.error("❌ Banco ou detector não inicializado")
                return
            
            conn = self.db.conn
            cursor = conn.cursor()
            cursor.execute("SELECT imgVetor, imgNome FROM user_rec_facial")
            resultados = cursor.fetchall()
            
            known_face_encodings = []
            known_face_names = []
            
            logger.info(f"📊 Encontrados {len(resultados)} registros no banco")
            
            for encoding_data, nome in resultados:
                nome_str = str(nome).strip()
                
                if (encoding_data and nome_str and 
                    nome_str.lower() not in ['desconhecido', 'unknown', '']):
                    
                    try:
                        if isinstance(encoding_data, (bytes, bytearray)):
                            encoding_list = pickle.loads(encoding_data)
                        
                        if encoding_list and isinstance(encoding_list, list):
                            encoding_array = np.array(encoding_list)
                            if encoding_array.shape == (128,):
                                valido, mensagem = self.detector_faces.verificar_encoding_qualidade(encoding_array)
                                if valido:
                                    known_face_encodings.append(encoding_array)
                                    known_face_names.append(nome_str)
                                    logger.info(f"✅ Encoding carregado: '{nome_str}' - {mensagem}")
                                else:
                                    logger.warning(f"⚠️ Encoding ignorado ('{nome_str}'): {mensagem}")
                        
                    except Exception as e:
                        logger.error(f"❌ Erro ao carregar encoding ('{nome_str}'): {e}")
            
            cursor.close()
            
            if known_face_encodings:
                self.detector_faces.carregar_encodings(known_face_encodings, known_face_names)
                nomes_unicos = list(set(known_face_names))
                logger.info(f"🎉 {len(known_face_encodings)} encodings válidos carregados")
                logger.info(f"📝 Pessoas cadastradas: {nomes_unicos}")
            else:
                logger.warning("⚠️ Nenhum encoding válido encontrado no banco")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar faces do banco: {e}")

    def _decode_image(self, image_data):
        """Decodifica imagem de base64 ou bytes para numpy array"""
        try:
            if isinstance(image_data, str):
                # Se for base64
                if image_data.startswith('data:image'):
                    # Remover header data:image/...;base64,
                    image_data = image_data.split(',')[1]
                image_data = base64.b64decode(image_data)
            
            if isinstance(image_data, bytes):
                # Converter bytes para numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = image_data
            
            if frame is None:
                raise ValueError("Não foi possível decodificar a imagem")
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar imagem: {e}")
            raise

    def processar_imagem(self, image_data):
        """
        Processa imagem e retorna detecções
        """
        try:
            frame = self._decode_image(image_data)
            
            # Processar detecções
            deteccoes = self._processar_deteccoes(frame)
            
            return {
                "success": True,
                "deteccoes": deteccoes,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar imagem: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def _processar_deteccoes(self, frame):
        """
        Processa todas as detecções no frame
        """
        deteccoes = {
            "faces": [],
            "objetos": [],
            "contagem_objetos": {},
            "estatisticas": {}
        }
        
        # Detecção de Faces
        if self.detector_faces:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    name = "Desconhecido"
                    confidence = 0.0
                    
                    if self.detector_faces.known_face_encodings:
                        face_distances = face_recognition.face_distance(
                            self.detector_faces.known_face_encodings, 
                            face_encoding
                        )
                        
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        confidence = max(0, 1 - best_distance)
                        
                        if best_distance <= self.detector_faces.tolerance and confidence >= self.detector_faces.min_confidence:
                            name = self.detector_faces.known_face_names[best_match_index]
                    
                    face_names.append({
                        "nome": name,
                        "confianca": float(confidence)
                    })
                
                deteccoes["faces"] = face_names
                logger.info(f"👥 Faces detectadas: {len(face_names)}")
                
            except Exception as e:
                logger.error(f"❌ Erro detecção faces: {e}")

        # Detecção de Objetos YOLO
        if self.detector_objetos:
            try:
                objetos_detectados, contagem_acumulada = self.detector_objetos.detectar_objetos_rapido(frame)
                
                deteccoes["objetos"] = objetos_detectados
                deteccoes["contagem_objetos"] = contagem_acumulada
                
                logger.info(f"📦 Objetos detectados: {len(objetos_detectados)}")
                
            except Exception as e:
                logger.error(f"❌ Erro detecção objetos: {e}")

        # Estatísticas
        deteccoes["estatisticas"] = {
            "total_faces": len(deteccoes["faces"]),
            "total_objetos": len(deteccoes["objetos"]),
            "faces_conhecidas": len([f for f in deteccoes["faces"] if f["nome"] != "Desconhecido"]),
            "faces_desconhecidas": len([f for f in deteccoes["faces"] if f["nome"] == "Desconhecido"])
        }

        return deteccoes

    def responder_pergunta(self, pergunta, deteccoes):
        """
        Responde pergunta baseada nas detecções
        """
        try:
            if not self.interpreter:
                return "Sistema temporariamente indisponível"
            
            # Extrair dados das detecções
            faces_nomes = [face["nome"] for face in deteccoes["faces"]]
            objetos_detectados = deteccoes["objetos"]
            
            resposta = self.interpreter.responder_pergunta(pergunta, objetos_detectados, faces_nomes)
            
            logger.info(f"💬 Pergunta: '{pergunta}' -> Resposta: '{resposta}'")
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta: {e}")
            return f"Erro ao processar pergunta: {str(e)}"

# Instância global do Occhio Cloud
occhio_cloud = None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de saúde da API"""
    return jsonify({
        "status": "healthy",
        "service": "Occhio Cloud",
        "timestamp": time.time()
    })

@app.route('/processar', methods=['POST'])
def processar_imagem():
    """
    Endpoint para processar imagem e retornar detecções
    """
    try:
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({
                "success": False,
                "error": "Nenhuma imagem fornecida. Use 'image' (form-data) ou 'image_data' (JSON base64)"
            }), 400
        
        # Obter dados da imagem
        if 'image' in request.files:
            image_file = request.files['image']
            image_data = image_file.read()
        else:
            image_data = request.json['image_data']
        
        # Processar imagem
        resultado = occhio_cloud.processar_imagem(image_data)
        
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /processar: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/perguntar', methods=['POST'])
def perguntar():
    """
    Endpoint para fazer pergunta sobre uma imagem
    """
    try:
        data = request.json
        
        if 'pergunta' not in data:
            return jsonify({
                "success": False,
                "error": "Pergunta não fornecida"
            }), 400
        
        pergunta = data['pergunta']
        image_data = data.get('image_data')
        deteccoes = data.get('deteccoes')
        
        # Se não forneceu detecções, mas forneceu imagem, processar primeiro
        if not deteccoes and image_data:
            resultado_processamento = occhio_cloud.processar_imagem(image_data)
            if not resultado_processamento['success']:
                return jsonify(resultado_processamento)
            deteccoes = resultado_processamento['deteccoes']
        
        if not deteccoes:
            return jsonify({
                "success": False,
                "error": "Forneça image_data ou deteccoes"
            }), 400
        
        # Responder pergunta
        resposta = occhio_cloud.responder_pergunta(pergunta, deteccoes)
        
        return jsonify({
            "success": True,
            "pergunta": pergunta,
            "resposta": resposta,
            "deteccoes": deteccoes,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Erro endpoint /perguntar: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/deteccoes/estatisticas', methods=['GET'])
def get_estatisticas():
    """Retorna estatísticas do sistema"""
    estatisticas = {
        "faces_cadastradas": len(occhio_cloud.detector_faces.known_face_names) if occhio_cloud.detector_faces else 0,
        "detector_objetos_ativo": occhio_cloud.detector_objetos is not None,
        "detector_faces_ativo": occhio_cloud.detector_faces is not None,
        "interpreter_ativo": occhio_cloud.interpreter is not None,
        "timestamp": time.time()
    }
    
    return jsonify(estatisticas)

@app.route('/faces', methods=['GET'])
def listar_faces():
    """Lista todas as faces cadastradas"""
    try:
        if not occhio_cloud.db:
            return jsonify({
                "success": False,
                "error": "Banco de dados não disponível"
            }), 500
            
        cursor = occhio_cloud.db.conn.cursor()
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
        
        return jsonify({
            "success": True,
            "faces": faces,
            "total": len(faces)
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao listar faces: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def iniciar_servidor(host='0.0.0.0', port=5000, api_key=None):
    """Inicia o servidor Flask"""
    global occhio_cloud
    
    try:
        # Inicializar Occhio Cloud
        occhio_cloud = OcchioCloud(api_key=api_key)
        
        logger.info(f"🌐 Iniciando servidor Occhio Cloud em {host}:{port}")
        app.run(host=host, port=port, debug=False)
        
    except Exception as e:
        logger.error(f"💥 ERRO AO INICIAR SERVIDOR: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Occhio Cloud Backend")
    parser.add_argument("--api_key", type=str, required=True, help="API Key da OpenAI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host do servidor")
    parser.add_argument("--port", type=int, default=5000, help="Porta do servidor")
    
    args = parser.parse_args()
    
    iniciar_servidor(host=args.host, port=args.port, api_key=args.api_key)#   D e p l o y   t r i g g e r   -   1 1 / 2 5 / 2 0 2 5   2 0 : 3 8 : 2 8  
    
 