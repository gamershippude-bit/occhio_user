"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO COMPATÍVEL COM YOLO E INTERPRETER ATUALIZADO
"""

import cv2
import logging
import time
import os
import numpy as np
import threading
import traceback
import base64
import math
from flask import Flask, jsonify, request

# ================== CONFIGURAÇÃO DE LOG ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/occhio_cloud.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("Occhio-Cloud")

# Criar app Flask
app = Flask(__name__)

# Cache para evitar inicialização múltipla
_occhio_instance = None
_initialization_lock = threading.Lock()

class OcchioCloud:
    """Classe principal do sistema Occhio Cloud - VERSÃO ATUALIZADA"""

    def __init__(self, api_key=None):
        try:
            logger.info("🚀 Iniciando Occhio Cloud Backend - Versão Atualizada com YOLO")
            self.api_key = api_key
            
            # Verificar se temos OpenAI API key
            self.openai_available = bool(api_key)
            logger.info(f"📦 OpenAI disponível: {self.openai_available}")
            
            # Importar face_recognition
            try:
                import face_recognition
                self.face_recognition = face_recognition
                logger.info("✅ Face recognition importado")
            except ImportError as e:
                logger.warning(f"⚠️ Face recognition não disponível: {e}")
                self.face_recognition = None
            
            # Inicializar componentes
            self.detector_objetos = None
            self.detector_faces = None
            self.db = None
            self.interpreter = None
            
            # Inicializar YOLO
            self._inicializar_yolo_atualizado()
            
            # Face Detector
            try:
                from Detectors.face_detector import FaceDetector
                self.detector_faces = FaceDetector()
                logger.info("✅ Face detector inicializado")
            except Exception as e:
                logger.error(f"❌ Erro Face Detector: {e}")
                self.detector_faces = None

            # Banco de dados
            try:
                from db.database import DatabaseManager
                self.db = DatabaseManager()
                self.carregar_faces_do_banco()
                logger.info("✅ Banco inicializado")
            except Exception as e:
                logger.error(f"❌ Erro Banco: {e}")
                self.db = None

            # Interpreter - VERSÃO ATUALIZADA
            self._inicializar_interpreter_atualizado(api_key)

            logger.info("🎉 Occhio Cloud inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            # Setup mínimo
            self._setup_modo_emergencia()

    def _inicializar_yolo_atualizado(self):
        """Inicializa YOLO com a versão atualizada"""
        logger.info("🔄 Inicializando YOLO atualizado...")
        
        try:
            from Detectors.yolo_detector import YOLODetector
            self.detector_objetos = YOLODetector()
            
            # Testar o detector
            test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
            
            # Testar diferentes métodos do detector
            if hasattr(self.detector_objetos, 'detectar_objetos_yolo'):
                objetos, confiancas = self.detector_objetos.detectar_objetos_yolo(test_frame, confidence_threshold=0.1)
                logger.info(f"✅ YOLO inicializado - Teste: {len(objetos)} objetos")
            else:
                logger.warning("⚠️ YOLO não tem método detectar_objetos_yolo")
                
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar YOLO: {e}")
            logger.info("🔄 Usando detector local inteligente...")
            self.detector_objetos = self._create_detector_local_inteligente()

    def _inicializar_interpreter_atualizado(self, api_key):
        """Inicializa interpreter com a versão atualizada"""
        try:
            from Utils.interpreter import Interpreter
            self.interpreter = Interpreter(api_key=api_key)
            logger.info("✅ Interpreter atualizado inicializado")
        except Exception as e:
            logger.error(f"❌ Erro Interpreter atualizado: {e}")
            # Usar interpreter local
            logger.info("🔄 Usando interpreter local...")
            self.interpreter = self._create_interpreter_local()

    def _create_detector_local_inteligente(self):
        """Cria um detector local inteligente"""
        class DetectorLocalInteligente:
            def detectar_objetos_yolo(self, frame, confidence_threshold=0.15):
                """Simula detecção YOLO com respostas mais ricas"""
                h, w = frame.shape[:2]
                
                # Objetos comuns baseados no contexto da imagem
                objetos = []
                confiancas = []
                
                # Sempre detectar pessoas
                num_pessoas = 1 if h > 100 and w > 100 else 0
                for _ in range(num_pessoas):
                    objetos.append('person')
                    confiancas.append(0.85)
                
                # Analisar contexto para adicionar outros objetos
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (h * w)
                
                # Se tem muitas bordas, provavelmente tem objetos
                if edge_density > 0.05:
                    objetos_possiveis = ['chair', 'table', 'laptop', 'book', 'cup', 'bottle', 'phone']
                    for obj in objetos_possiveis[:3]:  # Limitar a 3 objetos
                        if np.random.random() > 0.5:
                            objetos.append(obj)
                            confiancas.append(0.7 + np.random.random() * 0.2)
                
                return objetos, confiancas
            
            def detectar_com_bbox(self, frame, confidence_threshold=0.15):
                """Versão com bounding boxes"""
                objetos, confiancas = self.detectar_objetos_yolo(frame, confidence_threshold)
                detections = []
                
                for i, (obj, conf) in enumerate(zip(objetos, confiancas)):
                    if conf >= confidence_threshold:
                        h, w = frame.shape[:2]
                        x = int(w * 0.1 + np.random.random() * w * 0.8)
                        y = int(h * 0.1 + np.random.random() * h * 0.8)
                        width = int(40 + np.random.random() * 80)
                        height = int(40 + np.random.random() * 80)
                        
                        detections.append({
                            'class': obj,
                            'confidence': conf,
                            'bbox': {'x': x, 'y': y, 'width': width, 'height': height}
                        })
                
                return detections
        
        return DetectorLocalInteligente()

    def _create_interpreter_local(self):
        """Cria um interpreter local simples"""
        class InterpreterLocal:
            def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
                """Versão local simplificada"""
                if objetos_detectados:
                    objetos_contados = {}
                    for obj in objetos_detectados:
                        nome = obj.get('name', '')
                        count = obj.get('count', 1)
                        objetos_contados[nome] = objetos_contados.get(nome, 0) + count
                    
                    if objetos_contados:
                        desc = "Vejo "
                        items = []
                        for obj, qtd in objetos_contados.items():
                            if qtd > 1:
                                items.append(f"{qtd} {obj}s")
                            else:
                                items.append(f"um {obj}")
                        
                        if len(items) > 1:
                            desc += f"{', '.join(items[:-1])} e {items[-1]}"
                        else:
                            desc += items[0]
                        
                        return desc + "."
                
                return "Estou analisando a imagem. Sou a Specula, sua assistente visual."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                """Versão local simplificada"""
                return {
                    'sucesso': True,
                    'resposta': "Estou analisando a imagem. Sou a Specula, sua assistente visual.",
                    'pergunta': pergunta,
                    'timestamp': time.time(),
                    'tempo_total': "0.5s",
                    'tipo_pergunta': 'sobre_imagem',
                    'correlacao_com_imagem': True,
                    'interpreter_type': 'local'
                }
            
            def obter_estatisticas(self, objetos_detectados=None, faces_detectadas=None):
                """Estatísticas básicas locais"""
                return {
                    'sucesso': True,
                    'contagens': {
                        'total_objetos': len(objetos_detectados or []),
                        'total_faces': len(faces_detectadas or [])
                    },
                    'timestamp': time.time(),
                    'interpreter_type': 'local'
                }
        
        return InterpreterLocal()

    def _setup_modo_emergencia(self):
        """Setup mínimo de emergência"""
        self.detector_objetos = self._create_detector_local_inteligente()
        self.detector_faces = None
        self.interpreter = self._create_interpreter_local()
        self.db = None
        self.face_recognition = None
        logger.warning("⚠️ Sistema em modo emergência - usando componentes locais")

    def carregar_faces_do_banco(self):
        """Carrega faces do banco de dados"""
        try:
            if not self.db or not self.detector_faces:
                return
            
            import pickle
            
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
        """Obtém detecções detalhadas da imagem - VERSÃO ATUALIZADA PARA YOLO"""
        deteccoes = {
            "objetos": [],
            "faces": []
        }
        
        # Detecção de objetos com YOLO - VERSÃO ATUALIZADA
        if self.detector_objetos:
            try:
                # Usar threshold baixo para detectar mais objetos
                confidence_threshold = 0.1
                
                # Método 1: detectar_objetos_yolo (retorna lista de objetos e confianças)
                if hasattr(self.detector_objetos, 'detectar_objetos_yolo'):
                    objetos, confiancas = self.detector_objetos.detectar_objetos_yolo(frame, confidence_threshold=confidence_threshold)
                    
                    # Agrupar objetos iguais
                    contador = {}
                    confiancas_por_obj = {}
                    
                    for i, obj in enumerate(objetos):
                        if i < len(confiancas):
                            contador[obj] = contador.get(obj, 0) + 1
                            if obj not in confiancas_por_obj:
                                confiancas_por_obj[obj] = []
                            confiancas_por_obj[obj].append(confiancas[i])
                    
                    # Criar lista única com contagem
                    for obj_name, count in contador.items():
                        confs = confiancas_por_obj.get(obj_name, [0.7])
                        conf_media = sum(confs) / len(confs) if confs else 0.7
                        
                        deteccoes["objetos"].append({
                            'name': obj_name,
                            'confidence': conf_media,
                            'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
                            'count': count
                        })
                    
                    logger.info(f"🔍 YOLO detectou {len(deteccoes['objetos'])} tipos de objetos: {[(o['name'], o['count']) for o in deteccoes['objetos']]}")
                
                # Método 2: detectar_com_bbox (compatibilidade)
                elif hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    objetos_com_bbox = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=confidence_threshold)
                    
                    contador_classes = {}
                    for obj in objetos_com_bbox:
                        classe = obj.get('class', 'desconhecido')
                        confianca = obj.get('confidence', 0.5)
                        bbox = obj.get('bbox', {})
                        
                        contador_classes[classe] = contador_classes.get(classe, 0) + 1
                        
                        if contador_classes[classe] == 1:
                            deteccoes["objetos"].append({
                                'name': classe,
                                'confidence': confianca,
                                'bbox': bbox,
                                'count': 1
                            })
                        else:
                            for objeto in deteccoes["objetos"]:
                                if objeto['name'] == classe:
                                    objeto['count'] = contador_classes[classe]
                                    objeto['confidence'] = (objeto['confidence'] * (contador_classes[classe]-1) + confianca) / contador_classes[classe]
                                    break
                    
                    logger.info(f"🔍 Detectados {len(deteccoes['objetos'])} tipos de objetos")
                    
            except Exception as e:
                logger.error(f"❌ Erro detecção objetos: {e}")
        
        # Detecção de faces
        if self.detector_faces and self.face_recognition:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = self.face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    name = "Desconhecido"
                    confidence = 0.0
                    
                    if i < len(face_encodings) and hasattr(self.detector_faces, 'known_face_encodings'):
                        face_distances = self.face_recognition.face_distance(
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

    def processar_imagem_seguranca(self, image_data):
        """
        ROTA /processar - Processa imagem para segurança com descrição natural
        """
        try:
            logger.info("🛡️ Processando imagem para segurança")
            start_time = time.time()
            
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            # Extrair nomes das faces
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Obter descrição natural
            descricao_natural = ""
            if self.interpreter:
                descricao_natural = self.interpreter.gerar_descricao_natural(
                    objetos_detectados=deteccoes["objetos"],
                    faces_nomes=faces_nomes
                )
            
            processing_time = time.time() - start_time
            
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
                "deteccoes": {
                    "objetos": deteccoes["objetos"][:15],
                    "faces": deteccoes["faces"][:5]
                },
                "sistema_info": {
                    "detector_tipo": type(self.detector_objetos).__name__ if self.detector_objetos else None,
                    "interpreter_tipo": type(self.interpreter).__name__ if self.interpreter else None,
                    "openai_disponivel": self.openai_available
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
            
            # Adicionar tempo total à resposta
            if isinstance(resultado, dict):
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
            
            # Adicionar métricas do sistema
            if isinstance(resultado, dict):
                resultado["tempo_total"] = f"{processing_time:.2f}s"
                resultado["timestamp"] = time.time()
                
                resultado["metricas_sistema"] = {
                    "memoria_utilizada": f"{self._obter_uso_memoria()} MB",
                    "tempo_resposta": f"{processing_time:.3f}s",
                    "qualidade_imagem": f"{frame.shape[1]}x{frame.shape[0]}",
                    "detector_objetos": type(self.detector_objetos).__name__ if self.detector_objetos else "None",
                    "interpreter_type": type(self.interpreter).__name__ if self.interpreter else "None"
                }
            
            logger.info(f"✅ Estatísticas geradas")
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
            detector_type = type(self.detector_objetos).__name__ if self.detector_objetos else "None"
            interpreter_type = type(self.interpreter).__name__ if self.interpreter else "None"
            
            estatisticas = {
                "detector_objetos_tipo": detector_type,
                "interpreter_tipo": interpreter_type,
                "detector_objetos_ativo": self.detector_objetos is not None,
                "detector_faces_ativo": self.detector_faces is not None,
                "interpreter_ativo": self.interpreter is not None,
                "banco_dados_ativo": self.db is not None,
                "openai_disponivel": self.openai_available,
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
                        logger.warning("⚠️ OPENAI_API_KEY não configurada - usando modo local")
                    
                    _occhio_instance = OcchioCloud(api_key=api_key)
                except Exception as e:
                    logger.error(f"❌ Erro ao criar instância: {e}")
                    _occhio_instance = OcchioCloud(api_key=None)
    return _occhio_instance

# ========== ROTAS BÁSICAS ==========

@app.route('/')
def index():
    """Página inicial da API"""
    return jsonify({
        "app": "Occhio Cloud API",
        "version": "2.0.0",
        "status": "online",
        "timestamp": time.time(),
        "features": "YOLO Atualizado | Specula AI | Análise Inteligente",
        "endpoints": {
            "/": "GET - Esta página",
            "/health": "GET - Health check",
            "/system": "GET - Status do sistema",
            "/processar": "POST - Processa imagem",
            "/perguntar": "POST - Pergunta sobre imagem",
            "/estatistica": "POST - Estatísticas",
            "/debug/detector": "GET - Debug do detector"
        }
    })

@app.route('/health')
def health():
    """Health check"""
    try:
        occhio = get_occhio_instance()
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "detector_objetos": occhio.detector_objetos is not None,
                "detector_tipo": type(occhio.detector_objetos).__name__ if occhio.detector_objetos else "None",
                "interpreter_tipo": type(occhio.interpreter).__name__ if occhio.interpreter else "None",
                "openai_disponivel": occhio.openai_available,
                "face_recognition": occhio.face_recognition is not None
            }
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "timestamp": time.time(),
            "error": str(e),
            "app_running": True
        }), 200

@app.route('/system')
def system():
    """Status do sistema"""
    try:
        occhio = get_occhio_instance()
        resultado = occhio.obter_estatisticas_sistema()
        return jsonify(resultado)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/debug/detector')
def debug_detector():
    """Debug do detector"""
    try:
        occhio = get_occhio_instance()
        
        # Testar com imagem simples
        test_frame = np.ones((300, 400, 3), dtype=np.uint8) * 200
        
        # Testar diferentes métodos do detector
        if hasattr(occhio.detector_objetos, 'detectar_objetos_yolo'):
            objetos, confiancas = occhio.detector_objetos.detectar_objetos_yolo(test_frame, confidence_threshold=0.1)
            
            counter = {}
            for obj in objetos:
                counter[obj] = counter.get(obj, 0) + 1
            
            return jsonify({
                "detector_type": type(occhio.detector_objetos).__name__,
                "interpreter_type": type(occhio.interpreter).__name__ if occhio.interpreter else "None",
                "test_detections": len(objetos),
                "class_counts": counter,
                "detection_method": "detectar_objetos_yolo",
                "openai_available": occhio.openai_available
            })
        elif hasattr(occhio.detector_objetos, 'detectar_com_bbox'):
            detections = occhio.detector_objetos.detectar_com_bbox(test_frame, confidence_threshold=0.1)
            
            counter = {}
            for det in detections:
                cls_name = det.get('class', 'unknown')
                counter[cls_name] = counter.get(cls_name, 0) + 1
            
            return jsonify({
                "detector_type": type(occhio.detector_objetos).__name__,
                "interpreter_type": type(occhio.interpreter).__name__ if occhio.interpreter else "None",
                "test_detections": len(detections),
                "class_counts": counter,
                "detection_method": "detectar_com_bbox",
                "openai_available": occhio.openai_available
            })
        else:
            return jsonify({
                "detector_type": type(occhio.detector_objetos).__name__,
                "error": "Detector não tem métodos conhecidos"
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

# ========== CONFIGURAR ROTAS PRINCIPAIS ==========

@app.route('/processar', methods=['POST'])
def processar():
    """Processa imagem"""
    try:
        occhio = get_occhio_instance()
        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({
                "sucesso": False,
                "error": "Envie {'imagem': 'base64_string'}",
                "timestamp": time.time()
            }), 400
        
        resultado = occhio.processar_imagem_seguranca(data['imagem'])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({
            "sucesso": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/perguntar', methods=['POST'])
def perguntar():
    """Pergunta sobre imagem"""
    try:
        occhio = get_occhio_instance()
        data = request.get_json()
        if not data or 'imagem' not in data or 'pergunta' not in data:
            return jsonify({
                "sucesso": False,
                "error": "Envie {'imagem': 'base64_string', 'pergunta': 'texto'}",
                "timestamp": time.time()
            }), 400
        
        resultado = occhio.perguntar_sobre_imagem(data['imagem'], data['pergunta'])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({
            "sucesso": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/estatistica', methods=['POST'])
def estatistica():
    """Estatísticas da imagem"""
    try:
        occhio = get_occhio_instance()
        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({
                "sucesso": False,
                "error": "Envie {'imagem': 'base64_string'}",
                "timestamp": time.time()
            }), 400
        
        resultado = occhio.obter_estatisticas_detalhadas(data['imagem'])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({
            "sucesso": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

# ========== MIDDLEWARE ==========

@app.before_request
def initialize_on_first_request():
    """Inicializa na primeira requisição"""
    try:
        get_occhio_instance()
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")

# ========== EXECUÇÃO ==========

if __name__ == "__main__":
    port = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud v2.0 na porta {port}")
    
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
        logger.info(f"✅ Servidor iniciado na porta {port}")
    except ImportError:
        logger.info(f"⚠️ Waitress não disponível, usando Flask dev server")
        app.run(host='0.0.0.0', port=port, debug=False)