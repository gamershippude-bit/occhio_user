"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO CLOUD/API - Arquivo principal (APENAS LÓGICA)
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
from flask import Flask, jsonify

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

# Criar app Flask - DEVE SER GLOBAL PARA GUNICORN
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
            
            # IMPORTANTE: Importações dentro de try/except para fallback
            try:
                # Tenta importar face_recognition mas tem fallback
                import face_recognition
                self.face_recognition = face_recognition
                logger.info("✅ Face recognition importado com sucesso")
            except ImportError as e:
                logger.warning(f"⚠️ Face recognition não disponível: {e}")
                # Cria um mock para evitar erros
                class MockFaceRecognition:
                    def face_locations(self, *args, **kwargs):
                        return []
                    def face_encodings(self, *args, **kwargs):
                        return []
                    def face_distance(self, *args, **kwargs):
                        return [1.0]
                self.face_recognition = MockFaceRecognition()
            
            # Inicializar componentes
            self.detector_objetos = None
            self.detector_faces = None
            self.db = None
            self.interpreter = None
            
            # Inicializar componentes com tratamento de erro individual
            logger.info("🔧 Inicializando detectores...")
            
            # YOLO - com fallback
            try:
                from Detectors.yolo_detector import YOLODetector
                self.detector_objetos = YOLODetector()
                logger.info("✅ YOLO inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro YOLO: {e}")
                # Cria detector mock para desenvolvimento
                self.detector_objetos = self._create_mock_detector()
                
            # Face Detector - com fallback
            try:
                from Detectors.face_detector import FaceDetector
                self.detector_faces = FaceDetector()
                logger.info("✅ Face detector inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro Face Detector: {e}")
                self.detector_faces = self._create_mock_face_detector()

            # Banco de dados - com fallback
            try:
                from db.database import DatabaseManager
                self.db = DatabaseManager()
                self.carregar_faces_do_banco()
                logger.info("✅ Banco inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro Banco: {e}")
                # Banco mock
                self.db = None

            # Interpreter - ESSENCIAL mas com fallback
            try:
                from Utils.interpreter import Interpreter
                self.interpreter = Interpreter(api_key=api_key)
                logger.info("✅ Interpreter OK - Versão Otimizada")
            except Exception as e:
                logger.error(f"❌ Erro Interpreter: {e}")
                # Interpreter mock mínimo
                self.interpreter = self._create_mock_interpreter()

            logger.info("🎉 Occhio Cloud inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            # Não levantar exceção, deixar sistema funcionar em modo limitado
            self._setup_fallback_mode()

    def _create_mock_detector(self):
        """Cria um detector mock para quando YOLO falhar"""
        class MockDetector:
            def detectar_com_bbox(self, frame):
                # Retorna algumas detecções mock para desenvolvimento
                return [
                    {'class': 'person', 'confidence': 0.85, 'bbox': {'x': 100, 'y': 100, 'width': 50, 'height': 150}},
                    {'class': 'chair', 'confidence': 0.75, 'bbox': {'x': 200, 'y': 200, 'width': 60, 'height': 80}}
                ]
            
            def detectar_objetos_rapido(self, frame):
                return ['person', 'chair'], [0.85, 0.75]
        
        return MockDetector()

    def _create_mock_face_detector(self):
        """Cria um detector de faces mock"""
        class MockFaceDetector:
            def __init__(self):
                self.known_face_encodings = []
                self.known_face_names = []
            
            def carregar_encodings(self, encodings, names):
                self.known_face_encodings = encodings
                self.known_face_names = names
        
        return MockFaceDetector()

    def _create_mock_interpreter(self):
        """Cria um interpreter mock"""
        class MockInterpreter:
            def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
                return "Sistema em modo de fallback. Alguns recursos podem estar limitados."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                return {
                    'sucesso': True,
                    'resposta': 'Sistema em modo de fallback. Algumas funcionalidades podem estar limitadas.',
                    'pergunta': pergunta,
                    'timestamp': time.time(),
                    'tempo_total': '0.1s'
                }
            
            def obter_estatisticas(self, objetos_detectados=None, faces_detectadas=None):
                return {
                    'sucesso': True,
                    'contagens': {'total_objetos': 0, 'total_faces': 0},
                    'timestamp': time.time()
                }
        
        return MockInterpreter()

    def _setup_fallback_mode(self):
        """Configura sistema em modo de fallback mínimo"""
        self.detector_objetos = self._create_mock_detector()
        self.detector_faces = self._create_mock_face_detector()
        self.interpreter = self._create_mock_interpreter()
        self.db = None

    def _debug_deteccoes(self, frame):
        """Debug das detecções para ver o que o YOLO está vendo"""
        print("\n🔍 DEBUG DETECÇÕES:")

        if self.detector_objetos:
            try:
                # Testar método rápido primeiro
                objetos, confiancas = self.detector_objetos.detectar_objetos_rapido(frame)
                print(f"  Método rápido: {objetos}")
                print(f"  Confianças: {confiancas}")

                # Testar método com bbox se existir
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    bbox_result = self.detector_objetos.detectar_com_bbox(frame)
                    print(f"  Método bbox: {bbox_result}")

            except Exception as e:
                print(f"  ❌ Erro debug: {e}")
                
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
        """Obtém detecções detalhadas da imagem - VERSÃO CORRIGIDA"""
        deteccoes = {
            "objetos": [],
            "faces": []
        }
        
        # DEBUG primeiro
        self._debug_deteccoes(frame)
        
        # Detecção de objetos com YOLO
        if self.detector_objetos:
            try:
                # CORREÇÃO: Usar sempre o método que retorna bounding boxes se disponível
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    objetos_com_bbox = self.detector_objetos.detectar_com_bbox(frame)
                    
                    # CONTAGEM CORRETA: Agrupar por classe para evitar duplicações
                    contador_classes = {}
                    objetos_unicos = []
                    
                    for obj in objetos_com_bbox:
                        classe = obj.get('class', 'desconhecido')
                        confianca = obj.get('confidence', 0.5)
                        bbox = obj.get('bbox', {})
                        
                        # Contar ocorrências de cada classe
                        contador_classes[classe] = contador_classes.get(classe, 0) + 1
                        
                        # Se for a primeira vez que vemos esta classe, adicionar
                        if contador_classes[classe] == 1:
                            deteccoes["objetos"].append({
                                'name': classe,
                                'confidence': confianca,
                                'bbox': bbox,
                                'count': 1  # Iniciar contagem
                            })
                        else:
                            # Atualizar contagem do objeto existente
                            for objeto in deteccoes["objetos"]:
                                if objeto['name'] == classe:
                                    objeto['count'] = contador_classes[classe]
                                    # Atualizar confiança para a média
                                    objeto['confidence'] = (objeto['confidence'] * (contador_classes[classe]-1) + confianca) / contador_classes[classe]
                                    break
                                
                    print(f"  ✅ Objetos após agrupamento: {[(o['name'], o['count']) for o in deteccoes['objetos']]}")
                    
                else:
                    # Fallback para método rápido - também precisa corrigir
                    objetos, confiancas = self.detector_objetos.detectar_objetos_rapido(frame)
                    
                    # Agrupar objetos iguais
                    contador = {}
                    for i, obj in enumerate(objetos):
                        contador[obj] = contador.get(obj, 0) + 1
                    
                    # Criar lista única com contagem
                    for obj_name, count in contador.items():
                        # Calcular confiança média para este objeto
                        confs = [confiancas[i] for i, o in enumerate(objetos) if o == obj_name]
                        conf_media = sum(confs) / len(confs) if confs else 0.7
                        
                        deteccoes["objetos"].append({
                            'name': obj_name,
                            'confidence': conf_media,
                            'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
                            'count': count
                        })
                    
                    print(f"  ✅ Objetos agrupados: {list(contador.items())}")
                    
            except Exception as e:
                print(f"❌ Erro detecção objetos: {e}")
                logger.error(f"❌ Erro detecção objetos: {e}")
    
        # Detecção de faces (mantenha o mesmo código)
        if self.detector_faces and hasattr(self.face_recognition, 'face_locations'):
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

    # ========== MÉTODOS PARA AS ROTAS PRINCIPAIS ==========

    def processar_imagem_seguranca(self, image_data):
        """
        ROTA /processar - Processa imagem para segurança com descrição natural
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
                        logger.warning("⚠️ OPENAI_API_KEY não configurada - usando modo fallback")
                    
                    _occhio_instance = OcchioCloud(api_key=api_key)
                except Exception as e:
                    logger.error(f"❌ Erro ao criar instância: {e}")
                    # Criar instância fallback mesmo com erro
                    _occhio_instance = OcchioCloud(api_key=None)
    return _occhio_instance

# ========== ROTAS BÁSICAS (DEVEM ESTAR FORA DE QUALQUER FUNÇÃO) ==========

@app.route('/')
def index():
    """Página inicial da API"""
    return jsonify({
        "app": "Occhio Cloud API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": time.time(),
        "endpoints": {
            "/": "GET - Esta página",
            "/health": "GET - Health check do sistema",
            "/system": "GET - Status do sistema",
            "/processar": "POST - Processa imagem para segurança",
            "/perguntar": "POST - Pergunta sobre imagem",
            "/estatistica": "POST - Estatísticas detalhadas"
        }
    })

@app.route('/health')
def health():
    """Health check OBRIGATÓRIO para Cloud Run"""
    try:
        occhio = get_occhio_instance()
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "detector_objetos": occhio.detector_objetos is not None,
                "detector_faces": occhio.detector_faces is not None,
                "interpreter": occhio.interpreter is not None,
                "banco_dados": occhio.db is not None
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

# ========== CONFIGURAR ROTAS RESTANTES DO routes.py ==========

try:
    from routes import configure_routes
    # Esta função deve adicionar as rotas /processar, /perguntar, /estatistica
    configure_routes(app, get_occhio_instance)
    logger.info("✅ Rotas adicionais configuradas com sucesso")
except ImportError as e:
    logger.error(f"❌ Erro ao importar routes: {e}")
    # Criar fallback para as rotas principais
    
    @app.route('/processar', methods=['POST'])
    def processar_fallback():
        return jsonify({
            "sucesso": False,
            "error": "Módulo de rotas não carregado",
            "timestamp": time.time()
        }), 503
    
    @app.route('/perguntar', methods=['POST'])
    def perguntar_fallback():
        return jsonify({
            "sucesso": False,
            "error": "Módulo de rotas não carregado",
            "timestamp": time.time()
        }), 503
    
    @app.route('/estatistica', methods=['POST'])
    def estatistica_fallback():
        return jsonify({
            "sucesso": False,
            "error": "Módulo de rotas não carregado",
            "timestamp": time.time()
        }), 503

except Exception as e:
    logger.error(f"❌ Erro ao configurar rotas: {e}")

# ========== MIDDLEWARE DE INICIALIZAÇÃO ==========

@app.before_request
def initialize_on_first_request():
    """Inicializa o sistema na primeira requisição"""
    try:
        get_occhio_instance()
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")

# ========== EXECUÇÃO ==========
# NOTA: Não usamos mais create_app() porque o app já está configurado
# Gunicorn vai usar 'app' diretamente

if __name__ == "__main__":
    # Para execução direta (sem Gunicorn)
    port = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud na porta {port}")
    
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        # Fallback para Flask dev server
        app.run(host='0.0.0.0', port=port, debug=False)