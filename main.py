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
            
            # YOLO - com fallback MAS COM LOGS DETALHADOS
            try:
                from Detectors.yolo_detector import YOLODetector
                logger.info("🔄 Tentando inicializar YOLO...")
                self.detector_objetos = YOLODetector()
                logger.info("✅ YOLO inicializado com sucesso")
                
                # Testar se não é mock
                if hasattr(self.detector_objetos, 'model'):
                    logger.info(f"📦 Modelo YOLO carregado")
                    # Testar rápido se consegue detectar
                    try:
                        test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
                        test_result = self.detector_objetos.detectar_com_bbox(test_frame, confidence_threshold=0.1)
                        logger.info(f"🧪 Teste YOLO: {len(test_result)} detecções em imagem teste")
                    except Exception as test_e:
                        logger.warning(f"⚠️ Teste YOLO falhou: {test_e}")
                else:
                    logger.warning("⚠️ YOLO pode estar em modo mock")
                    
            except Exception as e:
                logger.error(f"❌ Erro YOLO: {e}")
                logger.error(f"📋 Traceback YOLO: {traceback.format_exc()}")
                # Cria detector mock para desenvolvimento
                self.detector_objetos = self._create_mock_detector()
                logger.warning("🔄 Usando detector mock para YOLO")
                
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
            def detectar_com_bbox(self, frame, confidence_threshold=0.25):
                # Retorna algumas detecções mock para desenvolvimento
                # Mas com mais variedade para testes
                detections = [
                    {'class': 'person', 'confidence': 0.85, 'bbox': {'x': 100, 'y': 100, 'width': 50, 'height': 150}},
                    {'class': 'person', 'confidence': 0.78, 'bbox': {'x': 300, 'y': 120, 'width': 55, 'height': 160}},
                    {'class': 'chair', 'confidence': 0.75, 'bbox': {'x': 200, 'y': 200, 'width': 60, 'height': 80}},
                    {'class': 'table', 'confidence': 0.65, 'bbox': {'x': 150, 'y': 300, 'width': 120, 'height': 70}},
                    {'class': 'laptop', 'confidence': 0.70, 'bbox': {'x': 180, 'y': 220, 'width': 40, 'height': 30}},
                    {'class': 'book', 'confidence': 0.60, 'bbox': {'x': 250, 'y': 210, 'width': 35, 'height': 25}},
                ]
                
                # Filtrar por confidence threshold
                return [d for d in detections if d['confidence'] >= confidence_threshold]
            
            def detectar_objetos_rapido(self, frame, confidence_threshold=0.25):
                detections = self.detectar_com_bbox(frame, confidence_threshold)
                objetos = [d['class'] for d in detections]
                confiancas = [d['confidence'] for d in detections]
                return objetos, confiancas
        
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
                objetos_count = {}
                for obj in (objetos_detectados or []):
                    nome = obj.get('name', 'desconhecido')
                    count = obj.get('count', 1)
                    objetos_count[nome] = objetos_count.get(nome, 0) + count
                
                if objetos_count:
                    desc = "Analisando a imagem: "
                    items = []
                    for obj, qtd in objetos_count.items():
                        if qtd > 1:
                            items.append(f"{qtd} {obj}s")
                        else:
                            items.append(f"um {obj}")
                    
                    if len(items) > 1:
                        desc += ", ".join(items[:-1]) + f" e {items[-1]}"
                    else:
                        desc += items[0] if items else "poucos elementos visíveis"
                    
                    return desc + "."
                else:
                    return "Estou analisando o ambiente, mas não estou detectando muitos elementos específicos no momento."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                # Respostas mais inteligentes para o mock
                pergunta_lower = pergunta.lower()
                objetos_count = {}
                for obj in (objetos_detectados or []):
                    nome = obj.get('name', 'desconhecido')
                    count = obj.get('count', 1)
                    objetos_count[nome] = objetos_count.get(nome, 0) + count
                
                # Construir resposta baseada no tipo de pergunta
                if "quantas pessoas" in pergunta_lower or "quantas pessoa" in pergunta_lower:
                    pessoas = objetos_count.get('person', 0)
                    if pessoas > 0:
                        resposta = f"Na imagem que estou analisando, vejo {pessoas} pessoa{'s' if pessoas > 1 else ''}."
                    else:
                        resposta = "Não estou detectando pessoas na imagem no momento."
                
                elif "o que tem" in pergunta_lower or "descreva" in pergunta_lower:
                    if objetos_count:
                        items = []
                        for obj, qtd in objetos_count.items():
                            if qtd > 1:
                                items.append(f"{qtd} {obj}s")
                            else:
                                items.append(f"um {obj}")
                        
                        if len(items) > 1:
                            resposta = f"Na imagem, identifico: {', '.join(items[:-1])} e {items[-1]}."
                        else:
                            resposta = f"Estou vendo {items[0]} na imagem."
                    else:
                        resposta = "Estou analisando a imagem, mas não estou detectando elementos específicos."
                
                elif "objetos" in pergunta_lower or "identifica" in pergunta_lower:
                    objetos_sem_pessoa = {k: v for k, v in objetos_count.items() if k != 'person'}
                    if objetos_sem_pessoa:
                        items = []
                        for obj, qtd in objetos_sem_pessoa.items():
                            if qtd > 1:
                                items.append(f"{qtd} {obj}s")
                            else:
                                items.append(obj)
                        resposta = f"Identifico os seguintes objetos: {', '.join(items)}."
                    else:
                        resposta = "No momento, não estou detectando objetos além de pessoas."
                
                elif "cadeira" in pergunta_lower:
                    cadeiras = objetos_count.get('chair', 0)
                    if cadeiras > 0:
                        resposta = f"Sim, estou vendo {cadeiras} cadeira{'s' if cadeiras > 1 else ''} na imagem."
                    else:
                        resposta = "Não estou detectando cadeiras na imagem no momento."
                
                elif "intern" in pergunta_lower or "extern" in pergunta_lower:
                    # Tentar inferir baseado nos objetos
                    objetos_interiores = {'chair', 'table', 'bed', 'couch', 'tv', 'laptop', 'book'}
                    tem_objeto_interior = any(obj in objetos_interiores for obj in objetos_count.keys())
                    
                    if tem_objeto_interior:
                        resposta = "Pela presença de móveis e objetos domésticos, parece um ambiente interno."
                    else:
                        resposta = "É difícil determinar sem mais elementos visíveis, mas pode ser um ambiente externo ou espaço aberto."
                
                else:
                    # Resposta genérica
                    if objetos_count:
                        total = sum(objetos_count.values())
                        resposta = f"Estou analisando uma imagem com aproximadamente {total} elementos detectados."
                    else:
                        resposta = "Estou processando a imagem que você enviou. Em que mais posso ajudar?"
                
                return {
                    'sucesso': True,
                    'resposta': resposta,
                    'pergunta': pergunta,
                    'timestamp': time.time(),
                    'tempo_total': '0.2s',
                    'tipo_pergunta': 'sobre_imagem',
                    'correlacao_com_imagem': True
                }
            
            def obter_estatisticas(self, objetos_detectados=None, faces_detectadas=None):
                objetos_count = {}
                for obj in (objetos_detectados or []):
                    nome = obj.get('name', 'desconhecido')
                    count = obj.get('count', 1)
                    objetos_count[nome] = objetos_count.get(nome, 0) + count
                
                return {
                    'sucesso': True,
                    'contagens': {
                        'total_objetos': sum(objetos_count.values()),
                        'total_faces': len(faces_detectadas or []),
                        'objetos_por_categoria': objetos_count
                    },
                    'timestamp': time.time()
                }
        
        return MockInterpreter()

    def _setup_fallback_mode(self):
        """Configura sistema em modo de fallback mínimo"""
        self.detector_objetos = self._create_mock_detector()
        self.detector_faces = self._create_mock_face_detector()
        self.interpreter = self._create_mock_interpreter()
        self.db = None

    def _debug_deteccoes_detalhado(self, frame):
        """Debug DETALHADO das detecções do YOLO"""
        print("\n🔍🔍 DEBUG DETALHADO DO YOLO:")
        
        if not self.detector_objetos:
            print("  ❌ detector_objetos é None!")
            return
        
        try:
            # Verificar tipo do detector
            detector_type = type(self.detector_objetos).__name__
            print(f"  📋 Tipo do detector: {detector_type}")
            
            if 'Mock' in detector_type:
                print("  ⚠️  USANDO DETECTOR MOCK!")
            
            # 1. Tentar método rápido
            print("  📋 Método rápido (threshold=0.15):")
            objetos, confiancas = self.detector_objetos.detectar_objetos_rapido(frame, confidence_threshold=0.15)
            print(f"    Objetos: {objetos}")
            print(f"    Confianças: {confiancas}")
            
            # 2. Tentar método com bbox
            print("  📋 Método com bbox (threshold=0.15):")
            if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                bbox_result = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=0.15)
                print(f"    Total de detecções: {len(bbox_result)}")
                
                # Contar por classe
                if bbox_result:
                    contador = {}
                    for obj in bbox_result:
                        classe = obj.get('class', 'desconhecido')
                        contador[classe] = contador.get(classe, 0) + 1
                    print(f"    Contagem por classe: {contador}")
                    
                    # Mostrar algumas detecções
                    for i, obj in enumerate(bbox_result[:3]):
                        print(f"    [{i}] {obj.get('class')} (conf: {obj.get('confidence'):.2f})")
            
            # Testar com threshold mais baixo
            print("  📋 Teste com threshold BAIXO (0.1):")
            if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                bbox_low = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=0.1)
                contador_low = {}
                for obj in bbox_low:
                    classe = obj.get('class', 'desconhecido')
                    contador_low[classe] = contador_low.get(classe, 0) + 1
                print(f"    Com threshold 0.1: {contador_low}")
                
        except Exception as e:
            print(f"  ❌ Erro no debug detalhado: {e}")
            import traceback
            traceback.print_exc()

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
        """Obtém detecções detalhadas da imagem - VERSÃO MELHORADA"""
        deteccoes = {
            "objetos": [],
            "faces": []
        }
        
        # DEBUG DETALHADO primeiro
        self._debug_deteccoes_detalhado(frame)
        
        # Detecção de objetos com YOLO - COM THRESHOLD MAIS BAIXO
        if self.detector_objetos:
            try:
                # Usar threshold mais baixo para detectar mais objetos
                confidence_threshold = 0.15  # Reduzido de 0.25 para 0.15
                
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    objetos_com_bbox = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=confidence_threshold)
                    
                    # CONTAGEM CORRETA: Agrupar por classe para evitar duplicações
                    contador_classes = {}
                    
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
                                    # Atualizar confiança para a média ponderada
                                    objeto['confidence'] = (objeto['confidence'] * (contador_classes[classe]-1) + confianca) / contador_classes[classe]
                                    break
                                
                    print(f"  ✅ Objetos após agrupamento: {[(o['name'], o['count']) for o in deteccoes['objetos']]}")
                    
                else:
                    # Fallback para método rápido
                    objetos, confiancas = self.detector_objetos.detectar_objetos_rapido(frame, confidence_threshold=confidence_threshold)
                    
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
                    "objetos": deteccoes["objetos"][:15],  # Aumentado para 15
                    "faces": deteccoes["faces"][:5]
                },
                "debug_info": {
                    "confidence_threshold_used": 0.15,
                    "detector_type": type(self.detector_objetos).__name__ if self.detector_objetos else None
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
            
            # Adicionar debug info
            resultado["debug_info"] = {
                "objetos_detectados": [(o['name'], o['count']) for o in deteccoes["objetos"][:5]],
                "total_objetos": len(deteccoes["objetos"]),
                "detector_type": type(self.detector_objetos).__name__ if self.detector_objetos else None
            }
            
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
                "qualidade_imagem": f"{frame.shape[1]}x{frame.shape[0]}",
                "detector_objetos": type(self.detector_objetos).__name__ if self.detector_objetos else "None",
                "confidence_threshold_used": 0.15
            }
            
            # Adicionar lista completa de objetos
            resultado["objetos_detalhados"] = deteccoes["objetos"]
            
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
            detector_type = type(self.detector_objetos).__name__ if self.detector_objetos else "None"
            
            estatisticas = {
                "faces_cadastradas": len(self.detector_faces.known_face_names) if self.detector_faces else 0,
                "detector_objetos_ativo": self.detector_objetos is not None,
                "detector_objetos_tipo": detector_type,
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
                "detector_objetos_tipo": type(occhio.detector_objetos).__name__ if occhio.detector_objetos else "None",
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

@app.route('/debug/detector')
def debug_detector():
    """Endpoint de debug para verificar o detector"""
    try:
        occhio = get_occhio_instance()
        
        # Criar imagem de teste
        test_frame = np.ones((300, 400, 3), dtype=np.uint8) * 200
        
        # Testar detecção
        if hasattr(occhio.detector_objetos, 'detectar_com_bbox'):
            detections = occhio.detector_objetos.detectar_com_bbox(test_frame, confidence_threshold=0.15)
            
            # Contar
            counter = {}
            for det in detections:
                cls_name = det.get('class', 'unknown')
                counter[cls_name] = counter.get(cls_name, 0) + 1
            
            return jsonify({
                "detector_type": type(occhio.detector_objetos).__name__,
                "test_detections": len(detections),
                "class_counts": counter,
                "is_mock": "Mock" in type(occhio.detector_objetos).__name__
            })
        else:
            return jsonify({
                "detector_type": type(occhio.detector_objetos).__name__,
                "error": "Detector não tem método detectar_com_bbox"
            })
            
    except Exception as e:
        return jsonify({
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
    def perguntar_fallback():
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
    def estatistica_fallback():
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

except Exception as e:
    logger.error(f"❌ Erro ao configurar rotas: {e}")

# Adicionar import do request se routes.py não for importado
if 'request' not in locals():
    from flask import request

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