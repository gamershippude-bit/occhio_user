"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
VERSÃO FINAL COM FILTRO CORRETO PARA PESSOAS
"""

# ================== CONFIGURAÇÃO SIMPLES ==================
import os

# Limpar variáveis de proxy do ambiente
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy']:
    if var in os.environ:
        print(f"⚠️ Removendo variável de ambiente {var}")
        os.environ.pop(var, None)

print("🚀 Occhio Cloud iniciando com filtro correto para pessoas")
print("=" * 60)

# ================== IMPORTS ORIGINAIS ==================
import cv2
import logging
import time
import numpy as np
import threading
import traceback
import base64
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
    """Classe principal do sistema Occhio Cloud"""

    def __init__(self, api_key=None):
        try:
            logger.info("=" * 60)
            logger.info("🚀 INICIANDO OCCHIO CLOUD BACKEND")
            logger.info("=" * 60)
            
            # VERIFICAÇÃO DETALHADA DA API KEY
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            
            logger.info(f"📦 API key disponível: {'✅ SIM' if self.api_key else '❌ NÃO'}")
            
            if self.api_key:
                # Verificar se é uma string válida
                if isinstance(self.api_key, str) and self.api_key.strip():
                    logger.info(f"📦 Tamanho da API key: {len(self.api_key)} caracteres")
                    logger.info(f"📦 Prefixo da API key: {self.api_key[:8]}...")
                    
                    if self.api_key.startswith('sk-'):
                        logger.info("✅ Formato OpenAI detectado (sk-...)")
                    else:
                        logger.warning("⚠️ API key não começa com 'sk-', mas tentaremos usar")
                    
                    self.openai_available = True
                else:
                    logger.warning("⚠️ API key está vazia ou inválida")
                    logger.warning(f"⚠️ Valor: {repr(self.api_key)}")
                    self.openai_available = False
            else:
                logger.warning("⚠️ OPENAI_API_KEY NÃO ENCONTRADA!")
                logger.warning("⚠️ O sistema funcionará em modo local")
                self.openai_available = False
            
            # Inicializar componentes
            self.detector_objetos = None
            self.detector_faces = None
            self.db = None
            self.interpreter = None
            
            # Inicializar YOLO
            self._inicializar_yolo()
            
            # Inicializar Interpreter (com a API key)
            self._inicializar_interpreter()
            
            logger.info("=" * 60)
            logger.info("🎉 OCCHIO CLOUD INICIALIZADO COM SUCESSO!")
            logger.info(f"📊 Estado: {'OpenAI ATIVO' if self.openai_available else 'MODO LOCAL'}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"💥 ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            # Setup mínimo
            self._setup_modo_emergencia()

    def _inicializar_yolo(self):
        """Inicializa YOLO"""
        logger.info("🔄 Inicializando YOLO...")
        
        try:
            from Detectors.yolo_detector import YOLODetector
            self.detector_objetos = YOLODetector()
            
            # Testar YOLO imediatamente
            if self.detector_objetos:
                logger.info("🧪 Testando YOLO com imagem de teste...")
                test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                cv2.rectangle(test_frame, (100, 100), (200, 300), (0, 0, 255), -1)
                
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    detections = self.detector_objetos.detectar_com_bbox(test_frame, confidence_threshold=0.01)
                    logger.info(f"✅ YOLO testado - Detectou {len(detections)} objetos")
                else:
                    logger.info("✅ YOLO inicializado (método antigo)")
            else:
                logger.error("❌ YOLO não inicializado")
                
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar YOLO: {e}")
            logger.info("🔄 Usando detector local...")
            self.detector_objetos = self._create_detector_local()

    def _inicializar_interpreter(self):
        """Inicializa interpreter"""
        logger.info("🔄 Inicializando Interpreter...")
        
        try:
            from Utils.interpreter import Interpreter
            
            # Log detalhado
            logger.info(f"📦 Passando API key para Interpreter: {'SIM' if self.api_key else 'NÃO'}")
            
            # Criar interpreter COM a API key
            self.interpreter = Interpreter(api_key=self.api_key)
            
            # Verificar estado do interpreter
            if hasattr(self.interpreter, 'openai_available'):
                if self.interpreter.openai_available:
                    logger.info("✅ Interpreter OpenAI inicializado com sucesso!")
                    # Confirmar que OpenAI está disponível
                    self.openai_available = True
                else:
                    logger.warning("⚠️ Interpreter está em modo local")
                    self.openai_available = False
            else:
                logger.error("❌ Interpreter não tem atributo 'openai_available'")
                self.openai_available = False
                
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar Interpreter: {e}")
            logger.info("🔄 Usando interpreter local...")
            self.interpreter = self._create_interpreter_local()
            self.openai_available = False

    def _create_detector_local(self):
        """Cria um detector local básico"""
        class DetectorLocal:
            def detectar_objetos_yolo(self, frame, confidence_threshold=0.15):
                h, w = frame.shape[:2]
                objetos = []
                confiancas = []
                
                # Detectar pessoas se a imagem for grande o suficiente
                if h > 100 and w > 100:
                    objetos.append('person')
                    confiancas.append(0.85)
                
                return objetos, confiancas
            
            def detectar_com_bbox(self, frame, confidence_threshold=0.15):
                objetos, confiancas = self.detectar_objetos_yolo(frame, confidence_threshold)
                detections = []
                
                for i, (obj, conf) in enumerate(zip(objetos, confiancas)):
                    if conf >= confidence_threshold:
                        detections.append({
                            'class': obj,
                            'confidence': conf,
                            'bbox': {'x': 100, 'y': 100, 'width': 50, 'height': 150}
                        })
                
                return detections
        
        return DetectorLocal()

    def _create_interpreter_local(self):
        """Cria um interpreter local básico"""
        class InterpreterLocal:
            def __init__(self):
                self.openai_available = False
            
            def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
                return "Olá! Sou a Specula, sua assistente visual."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                # Esta é uma versão local simples
                return {
                    'sucesso': True,
                    'resposta': "Olá! Sou a Specula. Estou em modo local.",
                    'pergunta': pergunta,
                    'timestamp': time.time(),
                    'tempo_total': "0.5s",
                    'tipo_pergunta': 'geral',
                    'correlacao_com_imagem': False,
                    'dados_utilizados': 'modo local'
                }
            
            def obter_estatisticas(self, objetos_detectados=None, faces_detectadas=None):
                return {
                    'sucesso': True,
                    'contagens': {
                        'total_objetos': len(objetos_detectados or []),
                        'total_faces': len(faces_detectadas or [])
                    },
                    'timestamp': time.time()
                }
        
        return InterpreterLocal()

    def _setup_modo_emergencia(self):
        """Setup mínimo de emergência"""
        self.detector_objetos = self._create_detector_local()
        self.interpreter = self._create_interpreter_local()
        self.detector_faces = None
        self.db = None
        self.openai_available = False
        logger.warning("⚠️ Sistema em modo emergência")

    def _decode_image(self, image_data):
        """Decodifica imagem"""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                raise ValueError(f"Tipo de imagem não suportado: {type(image_data)}")
            
            if frame is None:
                raise ValueError("Falha ao decodificar imagem")
            
            logger.info(f"📏 Imagem decodificada: {frame.shape[1]}x{frame.shape[0]} pixels")
            
            # Redimensionar se for muito grande
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                scale = min(1280/w, 720/h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                logger.info(f"📏 Imagem redimensionada: {new_w}x{new_h}")
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar imagem: {e}")
            raise

    def _obter_deteccoes_detalhadas(self, frame):
        """Obtém detecções detalhadas - COM FILTRO CORRETO PARA PESSOAS"""
        deteccoes = {
            "objetos": [],
            "faces": []
        }
        
        logger.info(f"🔍 Processando frame: {frame.shape[1]}x{frame.shape[0]}")
        
        # Detecção de objetos - FILTRO CRÍTICO PARA PESSOAS
        if self.detector_objetos:
            try:
                if hasattr(self.detector_objetos, 'detectar_com_bbox'):
                    logger.info("🎯 Chamando detectar_com_bbox com filtro rigoroso...")
                    
                    # THRESHOLD ALTO para evitar falsos positivos
                    confidence_threshold = 0.60  # 60% mínimo
                    
                    all_detections = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=confidence_threshold)
                    
                    logger.info(f"🔍 YOLO detectou {len(all_detections)} objetos brutos")
                    
                    # FILTRO ESPECIAL: Para pessoas, exigir confiança MUITO ALTA
                    detections_filtradas = []
                    pessoas_detectadas = []
                    
                    for det in all_detections:
                        if det['class'] == 'person':
                            if det['confidence'] >= 0.70:
                                pessoas_detectadas.append(det)
                                detections_filtradas.append(det)
                                logger.info(f"✅ Pessoa detectada com alta confiança: {det['confidence']:.3f}")
                            else:
                                logger.info(f"⏩ Ignorando pessoa com confiança baixa: {det['confidence']:.3f}")
                        else:
                            # Para outros objetos, manter threshold normal
                            detections_filtradas.append(det)
                    
                    # SE TEM MAIS DE 1 PESSOA, VERIFICAR SE SÃO FALSOS POSITIVOS
                    if len(pessoas_detectadas) > 1:
                        logger.warning(f"⚠️ Detectadas {len(pessoas_detectadas)} pessoas. Verificando falsos positivos...")
                        
                        # Ordenar por confiança
                        pessoas_detectadas.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        # Manter apenas a pessoa com MAIOR confiança
                        # REDUZIDO de 85% para 75% - mais realista
                        if pessoas_detectadas[0]['confidence'] > 0.75 and len(pessoas_detectadas) > 1:
                            # Se a melhor tem >75% e tem outras, verificar diferença
                            melhor_conf = pessoas_detectadas[0]['confidence']
                            segunda_conf = pessoas_detectadas[1]['confidence']
                            
                            # Se a segunda tem MUITO menos confiança (diferença > 20%), descarta
                            if melhor_conf - segunda_conf > 0.20:
                                # Remover da lista filtrada todas as pessoas exceto a melhor
                                detections_filtradas = [d for d in detections_filtradas if d['class'] != 'person']
                                
                                # Adicionar apenas a melhor
                                detections_filtradas.append(pessoas_detectadas[0])
                                logger.info(f"✅ Mantendo apenas 1 pessoa (melhor: {melhor_conf:.3f}, segunda: {segunda_conf:.3f})")
                                
                                # Remover as outras pessoas da lista original
                                for i in range(1, len(pessoas_detectadas)):
                                    logger.info(f"⏩ Descartando possível falso positivo: {pessoas_detectadas[i]['confidence']:.3f}")
                                        
                                    logger.info(f"🔍 Após filtro rigoroso: {len(detections_filtradas)} objetos")
                                        
                    # DEBUG detalhado
                    for i, det in enumerate(detections_filtradas[:10]):
                        logger.info(f"   [{i+1}] {det['class']}: {det['confidence']:.3f}")
                    
                    # Agrupar objetos
                    objetos_agrupados = {}
                    for det in detections_filtradas:
                        class_name = det['class']
                        if class_name not in objetos_agrupados:
                            objetos_agrupados[class_name] = {
                                'name': class_name,
                                'confidence': det['confidence'],
                                'bbox': det['bbox'],
                                'count': 0
                            }
                        objetos_agrupados[class_name]['count'] += 1
                    
                    # Adicionar à lista de detecções
                    for obj in objetos_agrupados.values():
                        deteccoes["objetos"].append(obj)
                        logger.info(f"✅ Objeto final: {obj['name']} (x{obj['count']}) - conf: {obj['confidence']:.2f}")
                    
                    logger.info(f"📊 Total objetos confiáveis: {len(deteccoes['objetos'])}")
                    
                else:
                    logger.warning("⚠️ Detector não tem método detectar_com_bbox")
                    
            except Exception as e:
                logger.error(f"❌ Erro detecção objetos: {e}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
        else:
            logger.warning("⚠️ Detector de objetos não disponível")
        
        return deteccoes

    def processar_imagem_seguranca(self, image_data):
        """
        ROTA /processar - Processa imagem
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
                    "total_faces": len(deteccoes["faces"])
                },
                "deteccoes": {
                    "objetos": deteccoes["objetos"][:10],
                    "faces": deteccoes["faces"][:5]
                },
                "sistema_info": {
                    "detector_tipo": type(self.detector_objetos).__name__,
                    "interpreter_tipo": type(self.interpreter).__name__,
                    "openai_disponivel": self.openai_available
                }
            }
            
            logger.info(f"✅ Segurança processada - {len(deteccoes['objetos'])} objetos")
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
        ROTA /perguntar - Pergunta sobre imagem
        """
        try:
            logger.info(f"💬 Processando pergunta: '{pergunta}'")
            start_time = time.time()
            
            # Decodificar imagem
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            # DEBUG: Logar o que foi detectado
            logger.info(f"📊 Para a pergunta '{pergunta}', detectamos:")
            for i, obj in enumerate(deteccoes["objetos"]):
                logger.info(f"   {i+1}. {obj['name']} (x{obj['count']}) - conf: {obj['confidence']:.2f}")
            
            # Extrair nomes das faces
            faces_nomes = [face['name'] for face in deteccoes["faces"]]
            
            # Usar interpreter
            if not self.interpreter:
                return {
                    "sucesso": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            resultado = self.interpreter.perguntar_sobre_imagem(
                pergunta=pergunta,
                objetos_detectados=deteccoes["objetos"],
                faces_nomes=faces_nomes
            )
            
            processing_time = time.time() - start_time
            
            # Garantir que resultado é um dicionário
            if isinstance(resultado, dict):
                resultado["tempo_total"] = f"{processing_time:.2f}s"
            else:
                resultado = {
                    'sucesso': True,
                    'resposta': str(resultado),
                    'pergunta': pergunta,
                    'timestamp': time.time(),
                    'tempo_total': f"{processing_time:.2f}s",
                    'tipo_pergunta': 'geral',
                    'correlacao_com_imagem': False,
                    'dados_utilizados': 'modo básico'
                }
            
            logger.info(f"✅ Pergunta respondida em {processing_time:.2f}s")
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
        ROTA /estatistica - Estatísticas
        """
        try:
            logger.info("📊 Gerando estatísticas")
            start_time = time.time()
            
            frame = self._decode_image(image_data)
            deteccoes = self._obter_deteccoes_detalhadas(frame)
            
            if not self.interpreter:
                return {
                    "sucesso": False,
                    "error": "Interpreter não disponível",
                    "timestamp": time.time()
                }
            
            resultado = self.interpreter.obter_estatisticas(
                objetos_detectados=deteccoes["objetos"],
                faces_detectadas=deteccoes["faces"]
            )
            
            processing_time = time.time() - start_time
            
            if isinstance(resultado, dict):
                resultado["tempo_total"] = f"{processing_time:.2f}s"
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro obter_estatisticas_detalhadas: {e}")
            return {
                "sucesso": False,
                "error": str(e),
                "timestamp": time.time()
            }

# Singleton para a instância
def get_occhio_instance():
    global _occhio_instance
    if _occhio_instance is None:
        with _initialization_lock:
            if _occhio_instance is None:
                try:
                    # Obter API key do ambiente
                    api_key = os.getenv('OPENAI_API_KEY')
                    
                    logger.info(f"🔧 Criando instância OcchioCloud...")
                    logger.info(f"📦 OPENAI_API_KEY do ambiente: {'✅ SIM' if api_key else '❌ NÃO'}")
                    
                    if api_key:
                        logger.info(f"📦 Tamanho da key: {len(api_key)}")
                    
                    _occhio_instance = OcchioCloud(api_key=api_key)
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao criar instância: {e}")
                    _occhio_instance = OcchioCloud(api_key=None)
    return _occhio_instance

# ========== ROTAS FLASK ==========

@app.route('/')
def index():
    return jsonify({
        "app": "Occhio Cloud API",
        "version": "4.0.0",
        "status": "online",
        "timestamp": time.time(),
        "features": "YOLO + Specula AI (OpenAI 0.28.1)",
        "endpoints": {
            "/": "GET - Esta página",
            "/health": "GET - Health check",
            "/processar": "POST - Processa imagem",
            "/perguntar": "POST - Pergunta sobre imagem",
            "/estatistica": "POST - Estatísticas"
        }
    })

@app.route('/health')
def health():
    try:
        occhio = get_occhio_instance()
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "detector_objetos": occhio.detector_objetos is not None,
                "detector_tipo": type(occhio.detector_objetos).__name__,
                "interpreter_tipo": type(occhio.interpreter).__name__,
                "openai_disponivel": occhio.openai_available
            }
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }), 200

@app.route('/processar', methods=['POST'])
def processar():
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

# ========== EXECUÇÃO ==========

if __name__ == "__main__":
    port = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud v4.0 na porta {port}")
    
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        app.run(host='0.0.0.0', port=port, debug=False)