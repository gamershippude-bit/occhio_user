"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
Versão: 5.0.0 - API Padronizada (Português)
"""
import os
import cv2
import logging
import time
import json
import numpy as np
import threading
import base64
from pathlib import Path
from flask import Flask, jsonify, request
from flask_sock import Sock
from typing import Dict, List, Any, Optional

# ================== CONFIGURAÇÃO ==================
# Limpar variáveis de proxy
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy']:
    if var in os.environ:
        os.environ.pop(var, None)

print("🚀 Occhio Cloud v5.0 - API Padronizada (Português)")
print("=" * 60)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/occhio_cloud.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("Occhio-Cloud")

app = Flask(__name__)
sock = Sock(app)

DASHBOARD_HTML = Path(__file__).parent / 'occhio_dashboard.html'

# Cache para singleton
_occhio_instance = None
_initialization_lock = threading.Lock()

class OcchioCloud:
    """Classe principal com API padronizada"""
    
    def __init__(self, api_key=None):
        try:
            logger.info("🚀 INICIANDO OCCHIO CLOUD v5.0")
            
            # API Key
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.openai_disponivel = bool(self.api_key and isinstance(self.api_key, str) and self.api_key.strip())
            
            logger.info(f"📦 OpenAI: {'✅ DISPONÍVEL' if self.openai_disponivel else '❌ INDISPONÍVEL'}")
            
            # Inicializar componentes
            self.detector_objetos = None
            self.interpreter = None
            
            self._inicializar_yolo()
            self._inicializar_interpreter()
            
            logger.info("🎉 Sistema inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"💥 Erro na inicialização: {e}")
            self._setup_modo_emergencia()
    
    def _inicializar_yolo(self):
        """Inicializa YOLO"""
        try:
            from Detectors.yolo_detector import YOLODetector
            self.detector_objetos = YOLODetector()
            logger.info("✅ YOLO inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar YOLO: {e}")
            self.detector_objetos = self._criar_detector_local()
    
    def _inicializar_interpreter(self):
        """Inicializa interpreter"""
        try:
            from Utils.interpreter import Interpreter
            self.interpreter = Interpreter(api_key=self.api_key)
            logger.info("✅ Interpreter inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar interpreter: {e}")
            self.interpreter = self._criar_interpreter_local()
    
    def _criar_detector_local(self):
        """Detector local de fallback"""
        class DetectorLocal:
            def detectar_com_bbox(self, frame, confidence_threshold=0.5):
                return []  # Sem detecções no modo local
        return DetectorLocal()
    
    def _criar_interpreter_local(self):
        """Interpreter local de fallback"""
        class InterpreterLocal:
            def __init__(self):
                self.openai_disponivel = False
            
            def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
                if objetos_detectados:
                    return f"Detectei {len(objetos_detectados)} objetos."
                return "Nenhum objeto detectado claramente."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                return {
                    "resposta": "Sistema em modo local. Sem resposta da OpenAI.",
                    "correlacao_com_imagem": False,
                    "confianca": 0.0
                }
        return InterpreterLocal()
    
    def _setup_modo_emergencia(self):
        """Setup de emergência"""
        self.detector_objetos = self._criar_detector_local()
        self.interpreter = self._criar_interpreter_local()
        self.openai_disponivel = False
        logger.warning("⚠️ Sistema em modo emergência")
    
    def _decodificar_imagem(self, dados_imagem: str) -> np.ndarray:
        """Decodifica imagem base64"""
        try:
            if isinstance(dados_imagem, str):
                if dados_imagem.startswith('data:image'):
                    dados_imagem = dados_imagem.split(',')[1]
                
                bytes_imagem = base64.b64decode(dados_imagem)
                nparr = np.frombuffer(bytes_imagem, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("Falha ao decodificar imagem")
                
                # Redimensionar se necessário
                altura, largura = frame.shape[:2]
                tamanho_maximo = 1280
                
                if largura > tamanho_maximo or altura > tamanho_maximo:
                    escala = min(tamanho_maximo/largura, tamanho_maximo/altura)
                    nova_largura, nova_altura = int(largura * escala), int(altura * escala)
                    frame = cv2.resize(frame, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
                
                return frame
            else:
                raise ValueError("dados_imagem deve ser string base64")
                
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar imagem: {e}")
            raise
    
    def _normalizar_coordenadas(self, bbox: Dict, largura_imagem: int, altura_imagem: int) -> Dict:
        """Normaliza coordenadas para 0-1"""
        try:
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            largura = bbox.get('width', 0)
            altura = bbox.get('height', 0)
            
            if largura_imagem > 0 and altura_imagem > 0:
                return {
                    'x': x / largura_imagem,
                    'y': y / altura_imagem,
                    'largura': largura / largura_imagem,
                    'altura': altura / altura_imagem
                }
            else:
                return {'x': 0, 'y': 0, 'largura': 0, 'altura': 0}
        except:
            return {'x': 0, 'y': 0, 'largura': 0, 'altura': 0}
    
    def _processar_imagem_interna(self, dados_imagem: str) -> Dict[str, Any]:
        """Processa imagem e retorna detecções no formato padronizado"""
        inicio_tempo = time.time()
        
        try:
            # Decodificar imagem
            frame = self._decodificar_imagem(dados_imagem)
            altura_imagem, largura_imagem = frame.shape[:2]
            
            # Detectar objetos
            deteccoes = []
            if self.detector_objetos and hasattr(self.detector_objetos, 'detectar_com_bbox'):
                deteccoes_brutas = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=0.5)
                
                for i, det in enumerate(deteccoes_brutas):
                    # Normalizar coordenadas
                    caixa_normalizada = self._normalizar_coordenadas(det['bbox'], largura_imagem, altura_imagem)
                    
                    deteccao = {
                        'id': i + 1,
                        'nome': det['class'],  # Nome em inglês (YOLO retorna em inglês)
                        'confianca': float(det['confidence']),  # Float 0.0-1.0
                        'caixa': caixa_normalizada,  # Coordenadas normalizadas
                        'caixa_pixels': det['bbox']  # Coordenadas em pixels (opcional)
                    }
                    deteccoes.append(deteccao)
            
            tempo_processamento_ms = int((time.time() - inicio_tempo) * 1000)
            
            return {
                'deteccoes': deteccoes,
                'info_imagem': {
                    'largura': largura_imagem,
                    'altura': altura_imagem
                },
                'tempo_processamento_ms': tempo_processamento_ms
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}")
            return {
                'deteccoes': [],
                'info_imagem': {'largura': 0, 'altura': 0},
                'tempo_processamento_ms': 0,
                'erro': str(e)
            }
    
    def processar(self, dados_imagem: str) -> Dict[str, Any]:
        
        try:
            # Processar imagem
            resultado = self._processar_imagem_interna(dados_imagem)
            
            # Preparar resposta
            resposta = {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),  # millis
                "tempo_processamento_ms": resultado['tempo_processamento_ms'],
                "dados": {
                    "resumo": {
                        "total_objetos": len(resultado['deteccoes'])
                    },
                    "deteccoes": resultado['deteccoes'],
                    "info_imagem": resultado.get('info_imagem', {})
                }
            }
            
            if 'erro' in resultado:
                resposta['erro'] = resultado['erro']
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro em processar: {e}")
            return {
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": 0,
                "erro": str(e)
            }
    
    def perguntar(self, dados_imagem: str, pergunta: str) -> Dict[str, Any]:
     
        inicio_tempo = time.time()
        
        try:
            # Processar imagem primeiro
            resultado_deteccao = self._processar_imagem_interna(dados_imagem)
            deteccoes = resultado_deteccao['deteccoes']
            
            # Preparar objetos para o interpreter
            objetos_para_interpreter = []
            for det in deteccoes[:10]:  # Limitar para não sobrecarregar
                objetos_para_interpreter.append({
                    'nome': det['nome'],
                    'confianca': det['confianca'],
                    'quantidade': 1
                })
            
            # Obter resposta do interpreter
            resposta_chat = None
            correlacao_com_imagem = False
            confianca_resposta = 0.0
            
            if self.interpreter and hasattr(self.interpreter, 'perguntar_sobre_imagem'):
                resultado_interpreter = self.interpreter.perguntar_sobre_imagem(
                    pergunta=pergunta,
                    objetos_detectados=objetos_para_interpreter,
                    faces_nomes=[]
                )
                
                # Extrair resposta do resultado
                if isinstance(resultado_interpreter, dict):
                    resposta_chat = resultado_interpreter.get('resposta')
                    correlacao_com_imagem = resultado_interpreter.get('correlacao_com_imagem', False)
                    confianca_resposta = float(resultado_interpreter.get('confianca', 0.0))
                else:
                    resposta_chat = str(resultado_interpreter)
            else:
                resposta_chat = "Interpreter não disponível."
            
            tempo_processamento_ms = int((time.time() - inicio_tempo) * 1000)
            
            # Detecções relevantes (apenas as mais confiantes)
            deteccoes_relevantes = []
            for det in sorted(deteccoes, key=lambda x: x['confianca'], reverse=True)[:3]:
                deteccoes_relevantes.append({
                    'nome': det['nome'],
                    'confianca': det['confianca']
                })
            
            return {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": tempo_processamento_ms,
                "dados": {
                    "pergunta": pergunta,
                    "resposta": resposta_chat or "Sem resposta disponível.",
                    "deteccoes_relevantes": deteccoes_relevantes,
                    "correlacao_com_imagem": correlacao_com_imagem,
                    "confianca_resposta": confianca_resposta,
                    "total_deteccoes": len(deteccoes)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro em perguntar: {e}")
            return {
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": 0,
                "erro": str(e)
            }
    
    def estatistica(self, dados_imagem: str) -> Dict[str, Any]:

        inicio_tempo = time.time()
        
        try:
            # Processar imagem
            resultado = self._processar_imagem_interna(dados_imagem)
            deteccoes = resultado['deteccoes']
            
            # Calcular estatísticas
            contagem_objetos = {}
            confiancas = []
            
            for det in deteccoes:
                nome = det['nome']
                contagem_objetos[nome] = contagem_objetos.get(nome, 0) + 1
                confiancas.append(det['confianca'])
            
            # Calcular estatísticas de confiança
            estatisticas_confianca = {}
            if confiancas:
                estatisticas_confianca = {
                    'media': round(float(np.mean(confiancas)), 3),
                    'maxima': round(float(np.max(confiancas)), 3),
                    'minima': round(float(np.min(confiancas)), 3),
                    'mediana': round(float(np.median(confiancas)), 3)
                }
            else:
                estatisticas_confianca = {
                    'media': 0,
                    'maxima': 0,
                    'minima': 0,
                    'mediana': 0
                }
            
            tempo_processamento_ms = int((time.time() - inicio_tempo) * 1000)
            
            return {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": tempo_processamento_ms,
                "dados": {
                    "resumo": {
                        "total_objetos": len(deteccoes),
                        "objetos_unicos": len(contagem_objetos)
                    },
                    "contagem_objetos": contagem_objetos,
                    "estatisticas_confianca": estatisticas_confianca,
                    "amostra_deteccoes": deteccoes[:5]  # Amostra de detecções
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro em estatistica: {e}")
            return {
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": 0,
                "erro": str(e)
            }

    def processar_stream(self, dados_imagem: str, confidence_threshold: float = 0.45) -> Dict[str, Any]:
        """Processa frame para streaming WebSocket — retorna bbox normalizadas (x, y, w, h)."""
        inicio = time.time()
        try:
            frame = self._decodificar_imagem(dados_imagem)
            altura, largura = frame.shape[:2]

            if largura > 640:
                escala = 640 / largura
                frame = cv2.resize(frame, (640, int(altura * escala)), interpolation=cv2.INTER_AREA)
                altura, largura = frame.shape[:2]

            deteccoes = []
            if self.detector_objetos and hasattr(self.detector_objetos, 'detectar_com_bbox'):
                brutas = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=confidence_threshold)
                for det in brutas:
                    bbox = det.get('bbox', {})
                    x = bbox.get('x', 0)
                    y = bbox.get('y', 0)
                    w = bbox.get('width', 0)
                    h = bbox.get('height', 0)
                    deteccoes.append({
                        'nome': det.get('class', '?'),
                        'confianca': round(float(det.get('confidence', 0)), 2),
                        'x': round(x / largura, 4),
                        'y': round(y / altura, 4),
                        'w': round(w / largura, 4),
                        'h': round(h / altura, 4),
                    })

            return {
                'deteccoes': deteccoes,
                'total': len(deteccoes),
                'ms': int((time.time() - inicio) * 1000),
                'resolucao': f'{largura}x{altura}',
            }
        except Exception as e:
            logger.error(f'Erro no stream: {e}')
            return {'erro': str(e), 'deteccoes': [], 'total': 0, 'ms': 0}

# Singleton
def get_occhio_instance():
    global _occhio_instance
    if _occhio_instance is None:
        with _initialization_lock:
            if _occhio_instance is None:
                api_key = os.getenv('OPENAI_API_KEY')
                _occhio_instance = OcchioCloud(api_key=api_key)
    return _occhio_instance

# ========== ROTAS FLASK ==========

@app.route('/')
def dashboard():
    """Dashboard de demonstração com câmera em tempo real."""
    if DASHBOARD_HTML.exists():
        return DASHBOARD_HTML.read_text(encoding='utf-8')
    return jsonify({'erro': 'Dashboard não encontrado'}), 404


@app.route('/api')
def index():
    return jsonify({
        "app": "Occhio Cloud API",
        "versao": "5.0.0",
        "status": "online",
        "timestamp": int(time.time() * 1000),
        "rotas": {
            "/": "GET - Dashboard ao vivo",
            "/api": "GET - Esta página",
            "/health": "GET - Health check",
            "/stream": "WS - Stream de vídeo em tempo real",
            "/processar": "POST - Processa imagem",
            "/perguntar": "POST - Pergunta sobre imagem",
            "/estatistica": "POST - Estatísticas da imagem"
        }
    })


@sock.route('/stream')
def stream_ws(ws):
    """WebSocket: browser envia frames, servidor devolve bounding boxes."""
    logger.info('Cliente conectado via WebSocket')
    frame_count = 0
    try:
        while True:
            mensagem = ws.receive()
            if mensagem is None:
                break
            try:
                dados = json.loads(mensagem)
            except json.JSONDecodeError:
                ws.send(json.dumps({'erro': 'JSON inválido'}))
                continue

            frame_b64 = dados.get('frame')
            if not frame_b64:
                ws.send(json.dumps({'erro': "Campo 'frame' não encontrado"}))
                continue

            threshold = float(dados.get('threshold', 0.45))
            occhio = get_occhio_instance()
            resultado = occhio.processar_stream(frame_b64, confidence_threshold=threshold)
            ws.send(json.dumps(resultado))

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f'{frame_count} frames processados | último: {resultado.get("ms")}ms')
    except Exception as e:
        logger.info(f'Cliente desconectado: {e}')

@app.route('/health', methods=['GET'])
def health():
    try:
        occhio = get_occhio_instance()
        return jsonify({
            "sucesso": True,
            "timestamp": int(time.time() * 1000),
            "status": "saudavel",
            "servicos": {
                "detector_yolo": occhio.detector_objetos is not None,
                "openai_interpreter": occhio.openai_disponivel,
                "modelo": "YOLOv8s"
            }
        })
    except Exception as e:
        return jsonify({
            "sucesso": False,
            "timestamp": int(time.time() * 1000),
            "status": "degradado",
            "erro": str(e)
        })

@app.route('/processar', methods=['POST'])
def processar():

    try:
        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Campo 'imagem' não encontrado no corpo da requisição",
                "codigo": "IMAGEM_NAO_ENCONTRADA"
            }), 400
        
        occhio = get_occhio_instance()
        resultado = occhio.processar(data['imagem'])
        
        status_code = 200 if resultado.get('sucesso', False) else 500
        return jsonify(resultado), status_code
        
    except Exception as e:
        logger.error(f"❌ Erro em /processar: {e}")
        return jsonify({
            "sucesso": False,
            "timestamp": int(time.time() * 1000),
            "erro": str(e),
            "codigo": "ERRO_SERVIDOR"
        }), 500

@app.route('/perguntar', methods=['POST'])
def perguntar():

    try:
        data = request.get_json()
        
        # Validação
        if not data:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Corpo da requisição vazio",
                "codigo": "REQUISICAO_VAZIA"
            }), 400
        
        if 'imagem' not in data:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Campo 'imagem' não encontrado",
                "codigo": "IMAGEM_NAO_ENCONTRADA"
            }), 400
        
        if 'pergunta' not in data:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Campo 'pergunta' não encontrado",
                "codigo": "PERGUNTA_NAO_ENCONTRADA"
            }), 400
        
        pergunta = data['pergunta'].strip()
        if len(pergunta) < 2:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Pergunta muito curta",
                "codigo": "PERGUNTA_CURTA"
            }), 400
        
        occhio = get_occhio_instance()
        resultado = occhio.perguntar(data['imagem'], pergunta)
        
        status_code = 200 if resultado.get('sucesso', False) else 500
        return jsonify(resultado), status_code
        
    except Exception as e:
        logger.error(f"❌ Erro em /perguntar: {e}")
        return jsonify({
            "sucesso": False,
            "timestamp": int(time.time() * 1000),
            "erro": str(e),
            "codigo": "ERRO_SERVIDOR"
        }), 500

@app.route('/estatistica', methods=['POST'])
def estatistica():

    try:
        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({
                "sucesso": False,
                "timestamp": int(time.time() * 1000),
                "erro": "Campo 'imagem' não encontrado",
                "codigo": "IMAGEM_NAO_ENCONTRADA"
            }), 400
        
        occhio = get_occhio_instance()
        resultado = occhio.estatistica(data['imagem'])
        
        status_code = 200 if resultado.get('sucesso', False) else 500
        return jsonify(resultado), status_code
        
    except Exception as e:
        logger.error(f"❌ Erro em /estatistica: {e}")
        return jsonify({
            "sucesso": False,
            "timestamp": int(time.time() * 1000),
            "erro": str(e),
            "codigo": "ERRO_SERVIDOR"
        }), 500

# ========== EXECUÇÃO ==========

if __name__ == "__main__":
    porta = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud v5.0 na porta {porta}")
    logger.info(f"   Dashboard: http://localhost:{porta}/")
    logger.info(f"   WebSocket: ws://localhost:{porta}/stream")

    logger.info("🔧 Inicializando componentes...")
    get_occhio_instance()

    # Flask dev server suporta WebSocket (Waitress não suporta)
    logger.info(f"🌐 Servindo com Flask (WebSocket habilitado) na porta {porta}...")
    app.run(host='0.0.0.0', port=porta, debug=False, threaded=True)