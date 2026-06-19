"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
Versão: 5.0.0 - API Padronizada (Português)
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import logging
import time
import json
import numpy as np
import threading
import base64
import tempfile
from pathlib import Path
from flask import Flask, jsonify, request
from flask_sock import Sock
from typing import Dict, List, Any, Optional

from Utils.glm_client import chat as glm_chat, glm_disponivel
from Utils.face_registry import FaceRegistry, CadastroSessao
from Utils.face_store import criar_face_store

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
            
            self.whisper_disponivel = bool(
                os.getenv('OPENAI_API_KEY', '').strip()
            )
            self.glm_disponivel = glm_disponivel()
            
            logger.info(f"📦 Whisper: {'✅ DISPONÍVEL' if self.whisper_disponivel else '❌ INDISPONÍVEL'}")
            logger.info(f"📦 GLM-5: {'✅ DISPONÍVEL' if self.glm_disponivel else '❌ INDISPONÍVEL'}")
            
            # Inicializar componentes
            self.detector_objetos = None
            self.interpreter = None
            self._stream_face_counter = 0
            self._last_rostos = []
            self.detector_faces = None
            self.face_store = None
            self.face_registry = None
            
            self._inicializar_yolo()
            self._inicializar_faces()
            self._inicializar_interpreter()
            self.glm_disponivel = getattr(self.interpreter, 'glm_disponivel', False)
            
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
    
    def _inicializar_faces(self):
        """Inicializa reconhecimento facial e banco de rostos."""
        try:
            from Detectors.face_detector import FaceDetector
            self.detector_faces = FaceDetector()
            self.face_store = criar_face_store()
            self.face_registry = FaceRegistry(self.detector_faces, self.face_store)
            self.face_registry.recarregar_rostos()
            logger.info('✅ Reconhecimento facial inicializado')
        except Exception as e:
            logger.error(f'❌ Erro ao inicializar faces: {e}')
            self.detector_faces = None
            self.face_store = None
            self.face_registry = None
    
    def _inicializar_interpreter(self):
        """Inicializa interpreter"""
        try:
            from Utils.interpreter import Interpreter
            self.interpreter = Interpreter()
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
                self.glm_disponivel = False
            
            def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
                if objetos_detectados:
                    return f"Detectei {len(objetos_detectados)} objetos."
                return "Nenhum objeto detectado claramente."
            
            def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
                return {
                    "resposta": "Sistema em modo local. Sem resposta do GLM.",
                    "correlacao_com_imagem": False,
                    "confianca": 0.0
                }
        return InterpreterLocal()
    
    def _setup_modo_emergencia(self):
        """Setup de emergência"""
        self.detector_objetos = self._criar_detector_local()
        self.detector_faces = None
        self.face_store = None
        self.face_registry = None
        self.interpreter = self._criar_interpreter_local()
        self.glm_disponivel = False
        self.whisper_disponivel = False
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
        """Processa frame para streaming WebSocket — objetos YOLO + rostos."""
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

            rostos = self._last_rostos
            alertas = []
            if self.face_registry:
                self._stream_face_counter += 1
                if self._stream_face_counter % 2 == 0:
                    face_w = 320
                    face_h = max(1, int(altura * face_w / largura))
                    face_frame = cv2.resize(frame, (face_w, face_h), interpolation=cv2.INTER_AREA)
                    rostos = self.face_registry.detectar_faces_stream(face_frame)
                    self._last_rostos = rostos
                    alertas = self.face_registry.verificar_alertas(rostos)

            return {
                'deteccoes': deteccoes,
                'rostos': rostos,
                'alertas': alertas,
                'total': len(deteccoes),
                'ms': int((time.time() - inicio) * 1000),
                'resolucao': f'{largura}x{altura}',
            }
        except Exception as e:
            logger.error(f'Erro no stream: {e}')
            return {'erro': str(e), 'deteccoes': [], 'rostos': [], 'alertas': [], 'total': 0, 'ms': 0}

# ========== VOZ (Whisper + GLM-5 + ElevenLabs) ==========

SYSTEM_PROMPT_VOZ = """Você é o Specula, assistente de acessibilidade para deficientes visuais.
O usuário usa a câmera em tempo real. Você recebe objetos detectados e rostos reconhecidos pelo nome.

REGRAS DE NARRAÇÃO (sua resposta será LIDA EM VOZ ALTA):
- Responda em português brasileiro, com no máximo 2 frases curtas.
- Responda EXATAMENTE o que foi perguntado — se pedirem "além de X" ou "o que mais", fale do RESTO, não repita X.
- Seja direto. Priorize objetos e contexto quando a pergunta for sobre a cena.
- NUNCA mencione parentesco, relação ou "seu amigo/irmão" — a menos que o usuário pergunte explicitamente (ex: "tem algum amigo aqui?").
- NUNCA mencione porcentagens ou confiança.
- NUNCA use "ele(a)" ou formas com barra — use o nome ou "a pessoa".
- Baseie-se APENAS nos dados fornecidos. Não invente.
- Para cadastrar rosto: diga "Diga cadastrar essa pessoa"."""

_elevenlabs_client = None
_elevenlabs_lock = threading.Lock()


def _get_elevenlabs_client():
    global _elevenlabs_client
    if _elevenlabs_client is not None:
        return _elevenlabs_client
    with _elevenlabs_lock:
        if _elevenlabs_client is not None:
            return _elevenlabs_client
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            return None
        try:
            from elevenlabs import ElevenLabs
            _elevenlabs_client = ElevenLabs(api_key=api_key)
            logger.info('✅ ElevenLabs inicializado')
        except Exception as e:
            logger.error(f'❌ Erro ao inicializar ElevenLabs: {e}')
        return _elevenlabs_client


def _formatar_contexto_voz(
    deteccoes: List[Dict],
    rostos: List[Dict],
    catalogo: Optional[Dict[str, dict]] = None,
) -> str:
    partes = []

    if catalogo:
        nomes = ', '.join(m['nome'] for m in catalogo.values())
        partes.append(f'Rostos cadastrados (referência): {nomes}')

    if deteccoes:
        contagem: Dict[str, int] = {}
        for d in deteccoes:
            nome = d.get('nome', '?')
            contagem[nome] = contagem.get(nome, 0) + 1
        linhas = []
        for nome, qtd in contagem.items():
            linhas.append(f'- {nome}' + (f' (×{qtd})' if qtd > 1 else ''))
        partes.append('Objetos:\n' + '\n'.join(linhas))
    else:
        partes.append('Objetos: nenhum.')

    if rostos:
        linhas = []
        for r in rostos:
            if r.get('conhecido'):
                linhas.append(f"- {r.get('nome', '?')}")
            else:
                linhas.append('- rosto desconhecido')
        partes.append('Rostos na câmera:\n' + '\n'.join(linhas))
    else:
        partes.append('Rostos: nenhum.')

    return '\n\n'.join(partes)


def _limpar_resposta_fala(texto: str) -> str:
    """Remove porcentagens e formas inadequadas para narração em voz."""
    import re
    texto = re.sub(r'\d+\s*%', '', texto)
    texto = re.sub(r'\(\s*\)', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def _rostos_unicos(rostos: List[Dict]) -> tuple:
    """Agrupa rostos por identidade — evita contar a mesma pessoa várias vezes."""
    conhecidos_map: Dict[str, Dict] = {}
    desconhecidos = 0
    for r in rostos:
        if r.get('conhecido') and r.get('nome'):
            chave = r['nome'].lower()
            if chave not in conhecidos_map:
                conhecidos_map[chave] = r
        else:
            desconhecidos += 1
    return list(conhecidos_map.values()), desconhecidos


def _pergunta_exclui_rosto_direto(pergunta: str) -> bool:
    """Perguntas sobre cena/objetos — devem ir para a IA, não resposta fixa de rosto."""
    t = pergunta.lower()
    return any(p in t for p in (
        'além', 'alem', 'além do', 'alem do', 'além de', 'alem de',
        'o que', 'oque', 'quais', 'quantos', 'quanto',
        'mais vê', 'mais ve', 'mais você', 'mais voce',
        'objeto', 'cena', 'ambiente', 'descrev', 'fala sobre',
        'me diga', 'exceto', 'fora ', 'resto', 'outra coisa', 'outro coisa',
        'tem algo', 'tem alguma coisa', 'o que tem', 'o que há', 'o que ha',
    ))


def _pergunta_sobre_parentesco(pergunta: str) -> Optional[str]:
    """Retorna termo de relação buscado (ex: 'amig') ou None."""
    t = pergunta.lower()
    filtros = (
        ('amig', ('amigo', 'amiga', 'amigos', 'amigas')),
        ('irm', ('irmão', 'irmao', 'irmã', 'irma', 'irmãos', 'irmaos')),
        ('famí', ('família', 'familia', 'parente', 'parentes', 'parentesco')),
        ('coleg', ('colega', 'colegas')),
        ('conhecid', ('conhecido', 'conhecida', 'conhecidos')),
    )
    for stem, palavras in filtros:
        if any(p in t for p in palavras):
            return stem
    return None


def _rostos_por_parentesco(conhecidos: List[Dict], catalogo: Optional[Dict[str, dict]], stem: str) -> List[Dict]:
    resultado = []
    for r in conhecidos:
        nome = r.get('nome', '')
        rel = (r.get('relacao') or '').lower()
        if not rel and catalogo:
            meta = catalogo.get(nome.lower(), {})
            rel = (meta.get('relacao') or '').lower()
        if stem in rel:
            resultado.append(r)
    return resultado


def _pergunta_identifica_rosto(pergunta: str) -> bool:
    """Só responde rosto diretamente em perguntas explícitas de identificação."""
    if _pergunta_exclui_rosto_direto(pergunta):
        return False
    t = pergunta.lower()
    if _pergunta_sobre_parentesco(pergunta):
        return True
    return any(p in t for p in (
        'quem é', 'quem e', 'quem ta', 'quem está', 'quem esta',
        'reconhece', 'conhece algu', 'identifica',
        'tem alguém', 'tem alguem', 'algum rosto', 'alguma pessoa',
        'quem você vê', 'quem voce ve', 'quem vc vê', 'quem vc ve',
        'está vendo algu', 'esta vendo algu', 'estou vendo algu',
        'quem aparece', 'quem tem aí', 'quem tem ai',
    ))


def _responder_pergunta_rostos(
    pergunta: str,
    rostos: List[Dict],
    catalogo: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """Respostas diretas só para identificação explícita ou filtro por parentesco."""
    t = pergunta.lower()
    if any(p in t for p in ('cadastr', 'registr', 'salv', 'memoriz', 'gravar')):
        return None

    filtro_rel = _pergunta_sobre_parentesco(pergunta)
    if not filtro_rel and not _pergunta_identifica_rosto(pergunta):
        return None

    if not rostos:
        if filtro_rel:
            return 'Não vejo ninguém com esse perfil agora.'
        return None

    conhecidos, qtd_desconhecidos = _rostos_unicos(rostos)

    if filtro_rel:
        matches = _rostos_por_parentesco(conhecidos, catalogo, filtro_rel)
        if matches:
            nomes = ', '.join(r.get('nome', '?') for r in matches)
            return f'Sim, {nomes}.'
        return 'Não vejo ninguém com esse perfil agora.'

    if conhecidos and qtd_desconhecidos == 0:
        if len(conhecidos) == 1:
            return f'Sim, {conhecidos[0].get("nome", "alguém")} está na câmera.'
        nomes = ', '.join(r.get('nome', '?') for r in conhecidos)
        return f'Sim, {len(conhecidos)} pessoas: {nomes}.'

    if qtd_desconhecidos > 0 and not conhecidos:
        if qtd_desconhecidos == 1:
            return 'Um rosto desconhecido. Diga "cadastrar essa pessoa" para salvar.'
        return f'{qtd_desconhecidos} rostos desconhecidos.'

    nomes = ', '.join(r.get('nome', '?') for r in conhecidos)
    return f'{nomes} e {qtd_desconhecidos} rosto(s) desconhecido(s).'


def gerar_resposta_voz(
    pergunta: str,
    deteccoes: List[Dict],
    rostos: List[Dict],
    catalogo: Optional[Dict[str, dict]] = None,
) -> str:
    resposta_direta = _responder_pergunta_rostos(pergunta, rostos, catalogo)
    if resposta_direta:
        return _limpar_resposta_fala(resposta_direta)

    if not glm_disponivel():
        return 'Serviço de IA indisponível no momento.'

    contexto = _formatar_contexto_voz(deteccoes, rostos, catalogo)
    user_content = f"""Cena atual da câmera:
{contexto}

Pergunta do usuário: "{pergunta}"

Responda só o que foi perguntado. Se pedir "além de" alguém ou "o que mais", ignore essa pessoa e descreva o resto:"""

    return _limpar_resposta_fala(glm_chat(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT_VOZ},
            {'role': 'user', 'content': user_content},
        ],
        max_tokens=120,
        temperature=0.5,
    ))


def transcrever_audio(audio_bytes: bytes) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY não configurada')

    import openai
    openai.api_key = api_key

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as audio_file:
            result = openai.Audio.transcribe(
                model='whisper-1',
                file=audio_file,
                language='pt',
            )
        return (result.get('text') or '').strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def sintetizar_voz(texto: str) -> bytes:
    client = _get_elevenlabs_client()
    if not client:
        raise ValueError('ELEVENLABS_API_KEY não configurada')

    audio_generator = client.text_to_speech.convert(
        voice_id='pNInz6obpgDQGcFmaJgB',
        text=texto,
        model_id='eleven_multilingual_v2',
        output_format='mp3_44100_128',
    )
    return b''.join(audio_generator)


def processar_pergunta_voz(
    audio_b64: str,
    deteccoes_atuais: list,
    rostos_atuais: list,
    frame_b64: Optional[str] = None,
    cadastro_sessao: Optional[CadastroSessao] = None,
    cadastro_lock: Optional[threading.Lock] = None,
) -> dict:
    audio_bytes = base64.b64decode(audio_b64)

    if len(audio_bytes) < 1200:
        return {
            'transcricao': '',
            'resposta': 'Gravação muito curta. Segure o botão por pelo menos 1 segundo.',
            'audio_b64': None,
            'cadastro_ativo': cadastro_sessao.em_andamento() if cadastro_sessao else False,
        }

    try:
        transcricao = transcrever_audio(audio_bytes)
    except Exception as e:
        err = str(e)
        if 'too short' in err.lower():
            return {
                'transcricao': '',
                'resposta': 'Gravação muito curta. Segure o botão por pelo menos 1 segundo.',
                'audio_b64': None,
                'cadastro_ativo': cadastro_sessao.em_andamento() if cadastro_sessao else False,
            }
        raise
    if not transcricao:
        return {
            'transcricao': '',
            'resposta': 'Não consegui entender sua pergunta. Tente falar novamente.',
            'audio_b64': None,
            'cadastro_ativo': False,
        }

    occhio = get_occhio_instance()
    resposta = None
    cadastro_ativo = False

    if cadastro_sessao and occhio.face_registry:
        frame = None
        if frame_b64:
            try:
                frame = occhio._decodificar_imagem(frame_b64)
            except Exception:
                pass
        lock = cadastro_lock or threading.Lock()
        with lock:
            resposta_cadastro = occhio.face_registry.processar_mensagem(
                cadastro_sessao, transcricao, frame=frame
            )
            if resposta_cadastro:
                resposta = resposta_cadastro
                cadastro_ativo = cadastro_sessao.em_andamento()

    if resposta is None:
        catalogo = None
        if occhio.face_registry:
            catalogo = occhio.face_registry.get_catalogo()
        resposta = gerar_resposta_voz(transcricao, deteccoes_atuais, rostos_atuais, catalogo)
    else:
        resposta = _limpar_resposta_fala(resposta)

    audio_b64_out = None
    audio_erro = None
    try:
        audio_mp3 = sintetizar_voz(resposta)
        audio_b64_out = base64.b64encode(audio_mp3).decode('ascii')
    except Exception as e:
        audio_erro = str(e)
        logger.error(f'ElevenLabs falhou (resposta em texto mantida): {e}')

    return {
        'transcricao': transcricao,
        'resposta': resposta,
        'audio_b64': audio_b64_out,
        'audio_erro': audio_erro,
        'cadastro_ativo': cadastro_ativo,
    }


class _StreamState:
    """Estado compartilhado da sessão WebSocket (frames + detecções)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.deteccoes_atuais: List[Dict] = []
        self.rostos_atuais: List[Dict] = []
        self.ultimo_frame_b64: Optional[str] = None
        self.voz_ocupada: bool = False

    def atualizar(self, frame_b64: str, deteccoes: list, rostos: list) -> None:
        with self.lock:
            self.ultimo_frame_b64 = frame_b64
            self.deteccoes_atuais = list(deteccoes)
            self.rostos_atuais = list(rostos)

    def set_voz_ocupada(self, ocupada: bool) -> None:
        with self.lock:
            self.voz_ocupada = ocupada

    def voz_esta_ocupada(self) -> bool:
        with self.lock:
            return self.voz_ocupada

    def snapshot_voz(self):
        with self.lock:
            return (
                list(self.deteccoes_atuais),
                list(self.rostos_atuais),
                self.ultimo_frame_b64,
            )


def _executar_voz_background(
    ws,
    audio_b64: str,
    stream_state: _StreamState,
    cadastro_sessao: CadastroSessao,
    cadastro_lock: threading.Lock,
) -> None:
    try:
        deteccoes, rostos, frame_b64 = stream_state.snapshot_voz()
        resultado = processar_pergunta_voz(
            audio_b64,
            deteccoes,
            rostos,
            frame_b64=frame_b64,
            cadastro_sessao=cadastro_sessao,
            cadastro_lock=cadastro_lock,
        )
        ws.send(json.dumps({'tipo': 'resposta_voz', **resultado}))
    except Exception as e:
        logger.exception('Erro ao processar voz em background')
        ws.send(json.dumps({
            'tipo': 'resposta_voz',
            'erro': str(e),
            'transcricao': '',
            'resposta': 'Ocorreu um erro ao processar sua pergunta.',
            'audio_b64': None,
            'cadastro_ativo': cadastro_sessao.em_andamento(),
        }))
    finally:
        stream_state.set_voz_ocupada(False)


# Singleton
def get_occhio_instance():
    global _occhio_instance
    if _occhio_instance is None:
        with _initialization_lock:
            if _occhio_instance is None:
                _occhio_instance = OcchioCloud()
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
            "/rostos": "GET - Lista rostos cadastrados",
            "/processar": "POST - Processa imagem",
            "/perguntar": "POST - Pergunta sobre imagem",
            "/estatistica": "POST - Estatísticas da imagem"
        }
    })


@sock.route('/stream')
def stream_ws(ws):
    """WebSocket: frames de vídeo → bounding boxes; áudio → resposta falada."""
    logger.info('Cliente conectado via WebSocket')
    frame_count = 0
    stream_state = _StreamState()
    cadastro_sessao = CadastroSessao()
    cadastro_lock = threading.Lock()
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

            if dados.get('tipo') == 'pergunta_voz':
                audio_b64 = dados.get('audio')
                if not audio_b64:
                    ws.send(json.dumps({
                        'tipo': 'resposta_voz',
                        'erro': "Campo 'audio' não encontrado",
                    }))
                    continue
                stream_state.set_voz_ocupada(True)
                threading.Thread(
                    target=_executar_voz_background,
                    args=(ws, audio_b64, stream_state, cadastro_sessao, cadastro_lock),
                    daemon=True,
                ).start()
                continue

            frame_b64 = dados.get('frame')
            if not frame_b64:
                ws.send(json.dumps({'erro': "Campo 'frame' não encontrado"}))
                continue

            threshold = float(dados.get('threshold', 0.45))
            try:
                occhio = get_occhio_instance()
                resultado = occhio.processar_stream(frame_b64, confidence_threshold=threshold)
                stream_state.atualizar(
                    frame_b64,
                    resultado.get('deteccoes', []),
                    resultado.get('rostos', []),
                )

                for alerta in resultado.get('alertas', []):
                    if stream_state.voz_esta_ocupada():
                        continue
                    msg = alerta.get('mensagem', '')
                    audio_alert = None
                    try:
                        audio_alert = base64.b64encode(sintetizar_voz(msg)).decode('ascii')
                    except Exception:
                        pass
                    ws.send(json.dumps({
                        'tipo': 'alerta_pessoa',
                        'nome': alerta.get('nome'),
                        'mensagem': msg,
                        'audio_b64': audio_alert,
                    }))

                ws.send(json.dumps(resultado))
            except Exception as e:
                logger.exception('Erro ao processar frame WebSocket')
                ws.send(json.dumps({
                    'erro': str(e),
                    'deteccoes': [],
                    'rostos': [],
                    'total': 0,
                    'ms': 0,
                }))
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f'{frame_count} frames processados | último: {resultado.get("ms")}ms')
    except Exception as e:
        logger.exception(f'WebSocket encerrado: {e}')

@app.route('/health', methods=['GET'])
def health():
    try:
        occhio = get_occhio_instance()
        store = occhio.face_store
        db_mysql = getattr(store, 'is_mysql', lambda: False)()
        rostos_cadastrados = []
        if store and hasattr(store, 'list_faces'):
            try:
                rostos_cadastrados = store.list_faces()
            except Exception:
                pass
        return jsonify({
            "sucesso": True,
            "timestamp": int(time.time() * 1000),
            "status": "saudavel",
            "servicos": {
                "detector_yolo": occhio.detector_objetos is not None,
                "detector_faces": occhio.detector_faces is not None,
                "face_database": occhio.face_store is not None,
                "glm_interpreter": getattr(occhio.interpreter, 'glm_disponivel', False),
                "whisper": occhio.whisper_disponivel,
                "modelo": "YOLOv8s + glm-5"
            },
            "rostos": rostos_cadastrados,
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

@app.route('/rostos', methods=['GET'])
def listar_rostos():
    try:
        occhio = get_occhio_instance()
        if not occhio.face_store:
            return jsonify({'sucesso': False, 'erro': 'Armazenamento de rostos indisponível'}), 503
        faces = occhio.face_store.list_faces()
        return jsonify({'sucesso': True, 'total': len(faces), 'rostos': faces})
    except Exception as e:
        return jsonify({'sucesso': False, 'erro': str(e)}), 500

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

def _warmup_em_background():
    """Carrega YOLO/GLM no boot para evitar timeout no primeiro frame."""
    try:
        logger.info('🔧 Warmup: carregando Occhio em background…')
        get_occhio_instance()
        logger.info('✅ Warmup concluído')
    except Exception as e:
        logger.error(f'❌ Warmup falhou: {e}')


threading.Thread(target=_warmup_em_background, daemon=True).start()

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