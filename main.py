"""
Occhio - Sistema de Visão Computacional para Deficientes Visuais
Versão: 5.1.0 - Interpretação de Perguntas Aprimorada
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
import re
import random
import numpy as np
import threading
import base64
import tempfile
from pathlib import Path
from flask import Flask, jsonify, request
from flask_sock import Sock
from typing import Dict, List, Any, Optional

from Utils.glm_client import chat as glm_chat, glm_disponivel
from Utils.face_registry import FaceRegistry, CadastroSessao, detectar_sim
from Utils.face_store import criar_face_store
from Utils.conversation_memory import ConversationMemory

for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy']:
    if var in os.environ:
        os.environ.pop(var, None)

print("🚀 Occhio Cloud v5.1 - Interpretação Aprimorada")
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

_occhio_instance = None
_initialization_lock = threading.Lock()


# ─────────────────────────────────────────────
# PROMPT PRINCIPAL — versão aprimorada
# ─────────────────────────────────────────────

SYSTEM_PROMPT_VOZ = """Você é o Specula, assistente de acessibilidade para deficientes visuais.
Recebe histórico da conversa, lista de objetos detectados pelo YOLO e rostos reconhecidos.

IDENTIDADE:
- Seja um companheiro, não um robô de comandos. Use linguagem natural e calorosa.
- Você tem memória curta: o histórico da conversa está nas mensagens anteriores.
  Use esse contexto. Não repita o que você já disse.

REGRAS (tudo será lido em voz alta):
1. Máximo 2 frases. Prefira 1 frase quando possível.
2. NUNCA repita algo já dito por você ou pelo usuário nessa conversa.
3. Se a pergunta usar "aquele", "isso", "ela", "ele" sem especificar — resolva
   pelo histórico. Não peça para o usuário repetir.
4. NUNCA mencione porcentagem, índice ou número de ID.
5. NUNCA invente objetos ou pessoas fora da lista fornecida.
6. Rostos: use o nome. Nunca "ele/ela" — use o nome ou "a pessoa".
7. NUNCA mencione parentesco (amigo, irmão) salvo se perguntado.
8. Agrupe repetições: "três cadeiras", não "cadeira, cadeira, cadeira".
9. Não comece com "Claro!", "Com certeza!" ou "Posso ver que...". Vá direto.
10. Cena vazia: diga uma vez de forma natural. Não repita se já disse.
"""

LISTAGEM_PATTERNS = (
    'quem você conhece', 'quem voce conhece',
    'quem você sabe', 'quem voce sabe',
    'quantas pessoas você conhece', 'quantas pessoas voce conhece',
    'quem está cadastrado', 'quem esta cadastrado',
    'me fala quem', 'lista quem',
)

PERIGO_PATTERNS = (
    'tem algo perigoso', 'tem alguma coisa perigosa',
    'tem perigo', 'cuidado com algo', 'algo para evitar',
)

OBJETOS_PERIGOSOS = {'knife', 'scissors', 'fire', 'gun', 'sword'}

REMOCAO_PATTERNS = [
    r'(?:esquece|esquecer|esqueca)\s+(?:o|a)?\s*(.+)',
    r'(?:remove|remover|apaga|apagar|exclui|excluir|deleta|deletar)\s+(?:o|a)?\s*(.+)',
    r'para\s+de\s+(?:lembrar|reconhecer)\s+(?:o|a)?\s*(.+)',
]

FALLBACKS_SEM_DETECCOES = [
    'Não estou conseguindo ver nada claramente agora.',
    'A cena está escura ou desfocada. Pode ajustar a câmera?',
    'Não identifiquei nada no momento.',
]

FALLBACKS_SEM_IA = [
    'Estou com dificuldade de pensar agora. Tente em instantes.',
    'Meu serviço de raciocínio está indisponível no momento.',
    'Não consegui processar isso agora. Tente novamente.',
]

NOMES_PT_OBJETOS = {
    'person': 'pessoa', 'chair': 'cadeira', 'table': 'mesa', 'bottle': 'garrafa',
    'cup': 'copo', 'laptop': 'computador', 'cell phone': 'celular', 'knife': 'faca',
    'scissors': 'tesoura',
}


def _eh_confirmacao(texto: str) -> bool:
    return detectar_sim(texto) is True


def _eh_negacao(texto: str) -> bool:
    return detectar_sim(texto) is False


def _detectar_intencao_remocao(texto: str) -> Optional[str]:
    """Retorna o nome a remover ou None."""
    for pat in REMOCAO_PATTERNS:
        m = re.search(pat, texto.lower())
        if m:
            nome = m.group(1).strip().rstrip('?.!')
            if len(nome) >= 2:
                return nome
    return None


def posicao_relativa(x_norm: float, y_norm: float) -> str:
    horiz = (
        "à sua esquerda" if x_norm < 0.35 else
        "à sua direita" if x_norm > 0.65 else
        "à sua frente"
    )
    return horiz


def _nome_objeto_pt(nome: str) -> str:
    return NOMES_PT_OBJETOS.get(nome.lower(), nome)


def _responder_posicao(pergunta: str, deteccoes: List[Dict], memoria: Optional[ConversationMemory]) -> Optional[str]:
    t = pergunta.lower()
    if 'onde' not in t and 'posição' not in t and 'posicao' not in t:
        return None

    candidatos = []
    for d in deteccoes:
        nome = d.get('nome', '')
        if nome and (nome.lower() in t or _nome_objeto_pt(nome).lower() in t):
            candidatos.append(d)

    if not candidatos and memoria:
        for obj in memoria.objetos_recentes():
            if obj.lower() in t or _nome_objeto_pt(obj).lower() in t:
                for d in deteccoes:
                    if d.get('nome', '').lower() == obj.lower():
                        candidatos.append(d)
                        break

    if not candidatos and memoria and memoria.objetos_recentes():
        ultimo = memoria.objetos_recentes()[-1]
        for d in deteccoes:
            if d.get('nome', '').lower() == ultimo.lower():
                candidatos.append(d)
                break

    if not candidatos:
        return None

    d = candidatos[0]
    cx = float(d.get('x', 0.5)) + float(d.get('w', 0)) / 2
    cy = float(d.get('y', 0.5)) + float(d.get('h', 0)) / 2
    nome_pt = _nome_objeto_pt(d.get('nome', 'objeto'))
    return f'{nome_pt.capitalize()}, {posicao_relativa(cx, cy)}.'


def _responder_listagem_rostos(pergunta: str, catalogo: Optional[Dict]) -> Optional[str]:
    if not any(p in pergunta.lower() for p in LISTAGEM_PATTERNS):
        return None
    if not catalogo:
        return 'Não conheço ninguém cadastrado ainda.'
    nomes = [v['nome'] for v in catalogo.values()]
    if not nomes:
        return 'Não conheço ninguém cadastrado ainda.'
    if len(nomes) <= 5:
        return f'Conheço {", ".join(nomes)}.'
    return f'Conheço {", ".join(nomes[:5])} e mais {len(nomes) - 5} pessoas.'


def _responder_perigo(pergunta: str, deteccoes: List[Dict]) -> Optional[str]:
    if not any(p in pergunta.lower() for p in PERIGO_PATTERNS):
        return None
    perigosos = [
        _nome_objeto_pt(d['nome'])
        for d in deteccoes
        if d.get('nome', '').lower() in OBJETOS_PERIGOSOS
    ]
    if perigosos:
        return f'Atenção: detectei {", ".join(perigosos)} na cena.'
    return 'Não vejo nada perigoso no momento.'


# ─────────────────────────────────────────────
# CLASSIFICADOR DE INTENÇÃO — substitui as
# funções fragmentadas de detecção de padrões
# ─────────────────────────────────────────────

class IntencaoPergunta:
    """
    Classifica a intenção da pergunta em uma única passagem,
    evitando listas sobrepostas e classificações conflitantes.
    """

    # Intenções mutuamente exclusivas, em ordem de prioridade
    CADASTRO      = "cadastro"
    PARENTESCO    = "parentesco"
    IDENTIFICACAO = "identificacao"
    CENA_GERAL    = "cena_geral"
    CENA_EXCLUSAO = "cena_exclusao"   # "além de X, o que mais?"
    DESCRICAO     = "descricao"

    # Termos que disparam cada intenção
    _CADASTRO_TERMOS = (
        'cadastrar', 'registrar', 'salvar', 'memorizar', 'gravar', 'adicionar',
    )
    _PARENTESCO_TERMOS = (
        'amigo', 'amiga', 'amigos', 'amigas',
        'irmão', 'irmao', 'irmã', 'irma', 'irmãos', 'irmaos',
        'família', 'familia', 'parente', 'parentes',
        'colega', 'colegas', 'conhecido', 'conhecida',
    )
    _IDENTIFICACAO_TERMOS = (
        'quem é', 'quem e', 'quem ta', 'quem está', 'quem esta',
        'quem você vê', 'quem voce ve', 'quem vc vê', 'quem vc ve',
        'quem aparece', 'quem tem aí', 'quem tem ai',
        'tem alguém', 'tem alguem', 'algum rosto', 'alguma pessoa',
        'reconhece', 'conhece algu', 'identifica',
        'está vendo algu', 'estou vendo algu',
    )
    _EXCLUSAO_TERMOS = (
        'além', 'alem', 'além do', 'alem do', 'além de', 'alem de',
        'exceto', 'fora ', 'além disso', 'alem disso',
        'o que mais', 'oque mais', 'mais alguma', 'mais algum',
        'outra coisa', 'outro coisa', 'resto', 'demais',
    )
    _CENA_TERMOS = (
        'o que', 'oque', 'quais', 'quantos', 'quanto',
        'tem algo', 'tem alguma', 'o que tem', 'o que há', 'o que ha',
        'objeto', 'cena', 'ambiente', 'lugar',
        'mais vê', 'mais ve', 'mais você', 'mais voce',
    )
    _DESCRICAO_TERMOS = (
        'descrev', 'fala sobre', 'me diga', 'explica', 'conta',
        'como está', 'como esta', 'como é', 'como e',
    )

    @classmethod
    def classificar(cls, pergunta: str) -> tuple[str, list[str]]:
        """
        Retorna (intenção, termos_excluidos).
        termos_excluidos: objetos/nomes mencionados na pergunta que não devem ser repetidos.
        """
        t = pergunta.lower()

        # 1. Cadastro tem prioridade máxima
        if any(p in t for p in cls._CADASTRO_TERMOS):
            return cls.CADASTRO, []

        # 2. Parentesco — pergunta sobre relação social
        if any(p in t for p in cls._PARENTESCO_TERMOS):
            return cls.PARENTESCO, []

        # 3. Exclusão — "além de X, o que mais?" → extrair X para não repetir
        if any(p in t for p in cls._EXCLUSAO_TERMOS):
            excluidos = cls._extrair_termos_excluidos(t)
            return cls.CENA_EXCLUSAO, excluidos

        # 4. Identificação de rosto
        if any(p in t for p in cls._IDENTIFICACAO_TERMOS):
            return cls.IDENTIFICACAO, []

        # 5. Pergunta geral sobre cena/objetos
        if any(p in t for p in cls._CENA_TERMOS):
            return cls.CENA_GERAL, []

        # 6. Descrição livre
        if any(p in t for p in cls._DESCRICAO_TERMOS):
            return cls.DESCRICAO, []

        # 7. Fallback — trata como descrição geral
        return cls.DESCRICAO, []

    @classmethod
    def _extrair_termos_excluidos(cls, texto: str) -> list[str]:
        """
        Extrai os objetos/nomes que o usuário quer EXCLUIR da resposta.
        Ex: "além da garrafa e da cadeira" → ["garrafa", "cadeira"]
        """
        import re
        # Remove artigos e preposições comuns antes de extrair substantivos
        limpo = re.sub(
            r'\b(além|alem|de|da|do|dos|das|e|exceto|fora|além de|o que|oque|mais)\b',
            ' ', texto
        )
        # Pega palavras com 4+ letras (evita artigos residuais)
        candidatos = re.findall(r'\b[a-záéíóúàâêôãõüç]{4,}\b', limpo)
        # Descarta verbos/advérbios comuns que não são objetos
        stopwords = {
            'mais', 'outro', 'outra', 'algum', 'alguma', 'coisa', 'cena',
            'você', 'voce', 'ainda', 'para', 'como', 'isso', 'esse', 'essa',
            'vejo', 'veja', 'veja', 'tens', 'temos', 'diga', 'fala', 'fale',
            'além', 'alem', 'qualquer', 'nenhum', 'nenhuma',
        }
        return [c for c in candidatos if c not in stopwords]


# ─────────────────────────────────────────────
# FORMATAÇÃO DE CONTEXTO — sem redundâncias
# ─────────────────────────────────────────────

def _formatar_contexto_voz(
    deteccoes: List[Dict],
    rostos: List[Dict],
    termos_excluidos: List[str] = None,
    catalogo: Optional[Dict[str, dict]] = None,
) -> str:
    """
    Formata o contexto enviado ao GLM de forma concisa e sem repetição.
    Aplica filtro de exclusão quando a intenção é CENA_EXCLUSAO.
    """
    partes = []
    termos_excluidos = [t.lower() for t in (termos_excluidos or [])]

    # ── Objetos detectados ──────────────────────────────────────────
    if deteccoes:
        contagem: Dict[str, int] = {}
        for d in deteccoes:
            nome = d.get('nome', '?').lower()
            # Aplica filtro de exclusão
            if termos_excluidos and any(ex in nome or nome in ex for ex in termos_excluidos):
                continue
            contagem[nome] = contagem.get(nome, 0) + 1

        if contagem:
            itens = []
            for nome, qtd in sorted(contagem.items(), key=lambda x: -x[1]):
                itens.append(f"{qtd}× {nome}" if qtd > 1 else nome)
            partes.append("Objetos: " + ", ".join(itens))
        else:
            partes.append("Objetos: nenhum além dos mencionados.")
    else:
        partes.append("Objetos: nenhum detectado.")

    # ── Rostos ──────────────────────────────────────────────────────
    if rostos:
        conhecidos, desconhecidos = _rostos_unicos(rostos)
        nomes_conhecidos = [r.get('nome', '?') for r in conhecidos]

        # Filtrar rostos excluídos (por nome)
        if termos_excluidos:
            nomes_conhecidos = [
                n for n in nomes_conhecidos
                if not any(ex in n.lower() for ex in termos_excluidos)
            ]

        partes_rosto = []
        if nomes_conhecidos:
            partes_rosto.append(", ".join(nomes_conhecidos))
        if desconhecidos > 0:
            partes_rosto.append(
                f"{desconhecidos} rosto(s) desconhecido(s)"
            )
        if partes_rosto:
            partes.append("Rostos: " + "; ".join(partes_rosto))
        else:
            partes.append("Rostos: nenhum (após filtro).")
    else:
        partes.append("Rostos: nenhum.")

    return "\n".join(partes)


# ─────────────────────────────────────────────
# RESPOSTAS DIRETAS — apenas para casos simples
# onde a IA não agrega valor
# ─────────────────────────────────────────────

def _resposta_direta(
    intencao: str,
    pergunta: str,
    rostos: List[Dict],
    catalogo: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """
    Retorna resposta de texto fixo APENAS quando a IA não é necessária.
    Nos demais casos retorna None para que a IA responda.
    """
    conhecidos, qtd_desconhecidos = _rostos_unicos(rostos)

    if intencao == IntencaoPergunta.IDENTIFICACAO:
        if not rostos:
            return "Não vejo nenhuma pessoa na câmera agora."
        if conhecidos and qtd_desconhecidos == 0:
            nomes = ", ".join(r.get("nome", "?") for r in conhecidos)
            return f"{nomes} está{'o' if len(conhecidos) > 1 else ''} na câmera."
        if not conhecidos and qtd_desconhecidos > 0:
            return None
        # Misto — deixa a IA formular
        return None

    if intencao == IntencaoPergunta.PARENTESCO:
        t = pergunta.lower()
        stem = None
        mapeamento = {
            'amig': ('amigo', 'amiga', 'amigos', 'amigas'),
            'irm':  ('irmão', 'irmao', 'irmã', 'irma'),
            'famí': ('família', 'familia', 'parente', 'parentes'),
            'coleg': ('colega', 'colegas'),
            'conhecid': ('conhecido', 'conhecida'),
        }
        for s, palavras in mapeamento.items():
            if any(p in t for p in palavras):
                stem = s
                break

        if stem and catalogo:
            matches = [
                r for r in conhecidos
                if stem in (catalogo.get(r.get('nome', '').lower(), {}).get('relacao', '') or '').lower()
            ]
            if matches:
                nomes = ", ".join(r.get('nome', '?') for r in matches)
                return f"Sim, {nomes}."
            return "Não vejo ninguém com esse perfil agora."

        # Sem catálogo ou stem não mapeado — deixa a IA
        return None

    # Para todas as outras intenções a IA responde
    return None


# ─────────────────────────────────────────────
# GERADOR DE RESPOSTA PRINCIPAL
# ─────────────────────────────────────────────

def _limpar_resposta_fala(texto: str) -> str:
    """Remove artefatos inadequados para narração em voz."""
    import re
    texto = re.sub(r'\d+\s*%', '', texto)
    texto = re.sub(r'\(\s*\)', '', texto)
    # Remove introduções desnecessárias
    texto = re.sub(
        r'^(claro[,!]?\s*|com certeza[,!]?\s*|sim[,!]?\s*claro[,!]?\s*|'
        r'posso ver que\s*|eu vejo que\s*|com base na imagem[,]?\s*)',
        '', texto, flags=re.IGNORECASE
    )
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Garante capitalização
    if texto:
        texto = texto[0].upper() + texto[1:]
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


def gerar_resposta_voz(
    pergunta: str,
    deteccoes: List[Dict],
    rostos: List[Dict],
    memoria: Optional[ConversationMemory] = None,
    catalogo: Optional[Dict[str, dict]] = None,
    face_registry: Optional[FaceRegistry] = None,
    cadastro_sessao: Optional[CadastroSessao] = None,
) -> tuple:
    """
    Pipeline principal de geração de resposta.
    Retorna (resposta, sugestao_cadastro_pendente).
    """

    def _gravar(resposta: str, sug: bool = False) -> tuple:
        if memoria and resposta:
            memoria.adicionar_turno(pergunta, resposta, deteccoes, rostos)
        return _limpar_resposta_fala(resposta), sug

    resp = _responder_listagem_rostos(pergunta, catalogo)
    if resp:
        return _gravar(resp)

    resp = _responder_perigo(pergunta, deteccoes)
    if resp:
        return _gravar(resp)

    intencao, termos_excluidos = IntencaoPergunta.classificar(pergunta)

    resp = _responder_posicao(pergunta, deteccoes, memoria)
    if resp:
        return _gravar(resp)

    if intencao == IntencaoPergunta.IDENTIFICACAO and face_registry and cadastro_sessao:
        sugestao = face_registry.sugerir_cadastro_se_desconhecido(rostos, cadastro_sessao)
        if sugestao:
            return _gravar(sugestao, sug=True)

    resposta_direta = _resposta_direta(intencao, pergunta, rostos, catalogo)
    if resposta_direta:
        return _gravar(resposta_direta)

    if not deteccoes and not rostos and intencao in (
        IntencaoPergunta.CENA_GERAL, IntencaoPergunta.CENA_EXCLUSAO, IntencaoPergunta.DESCRICAO,
    ):
        return _gravar(random.choice(FALLBACKS_SEM_DETECCOES))

    if not glm_disponivel():
        return _gravar(random.choice(FALLBACKS_SEM_IA))

    contexto = _formatar_contexto_voz(deteccoes, rostos, termos_excluidos, catalogo)

    instrucao_extra = ""
    if intencao == IntencaoPergunta.CENA_EXCLUSAO and termos_excluidos:
        excluidos_fmt = ", ".join(termos_excluidos)
        instrucao_extra = (
            f"\nIMPORTANTE: O usuário já sabe sobre '{excluidos_fmt}'. "
            f"NÃO mencione isso. Fale apenas do que está na lista acima."
        )
    elif intencao == IntencaoPergunta.CENA_GERAL:
        instrucao_extra = "\nDescreva os objetos presentes de forma natural e agrupada."
    elif intencao == IntencaoPergunta.DESCRICAO:
        instrucao_extra = "\nFaça uma descrição breve e natural da cena."

    if memoria and memoria.objetos_recentes():
        objs = ", ".join(memoria.objetos_recentes())
        instrucao_extra += f"\nObjetos mencionados recentemente: {objs}."

    user_content = (
        f"Cena atual:\n{contexto}"
        f"{instrucao_extra}\n\n"
        f"Pergunta: \"{pergunta}\"\n\n"
        "Responda de forma direta e natural, sem repetir o que o usuário disse:"
    )

    historico = memoria.contexto_para_glm() if memoria else []

    resposta = glm_chat(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT_VOZ},
            *historico,
            {'role': 'user', 'content': user_content},
        ],
        max_tokens=120,
        temperature=0.4,
    )

    resposta = _limpar_resposta_fala(resposta)

    if memoria:
        memoria.adicionar_turno(pergunta, resposta, deteccoes, rostos)

    return resposta, False


def _montar_resposta_voz(
    transcricao: str,
    resposta: str,
    cadastro_ativo: bool = False,
) -> dict:
    return {
        'transcricao': transcricao,
        'resposta': _limpar_resposta_fala(resposta),
        'audio_b64': None,
        'cadastro_ativo': cadastro_ativo,
    }


def _finalizar_resposta_voz(transcricao: str, resposta: str, cadastro_ativo: bool = False) -> dict:
    resultado = _montar_resposta_voz(transcricao, resposta, cadastro_ativo)
    audio_erro = None
    try:
        audio_mp3 = sintetizar_voz(resultado['resposta'])
        resultado['audio_b64'] = base64.b64encode(audio_mp3).decode('ascii')
    except Exception as e:
        audio_erro = str(e)
        logger.error(f'ElevenLabs falhou (resposta em texto mantida): {e}')
    resultado['audio_erro'] = audio_erro
    return resultado


# ─────────────────────────────────────────────
# CLASSE PRINCIPAL — sem alterações estruturais
# ─────────────────────────────────────────────

class OcchioCloud:
    """Classe principal com API padronizada"""

    def __init__(self, api_key=None):
        try:
            logger.info("🚀 INICIANDO OCCHIO CLOUD v5.1")

            self.whisper_disponivel = bool(
                os.getenv('OPENAI_API_KEY', '').strip()
            )
            self.glm_disponivel = glm_disponivel()

            logger.info(f"📦 Whisper: {'✅ DISPONÍVEL' if self.whisper_disponivel else '❌ INDISPONÍVEL'}")
            logger.info(f"📦 GLM-5: {'✅ DISPONÍVEL' if self.glm_disponivel else '❌ INDISPONÍVEL'}")

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
        try:
            from Detectors.yolo_detector import YOLODetector
            self.detector_objetos = YOLODetector()
            logger.info("✅ YOLO inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar YOLO: {e}")
            self.detector_objetos = self._criar_detector_local()

    def _inicializar_faces(self):
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
        try:
            from Utils.interpreter import Interpreter
            self.interpreter = Interpreter()
            logger.info("✅ Interpreter inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar interpreter: {e}")
            self.interpreter = self._criar_interpreter_local()

    def _criar_detector_local(self):
        class DetectorLocal:
            def detectar_com_bbox(self, frame, confidence_threshold=0.5):
                return []
        return DetectorLocal()

    def _criar_interpreter_local(self):
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
        self.detector_objetos = self._criar_detector_local()
        self.detector_faces = None
        self.face_store = None
        self.face_registry = None
        self.interpreter = self._criar_interpreter_local()
        self.glm_disponivel = False
        self.whisper_disponivel = False
        logger.warning("⚠️ Sistema em modo emergência")

    def _decodificar_imagem(self, dados_imagem: str) -> np.ndarray:
        try:
            if isinstance(dados_imagem, str):
                if dados_imagem.startswith('data:image'):
                    dados_imagem = dados_imagem.split(',')[1]
                bytes_imagem = base64.b64decode(dados_imagem)
                nparr = np.frombuffer(bytes_imagem, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Falha ao decodificar imagem")
                altura, largura = frame.shape[:2]
                tamanho_maximo = 1280
                if largura > tamanho_maximo or altura > tamanho_maximo:
                    escala = min(tamanho_maximo / largura, tamanho_maximo / altura)
                    nova_largura = int(largura * escala)
                    nova_altura = int(altura * escala)
                    frame = cv2.resize(frame, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
                return frame
            else:
                raise ValueError("dados_imagem deve ser string base64")
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar imagem: {e}")
            raise

    def _normalizar_coordenadas(self, bbox: Dict, largura_imagem: int, altura_imagem: int) -> Dict:
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
            return {'x': 0, 'y': 0, 'largura': 0, 'altura': 0}
        except Exception:
            return {'x': 0, 'y': 0, 'largura': 0, 'altura': 0}

    def _processar_imagem_interna(self, dados_imagem: str) -> Dict[str, Any]:
        inicio_tempo = time.time()
        try:
            frame = self._decodificar_imagem(dados_imagem)
            altura_imagem, largura_imagem = frame.shape[:2]
            deteccoes = []
            if self.detector_objetos and hasattr(self.detector_objetos, 'detectar_com_bbox'):
                deteccoes_brutas = self.detector_objetos.detectar_com_bbox(frame, confidence_threshold=0.5)
                for i, det in enumerate(deteccoes_brutas):
                    caixa_normalizada = self._normalizar_coordenadas(det['bbox'], largura_imagem, altura_imagem)
                    deteccoes.append({
                        'id': i + 1,
                        'nome': det['class'],
                        'confianca': float(det['confidence']),
                        'caixa': caixa_normalizada,
                        'caixa_pixels': det['bbox']
                    })
            return {
                'deteccoes': deteccoes,
                'info_imagem': {'largura': largura_imagem, 'altura': altura_imagem},
                'tempo_processamento_ms': int((time.time() - inicio_tempo) * 1000)
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
            resultado = self._processar_imagem_interna(dados_imagem)
            resposta = {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": resultado['tempo_processamento_ms'],
                "dados": {
                    "resumo": {"total_objetos": len(resultado['deteccoes'])},
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
            resultado_deteccao = self._processar_imagem_interna(dados_imagem)
            deteccoes = resultado_deteccao['deteccoes']
            objetos_para_interpreter = [
                {'nome': det['nome'], 'confianca': det['confianca'], 'quantidade': 1}
                for det in deteccoes[:10]
            ]
            resposta_chat = None
            correlacao_com_imagem = False
            confianca_resposta = 0.0
            if self.interpreter and hasattr(self.interpreter, 'perguntar_sobre_imagem'):
                resultado_interpreter = self.interpreter.perguntar_sobre_imagem(
                    pergunta=pergunta,
                    objetos_detectados=objetos_para_interpreter,
                    faces_nomes=[]
                )
                if isinstance(resultado_interpreter, dict):
                    resposta_chat = resultado_interpreter.get('resposta')
                    correlacao_com_imagem = resultado_interpreter.get('correlacao_com_imagem', False)
                    confianca_resposta = float(resultado_interpreter.get('confianca', 0.0))
                else:
                    resposta_chat = str(resultado_interpreter)
            else:
                resposta_chat = "Interpreter não disponível."
            deteccoes_relevantes = [
                {'nome': det['nome'], 'confianca': det['confianca']}
                for det in sorted(deteccoes, key=lambda x: x['confianca'], reverse=True)[:3]
            ]
            return {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": int((time.time() - inicio_tempo) * 1000),
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
            resultado = self._processar_imagem_interna(dados_imagem)
            deteccoes = resultado['deteccoes']
            contagem_objetos = {}
            confiancas = []
            for det in deteccoes:
                nome = det['nome']
                contagem_objetos[nome] = contagem_objetos.get(nome, 0) + 1
                confiancas.append(det['confianca'])
            if confiancas:
                estatisticas_confianca = {
                    'media':   round(float(np.mean(confiancas)), 3),
                    'maxima':  round(float(np.max(confiancas)), 3),
                    'minima':  round(float(np.min(confiancas)), 3),
                    'mediana': round(float(np.median(confiancas)), 3)
                }
            else:
                estatisticas_confianca = {'media': 0, 'maxima': 0, 'minima': 0, 'mediana': 0}
            return {
                "sucesso": True,
                "timestamp": int(time.time() * 1000),
                "tempo_processamento_ms": int((time.time() - inicio_tempo) * 1000),
                "dados": {
                    "resumo": {
                        "total_objetos": len(deteccoes),
                        "objetos_unicos": len(contagem_objetos)
                    },
                    "contagem_objetos": contagem_objetos,
                    "estatisticas_confianca": estatisticas_confianca,
                    "amostra_deteccoes": deteccoes[:5]
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


# ─────────────────────────────────────────────
# VOZ — Whisper + GLM + ElevenLabs
# ─────────────────────────────────────────────

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
    memoria: Optional[ConversationMemory] = None,
    stream_state: Optional['_StreamState'] = None,
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
            'resposta': 'Não consegui entender. Tente falar novamente.',
            'audio_b64': None,
            'cadastro_ativo': False,
        }

    occhio = get_occhio_instance()
    resposta = None
    cadastro_ativo = False

    frame = None
    if frame_b64:
        try:
            frame = occhio._decodificar_imagem(frame_b64)
        except Exception:
            pass

    # ── Remoção pendente (confirmação) ──────────────────────────────
    if stream_state and stream_state.remocao_pendente:
        nome_pendente = stream_state.remocao_pendente
        if _eh_confirmacao(transcricao):
            stream_state.remocao_pendente = None
            removido = False
            if occhio.face_registry:
                removido = occhio.face_registry.remover_por_nome(nome_pendente)
            msg = (
                f'Pronto, não vou mais reconhecer {nome_pendente}.'
                if removido else
                'Não encontrei ninguém com esse nome cadastrado.'
            )
            if memoria:
                memoria.adicionar_turno(transcricao, msg, deteccoes_atuais, rostos_atuais)
            return _finalizar_resposta_voz(transcricao, msg)
        if _eh_negacao(transcricao):
            stream_state.remocao_pendente = None
            return _finalizar_resposta_voz(transcricao, 'Ok, mantive o cadastro.')
        stream_state.remocao_pendente = None

    # ── Nova intenção de remoção ────────────────────────────────────
    nome_remover = _detectar_intencao_remocao(transcricao)
    if nome_remover and stream_state and not (cadastro_sessao and cadastro_sessao.em_andamento()):
        stream_state.remocao_pendente = nome_remover
        msg = f'Tem certeza que quer que eu esqueça {nome_remover}?'
        if memoria:
            memoria.adicionar_turno(transcricao, msg, deteccoes_atuais, rostos_atuais)
        return _finalizar_resposta_voz(transcricao, msg)

    # ── Confirmação de cadastro sugerido ─────────────────────────────
    if stream_state and stream_state.sugestao_cadastro_pendente and occhio.face_registry:
        if _eh_confirmacao(transcricao):
            stream_state.sugestao_cadastro_pendente = False
            lock = cadastro_lock or threading.Lock()
            with lock:
                resposta = occhio.face_registry.iniciar_cadastro_confirmado(
                    cadastro_sessao, frame=frame
                )
                cadastro_ativo = cadastro_sessao.em_andamento() if cadastro_sessao else False
            if resposta and memoria:
                memoria.adicionar_turno(transcricao, resposta, deteccoes_atuais, rostos_atuais)
            return _finalizar_resposta_voz(transcricao, resposta or 'Não consegui iniciar o cadastro.', cadastro_ativo)
        if _eh_negacao(transcricao):
            stream_state.sugestao_cadastro_pendente = False
            return _finalizar_resposta_voz(transcricao, 'Ok, sem problemas.')
        stream_state.sugestao_cadastro_pendente = False

    # ── Cadastro em andamento ─────────────────────────────────────────
    if cadastro_sessao and occhio.face_registry:
        lock = cadastro_lock or threading.Lock()
        with lock:
            resposta_cadastro = occhio.face_registry.processar_mensagem(
                cadastro_sessao, transcricao, frame=frame
            )
            if resposta_cadastro:
                resposta = resposta_cadastro
                cadastro_ativo = cadastro_sessao.em_andamento()

    # ── Resposta principal (GLM + memória) ────────────────────────────
    if resposta is None:
        catalogo = occhio.face_registry.get_catalogo() if occhio.face_registry else None
        resposta, sugestao = gerar_resposta_voz(
            transcricao,
            deteccoes_atuais,
            rostos_atuais,
            memoria=memoria,
            catalogo=catalogo,
            face_registry=occhio.face_registry,
            cadastro_sessao=cadastro_sessao,
        )
        if stream_state and sugestao:
            stream_state.sugestao_cadastro_pendente = True
    else:
        resposta = _limpar_resposta_fala(resposta)
        if memoria:
            memoria.adicionar_turno(transcricao, resposta, deteccoes_atuais, rostos_atuais)

    return _finalizar_resposta_voz(transcricao, resposta, cadastro_ativo)


class _StreamState:
    """Estado compartilhado da sessão WebSocket."""

    def __init__(self):
        self.lock = threading.Lock()
        self.deteccoes_atuais: List[Dict] = []
        self.rostos_atuais: List[Dict] = []
        self.ultimo_frame_b64: Optional[str] = None
        self.voz_ocupada: bool = False
        self.memoria = ConversationMemory()
        self.remocao_pendente: Optional[str] = None
        self.sugestao_cadastro_pendente: bool = False

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
            memoria=stream_state.memoria,
            stream_state=stream_state,
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


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

def get_occhio_instance():
    global _occhio_instance
    if _occhio_instance is None:
        with _initialization_lock:
            if _occhio_instance is None:
                _occhio_instance = OcchioCloud()
    return _occhio_instance


# ─────────────────────────────────────────────
# ROTAS FLASK — sem alterações
# ─────────────────────────────────────────────

@app.route('/')
def dashboard():
    if DASHBOARD_HTML.exists():
        return DASHBOARD_HTML.read_text(encoding='utf-8')
    return jsonify({'erro': 'Dashboard não encontrado'}), 404


@app.route('/api')
def index():
    return jsonify({
        "app": "Occhio Cloud API",
        "versao": "5.1.0",
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
        return jsonify(resultado), 200 if resultado.get('sucesso') else 500
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
        return jsonify(resultado), 200 if resultado.get('sucesso') else 500
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
        return jsonify(resultado), 200 if resultado.get('sucesso') else 500
    except Exception as e:
        logger.error(f"❌ Erro em /estatistica: {e}")
        return jsonify({
            "sucesso": False,
            "timestamp": int(time.time() * 1000),
            "erro": str(e),
            "codigo": "ERRO_SERVIDOR"
        }), 500


# ─────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────

def _warmup_em_background():
    try:
        logger.info('🔧 Warmup: carregando Occhio em background…')
        get_occhio_instance()
        logger.info('✅ Warmup concluído')
    except Exception as e:
        logger.error(f'❌ Warmup falhou: {e}')


threading.Thread(target=_warmup_em_background, daemon=True).start()

if __name__ == "__main__":
    porta = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud v5.1 na porta {porta}")
    logger.info(f"   Dashboard: http://localhost:{porta}/")
    logger.info(f"   WebSocket: ws://localhost:{porta}/stream")
    logger.info("🔧 Inicializando componentes...")
    get_occhio_instance()
    logger.info(f" Servindo com Flask (WebSocket habilitado) na porta {porta}...")
    app.run(host='0.0.0.0', port=porta, debug=False, threaded=True)