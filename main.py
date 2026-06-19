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
import signal
from pathlib import Path
from flask import Flask, jsonify, request
from flask_sock import Sock
from typing import Dict, List, Any, Optional

from Utils.glm_client import chat as glm_chat, glm_disponivel
from Utils.face_registry import (
    FaceRegistry,
    detectar_intencao_cadastro,
    detectar_sim,
    recarregar_estado_facial,
)
from Detectors.face_detector import _DLIB_LOCK
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
            return None
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


def _formatar_objetos_camera(deteccoes: List[Dict]) -> str:
    if not deteccoes:
        return ""
    itens = []
    for d in deteccoes:
        nome = d.get('nome', '?')
        conf = d.get('confianca', d.get('confidence'))
        if conf is not None:
            pct = int(round(float(conf) * 100))
            itens.append(f"{nome} ({pct}% confiança)")
        else:
            itens.append(nome)
    return ", ".join(itens)


def _formatar_rostos_camera(rostos: List[Dict], catalogo: Optional[Dict[str, dict]] = None) -> str:
    if not rostos:
        return ""
    conhecidos, desconhecidos = _rostos_unicos(rostos)
    partes = []
    for r in conhecidos:
        nome = r.get('nome', '?')
        rel = None
        if catalogo:
            rel = catalogo.get(nome.lower(), {}).get('relacao')
        partes.append(f"{nome} ({rel})" if rel else nome)
    if desconhecidos > 0:
        s = "s" if desconhecidos > 1 else ""
        partes.append(f"{desconhecidos} rosto{s} desconhecido{s}")
    return ", ".join(partes)


def _formatar_catalogo_banco(catalogo: Optional[Dict[str, dict]]) -> str:
    if not catalogo:
        return ""
    nomes_relacoes = [
        f"{v.get('nome', '?')} ({v.get('relacao', 'sem relação definida')})"
        for v in catalogo.values() if v.get('nome')
    ]
    return ", ".join(nomes_relacoes) if nomes_relacoes else ""


def montar_catalogo_str(catalogo) -> str:
    """Serializa o catálogo de forma segura para o prompt do GLM."""
    if not catalogo:
        return 'Nenhuma pessoa cadastrada.'

    try:
        linhas = []
        for v in catalogo.values():
            nome = str(v.get('nome', '')).strip()
            relacao = str(v.get('relacao', 'não definida')).strip()
            avisar = 'sim' if v.get('avisar') else 'não'

            if nome:
                linhas.append(f'- {nome} | relação: {relacao} | avisar quando aparecer: {avisar}')

        if not linhas:
            return 'Nenhuma pessoa cadastrada.'

        return '\n'.join(linhas)

    except Exception as e:
        logger.error(f'❌ Erro ao serializar catálogo: {e} | catalogo type: {type(catalogo)}')
        return 'Erro ao carregar lista de pessoas cadastradas.'


def _contar_pessoas_catalogo(catalogo) -> int:
    if not catalogo:
        return 0
    return sum(1 for v in catalogo.values() if str(v.get('nome', '')).strip())


def _cadastro_ativo(occhio) -> bool:
    return getattr(occhio, '_cadastro_pendente', None) is not None


def _valor_cadastro_campo(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        t = val.strip()
        if not t or t.lower() in ('null', 'none', 'n/a'):
            return None
        return t
    return str(val).strip() or None


def _bool_avisar_cadastro(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', 'sim', 's', 'yes', '1')
    return bool(val) if val is not None else False


def extrair_dados_cadastro_via_glm(transcricao: str) -> Optional[dict]:
    """Pede ao GLM para extrair nome, relação e aviso da frase do usuário."""
    transcricao = (transcricao or '').strip()
    if not transcricao or not glm_disponivel():
        return None

    prompt = f"""O usuário quer cadastrar uma pessoa. Extraia as informações da frase abaixo.

Frase: "{transcricao}"

Responda APENAS com JSON válido neste formato exato, sem texto adicional:
{{
  "nome": "nome da pessoa ou null se não mencionado",
  "relacao": "relação com a pessoa ou null se não mencionado",
  "avisar": true ou false (true se o usuário quer ser avisado quando a pessoa aparecer, false se não quer ou não mencionou)
}}

Se o nome não foi mencionado, retorne null no campo nome.
Se a relação não foi mencionada, retorne null no campo relacao."""

    try:
        resposta = glm_chat([{'role': 'user', 'content': prompt}], max_tokens=120, temperature=0.1)
        texto = re.sub(r'```json|```', '', resposta).strip()
        return json.loads(texto)
    except Exception as e:
        logger.error('❌ Erro ao extrair dados de cadastro: %s', e)
        return None


def _salvar_cadastro_rosto(occhio, nome: str, relacao: str, avisar: bool) -> str:
    if not occhio.face_registry or not occhio.face_registry.face_store:
        return 'Cadastro facial indisponível no momento.'

    encoding, erro = occhio.face_registry.capturar_encoding_async(occhio)
    if erro:
        return erro
    if encoding is None:
        return 'Não encontrei um rosto na câmera. Pode se aproximar?'

    with _DLIB_LOCK:
        if occhio.face_registry.face_store.face_exists(encoding):
            existente = occhio.face_registry.face_store.find_existing_name(encoding)
            if existente:
                return f'{existente} já está cadastrado.'
            return 'Essa pessoa já parece estar cadastrada.'

    ok = occhio.face_registry.face_store.save_face(
        face_encoding=encoding,
        nome=nome,
        relacao=relacao,
        label=nome,
        avisar=avisar,
    )
    if not ok:
        return 'Não consegui salvar. Tenta de novo.'

    recarregar_estado_facial(occhio)
    occhio._cadastro_pendente = None
    fala = f'Pronto, cadastrei {nome} como {relacao}.'
    if avisar:
        fala += ' Vou avisar quando ele aparecer.'
    return fala


def _processar_fluxo_cadastro(occhio, transcricao: str) -> str:
    """Extrai dados via GLM, completa pendências e salva quando possível."""
    if not hasattr(occhio, '_cadastro_pendente'):
        occhio._cadastro_pendente = None

    pendente = dict(occhio._cadastro_pendente or {})
    dados_novos = extrair_dados_cadastro_via_glm(transcricao)

    nome = pendente.get('nome') or (_valor_cadastro_campo(dados_novos.get('nome')) if dados_novos else None)
    relacao = pendente.get('relacao') or (_valor_cadastro_campo(dados_novos.get('relacao')) if dados_novos else None)
    avisar = pendente.get('avisar', False)
    if dados_novos and 'avisar' in dados_novos:
        avisar = _bool_avisar_cadastro(dados_novos.get('avisar'))

    if nome and relacao:
        return _salvar_cadastro_rosto(occhio, nome, relacao, avisar)

    if relacao and not nome:
        occhio._cadastro_pendente = {'relacao': relacao, 'avisar': avisar}
        return 'Qual o nome da pessoa?'

    if nome and not relacao:
        occhio._cadastro_pendente = {'nome': nome, 'avisar': avisar}
        return f'Qual sua relação com {nome}?'

    occhio._cadastro_pendente = {'avisar': avisar}
    return 'Qual o nome e sua relação com essa pessoa?'


def executar_acao_banco(acao: dict, occhio) -> str:
    tipo = acao.get('tipo')
    nome_atual = acao.get('nome_atual', '')
    novo_valor = acao.get('novo_valor', '')

    store = occhio.face_registry.face_store if occhio.face_registry else None
    if not store:
        return 'erro: sem acesso ao banco'

    ok = False
    if tipo == 'renomear':
        ok = store.rename_face(nome_atual, novo_valor)
    elif tipo == 'deletar':
        ok = store.delete_face(nome_atual)
    elif tipo == 'atualizar_relacao':
        ok = store.update_relacao(nome_atual, novo_valor)
    elif tipo == 'atualizar_aviso':
        ok = store.update_aviso(
            nome_atual,
            str(novo_valor).lower() in ('sim', 'true', '1', 's', 'yes'),
        )
    else:
        return 'erro: tipo de ação desconhecido'

    if ok:
        recarregar_estado_facial(occhio)

    return 'ok' if ok else 'não encontrado'


def _processar_resposta_glm_acao(
    resposta: str,
    occhio,
    pergunta: Optional[str] = None,
) -> str:
    """Separa AÇÃO/FALA do GLM, executa escrita no banco e retorna só o texto falado."""
    texto = (resposta or '').strip()
    if not re.match(r'^A[CÇ]ÃO:', texto, re.IGNORECASE):
        return _limpar_resposta_fala(texto)

    try:
        partes = re.split(r'\n\s*FALA:\s*', texto, maxsplit=1, flags=re.IGNORECASE)
        if len(partes) < 2:
            match = re.match(
                r'^A[CÇ]ÃO:\s*(\{.*?\})\s*(?:\n\s*)?(?:FALA:\s*)?(.*)$',
                texto,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                return _limpar_resposta_fala('Não consegui processar essa solicitação.')
            acao_json_str, fala = match.group(1).strip(), match.group(2).strip()
        else:
            acao_json_str = re.sub(r'^A[CÇ]ÃO:\s*', '', partes[0], flags=re.IGNORECASE).strip()
            fala = partes[1].strip()

        acao = json.loads(acao_json_str)
        if acao.get('tipo') == 'cadastrar' and pergunta:
            logger.info('📝 GLM sinalizou cadastro — iniciando fluxo via extração')
            return _limpar_resposta_fala(_processar_fluxo_cadastro(occhio, pergunta))

        resultado = executar_acao_banco(acao, occhio)
        logger.info('🗄️ Ação banco: %s → %s', acao, resultado)

        if fala:
            return _limpar_resposta_fala(fala)
        if resultado == 'ok':
            return 'Pronto, alteração feita.'
        if resultado == 'não encontrado':
            return 'Não encontrei essa pessoa no cadastro.'
        return 'Não consegui fazer essa alteração agora.'
    except Exception as e:
        logger.error('Erro ao processar ação GLM: %s', e)
        if re.search(r'FALA:\s*', texto, re.IGNORECASE):
            fala = re.split(r'FALA:\s*', texto, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            if fala:
                return _limpar_resposta_fala(fala)
        return _limpar_resposta_fala('Não consegui processar essa solicitação.')


def _montar_system_prompt_voz(
    deteccoes: List[Dict],
    rostos: List[Dict],
    catalogo: Optional[Dict[str, dict]] = None,
) -> str:
    objetos_str = _formatar_objetos_camera(deteccoes)
    rostos_visiveis_str = _formatar_rostos_camera(rostos, catalogo)
    catalogo_str = montar_catalogo_str(catalogo)
    total_banco = _contar_pessoas_catalogo(catalogo)
    tem_cadastrados = total_banco > 0
    tem_rostos_agora = bool(rostos)

    return f"""Você é o Specula, assistente de visão computacional com memória de pessoas.
Sua resposta será lida em voz alta — máximo 2 frases curtas, português brasileiro, direto ao ponto.
Nunca mencione porcentagens na resposta. Nunca invente objetos ou pessoas fora do contexto abaixo.

## O que a câmera está vendo agora
{objetos_str if objetos_str else "Nenhum objeto detectado no momento."}

## Rostos reconhecidos neste momento
{rostos_visiveis_str if rostos_visiveis_str else "Nenhum rosto reconhecido no frame atual."}

## Pessoas cadastradas no banco
Total: {total_banco}
{catalogo_str}

IMPORTANTE: Responda APENAS com base nas informações acima.
Se não há pessoas cadastradas, diga isso claramente.
Nunca invente nomes, números ou dados que não estejam listados acima.

## Ações que você pode executar quando o usuário pedir
- Cadastrar uma pessoa nova na câmera
- Renomear uma pessoa: muda o nome no banco
- Deletar uma pessoa: remove do banco permanentemente
- Atualizar relação: muda como a pessoa é classificada (amigo, familiar, colega…)
- Atualizar aviso: ativa ou desativa o alerta quando a pessoa aparecer na câmera

Quando o usuário pedir para cadastrar alguém na câmera, responda SEMPRE neste formato:
AÇÃO: {{"tipo": "cadastrar", "nome_atual": null, "novo_valor": null}}
FALA: Vou cadastrar a pessoa na câmera.

Quando o usuário pedir para modificar, renomear, deletar ou atualizar dados de uma pessoa cadastrada, responda SEMPRE neste formato exato:
AÇÃO: {{"tipo": "renomear" | "deletar" | "atualizar_relacao" | "atualizar_aviso", "nome_atual": "nome no banco", "novo_valor": "novo valor"}}
FALA: <o que dizer ao usuário em voz alta, confirmando o que foi feito>

Para qualquer outra pergunta que não envolva modificar o banco ou cadastrar alguém, responda normalmente sem o prefixo AÇÃO.

## Estado do sistema
- Pessoas cadastradas: {"sim" if tem_cadastrados else "não"} ({total_banco} no banco)
- Rostos visíveis agora: {"sim" if tem_rostos_agora else "não"}

Responda de forma natural e direta. Use o contexto completo acima para entender o que o usuário quer saber — seja sobre o que está na câmera agora, sobre quem você conhece no banco, ou sobre qualquer combinação disso. Quando executar uma ação, confirme o que foi feito na resposta falada (campo FALA)."""


def gerar_resposta_voz(
    pergunta: str,
    deteccoes: List[Dict],
    rostos: List[Dict],
    memoria: Optional[ConversationMemory] = None,
    catalogo: Optional[Dict[str, dict]] = None,
    face_registry: Optional[FaceRegistry] = None,
    occhio=None,
) -> tuple:
    """
    Pipeline principal de geração de resposta.
    Retorna (resposta, sugestao_cadastro_pendente).
    """

    def _gravar(resposta: str, sug: bool = False) -> tuple:
        if memoria and resposta:
            memoria.adicionar_turno(pergunta, resposta, deteccoes, rostos)
        return _limpar_resposta_fala(resposta), sug

    pergunta = (pergunta or '').strip()
    if not pergunta:
        return _gravar('Não consegui entender sua pergunta.')

    intencao, termos_excluidos = IntencaoPergunta.classificar(pergunta)

    instancia = occhio or get_occhio_instance()
    if intencao == IntencaoPergunta.IDENTIFICACAO and face_registry:
        sugestao = face_registry.sugerir_cadastro_se_desconhecido(rostos, instancia)
        if sugestao:
            return _gravar(sugestao, sug=True)

    if not glm_disponivel():
        resp = _responder_perigo(pergunta, deteccoes)
        if resp:
            return _gravar(resp)
        resp = _responder_posicao(pergunta, deteccoes, memoria)
        if resp:
            return _gravar(resp)
        resposta_direta = _resposta_direta(intencao, pergunta, rostos, catalogo)
        if resposta_direta:
            return _gravar(resposta_direta)
        if (
            not deteccoes and not rostos and not catalogo
            and intencao in (
                IntencaoPergunta.CENA_GERAL, IntencaoPergunta.CENA_EXCLUSAO, IntencaoPergunta.DESCRICAO,
            )
        ):
            return _gravar(random.choice(FALLBACKS_SEM_DETECCOES))
        return _gravar(random.choice(FALLBACKS_SEM_IA))

    instrucao_extra = ''
    if intencao == IntencaoPergunta.CENA_EXCLUSAO and termos_excluidos:
        excluidos_fmt = ', '.join(termos_excluidos)
        instrucao_extra = (
            f'O usuário já sabe sobre "{excluidos_fmt}". '
            f'Não mencione isso na resposta.\n'
        )
    if memoria and memoria.objetos_recentes():
        objs = ', '.join(memoria.objetos_recentes())
        instrucao_extra += f'Objetos mencionados recentemente na conversa: {objs}.\n'

    system_prompt = _montar_system_prompt_voz(deteccoes, rostos, catalogo)
    historico = memoria.contexto_para_glm() if memoria else []

    user_content = (
        f'Pergunta do usuário: "{pergunta}"\n\n'
        f'{instrucao_extra}'
        'Responda de forma direta e natural em português brasileiro, '
        'usando o contexto do sistema para decidir o que é relevante. '
        'Não repita a pergunta na resposta.'
    )

    logger.info(
        '🎤 GLM — pergunta: "%s" | objetos: %d | rostos: %d | cadastrados: %d',
        pergunta,
        len(deteccoes),
        len(rostos),
        len(catalogo) if catalogo else 0,
    )

    resposta = glm_chat(
        messages=[
            {'role': 'system', 'content': system_prompt},
            *historico,
            {'role': 'user', 'content': user_content},
        ],
        max_tokens=200,
        temperature=0.4,
    )

    resposta = _processar_resposta_glm_acao(resposta, instancia, pergunta=pergunta)

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
            self._encoding_request = threading.Event()
            self._encoding_lock = threading.Lock()
            self._encoding_result = None
            self._encoding_erro = None
            self._cadastro_pendente = None
            self._cadastro_lock = threading.Lock()

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
            logger.info('✅ Interpreter inicializado')
        except Exception as e:
            logger.warning(f'⚠️ Interpreter falhou no warmup: {e} — sistema continua sem ele')
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
        self._encoding_request = threading.Event()
        self._encoding_lock = threading.Lock()
        self._encoding_result = None
        self._encoding_erro = None
        self._cadastro_pendente = None
        self._cadastro_lock = threading.Lock()
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

            capturando_encoding = self._encoding_request.is_set()
            if capturando_encoding and self.detector_faces:
                encoding, erro = self.detector_faces.extrair_encoding_principal(frame)
                with self._encoding_lock:
                    if encoding is not None:
                        self._encoding_result = encoding
                        self._encoding_erro = None
                    else:
                        self._encoding_result = None
                        self._encoding_erro = erro or 'Não encontrei rosto claro na câmera.'
                self._encoding_request.clear()

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
            if self.face_registry and not capturando_encoding:
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
    memoria: Optional[ConversationMemory] = None,
    stream_state: Optional['_StreamState'] = None,
) -> dict:
    audio_bytes = base64.b64decode(audio_b64)

    if len(audio_bytes) < 1200:
        return {
            'transcricao': '',
            'resposta': 'Gravação muito curta. Segure o botão por pelo menos 1 segundo.',
            'audio_b64': None,
            'cadastro_ativo': False,
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
                'cadastro_ativo': False,
            }
        raise

    if not transcricao:
        return {
            'transcricao': '',
            'resposta': 'Não consegui entender. Tente falar novamente.',
            'audio_b64': None,
            'cadastro_ativo': False,
        }

    logger.info('🎤 Transcrição: "%s"', transcricao)

    transcricao_atual = transcricao
    try:
        occhio = get_occhio_instance()

        # ── Remoção pendente (confirmação) ──────────────────────────────
        if stream_state and stream_state.remocao_pendente:
            nome_pendente = stream_state.remocao_pendente
            if _eh_confirmacao(transcricao_atual):
                stream_state.remocao_pendente = None
                removido = False
                if occhio.face_registry:
                    removido = occhio.face_registry.remover_por_nome(nome_pendente)
                if removido:
                    recarregar_estado_facial(occhio)
                msg = (
                    f'Pronto, não vou mais reconhecer {nome_pendente}.'
                    if removido else
                    'Não encontrei ninguém com esse nome cadastrado.'
                )
                if memoria:
                    memoria.adicionar_turno(transcricao_atual, msg, deteccoes_atuais, rostos_atuais)
                return _finalizar_resposta_voz(transcricao_atual, msg)
            if _eh_negacao(transcricao_atual):
                stream_state.remocao_pendente = None
                return _finalizar_resposta_voz(transcricao_atual, 'Ok, mantive o cadastro.')
            stream_state.remocao_pendente = None

        # ── Cadastro pendente — completa antes do GLM normal ────────────
        if _cadastro_ativo(occhio) and occhio.face_registry:
            with occhio._cadastro_lock:
                resposta = _processar_fluxo_cadastro(occhio, transcricao_atual)
            cadastro_ativo = _cadastro_ativo(occhio)
            if memoria:
                memoria.adicionar_turno(transcricao_atual, resposta, deteccoes_atuais, rostos_atuais)
            return _finalizar_resposta_voz(transcricao_atual, resposta, cadastro_ativo)

        # ── Nova intenção de remoção ────────────────────────────────────
        nome_remover = _detectar_intencao_remocao(transcricao_atual)
        if nome_remover and stream_state and not _cadastro_ativo(occhio):
            stream_state.remocao_pendente = nome_remover
            msg = f'Tem certeza que quer que eu esqueça {nome_remover}?'
            if memoria:
                memoria.adicionar_turno(transcricao_atual, msg, deteccoes_atuais, rostos_atuais)
            return _finalizar_resposta_voz(transcricao_atual, msg)

        # ── Confirmação de cadastro sugerido ─────────────────────────────
        if stream_state and stream_state.sugestao_cadastro_pendente and occhio.face_registry:
            if _eh_confirmacao(transcricao_atual):
                stream_state.sugestao_cadastro_pendente = False
                with occhio._cadastro_lock:
                    resposta = _processar_fluxo_cadastro(occhio, transcricao_atual)
                cadastro_ativo = _cadastro_ativo(occhio)
                if memoria:
                    memoria.adicionar_turno(transcricao_atual, resposta, deteccoes_atuais, rostos_atuais)
                return _finalizar_resposta_voz(transcricao_atual, resposta, cadastro_ativo)
            if _eh_negacao(transcricao_atual):
                stream_state.sugestao_cadastro_pendente = False
                return _finalizar_resposta_voz(transcricao_atual, 'Ok, sem problemas.')
            stream_state.sugestao_cadastro_pendente = False

        # ── Intenção explícita de cadastro ───────────────────────────────
        if occhio.face_registry and detectar_intencao_cadastro(transcricao_atual):
            with occhio._cadastro_lock:
                resposta = _processar_fluxo_cadastro(occhio, transcricao_atual)
            cadastro_ativo = _cadastro_ativo(occhio)
            if memoria:
                memoria.adicionar_turno(transcricao_atual, resposta, deteccoes_atuais, rostos_atuais)
            return _finalizar_resposta_voz(transcricao_atual, resposta, cadastro_ativo)

        # ── Resposta principal (GLM + memória) ────────────────────────────
        catalogo = None
        if occhio.face_registry:
            catalogo = occhio.face_registry.get_catalogo()
        resposta, sugestao = gerar_resposta_voz(
            transcricao_atual,
            deteccoes_atuais,
            rostos_atuais,
            memoria=memoria,
            catalogo=catalogo,
            face_registry=occhio.face_registry,
            occhio=occhio,
        )
        if stream_state and sugestao:
            stream_state.sugestao_cadastro_pendente = True

        cadastro_ativo = _cadastro_ativo(occhio)
        return _finalizar_resposta_voz(transcricao_atual, resposta, cadastro_ativo)

    except Exception as e:
        logger.error(f'❌ Erro no pipeline de voz: {e}')
        try:
            recarregar_estado_facial(
                occhio if 'occhio' in locals() and occhio else get_occhio_instance(),
            )
        except Exception:
            pass
        occhio_err = occhio if 'occhio' in locals() and occhio else get_occhio_instance()
        return _finalizar_resposta_voz(
            transcricao_atual,
            'Ocorreu um erro interno. Pode repetir?',
            _cadastro_ativo(occhio_err),
        )


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
) -> None:
    try:
        deteccoes, rostos, frame_b64 = stream_state.snapshot_voz()
        resultado = processar_pergunta_voz(
            audio_b64,
            deteccoes,
            rostos,
            frame_b64=frame_b64,
            memoria=stream_state.memoria,
            stream_state=stream_state,
        )
        try:
            ws.send(json.dumps({'tipo': 'resposta_voz', **resultado}))
        except Exception as send_err:
            err_msg = str(send_err).lower()
            if 'connection closed' in err_msg or 'closed' in err_msg:
                logger.warning('Cliente desconectou antes de receber resposta de voz')
            else:
                raise
    except Exception as e:
        logger.exception('Erro ao processar voz em background')
        try:
            ws.send(json.dumps({
                'tipo': 'resposta_voz',
                'erro': str(e),
                'transcricao': '',
                'resposta': 'Ocorreu um erro ao processar sua pergunta.',
                'audio_b64': None,
                'cadastro_ativo': _cadastro_ativo(get_occhio_instance()),
            }))
        except Exception:
            logger.warning('Não foi possível enviar erro de voz — cliente desconectado')
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
    try:
        while True:
            try:
                mensagem = ws.receive()
            except Exception as recv_err:
                if 'Connection closed' in str(recv_err) or 'connection closed' in str(recv_err).lower():
                    logger.info('WebSocket encerrado pelo cliente')
                    break
                raise
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
                    args=(ws, audio_b64, stream_state),
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
                    try:
                        ws.send(json.dumps({
                            'tipo': 'alerta_pessoa',
                            'nome': alerta.get('nome'),
                            'mensagem': msg,
                            'audio_b64': audio_alert,
                        }))
                    except Exception:
                        break
                else:
                    try:
                        ws.send(json.dumps(resultado))
                    except Exception:
                        break
            except Exception as e:
                logger.exception('Erro ao processar frame WebSocket')
                try:
                    ws.send(json.dumps({
                        'erro': str(e),
                        'deteccoes': [],
                        'rostos': [],
                        'total': 0,
                        'ms': 0,
                    }))
                except Exception:
                    break
                continue

            frame_count += 1
            ultimo_ms = resultado.get('ms', 0)
            if ultimo_ms > 400:
                logger.warning(f'⚠️ Latência alta: {ultimo_ms}ms no frame {frame_count}')
    except Exception as e:
        if 'Connection closed' in str(e) or 'connection closed' in str(e).lower():
            logger.info('WebSocket encerrado pelo cliente')
        else:
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


def _handler_segv(signum, frame):
    logger.error('❌ SIGSEGV capturado — tentando recuperar')


if __name__ == "__main__":
    signal.signal(signal.SIGSEGV, _handler_segv)
    porta = int(os.getenv('PORT', '8080'))
    logger.info(f"🚀 Iniciando Occhio Cloud v5.1 na porta {porta}")
    logger.info(f"   Dashboard: http://localhost:{porta}/")
    logger.info(f"   WebSocket: ws://localhost:{porta}/stream")
    logger.info("🔧 Inicializando componentes...")
    get_occhio_instance()
    logger.info(f" Servindo com Flask (WebSocket habilitado) na porta {porta}...")
    app.run(host='0.0.0.0', port=porta, debug=False, threaded=True)