"""
Fluxo conversacional de cadastro de rostos conhecidos.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PALAVRAS_CADASTRO = (
    'cadastrar', 'registrar', 'adicionar', 'salvar', 'gravar',
    'conhecer', 'memorizar', 'lembrar', 'anotar',
)
PALAVRAS_SIM = ('sim', 'quero', 'pode', 'claro', 'por favor', 'com certeza', 'positivo', 'ok')
PALAVRAS_NAO = ('não', 'nao', 'negativo', 'deixa', 'dispensa', 'sem aviso')


@dataclass
class CadastroSessao:
    estado: str = 'idle'
    encoding: Optional[np.ndarray] = None
    nome: str = ''
    relacao: str = ''
    iniciado_em: float = 0.0

    def reset(self) -> None:
        self.estado = 'idle'
        self.encoding = None
        self.nome = ''
        self.relacao = ''
        self.iniciado_em = 0.0

    def em_andamento(self) -> bool:
        return self.estado != 'idle'


def detectar_intencao_cadastro(texto: str) -> bool:
    t = texto.lower()
    frases_diretas = (
        'cadastrar rosto', 'cadastrar essa pessoa', 'cadastrar esse rosto',
        'cadastra essa pessoa', 'cadastra esse rosto', 'cadastra o rosto',
        'salvar rosto', 'salvar essa pessoa', 'salvar esse rosto',
        'registrar rosto', 'registrar essa pessoa', 'registrar esse rosto',
        'identificar rosto', 'identificar essa pessoa', 'identificar esse rosto',
        'memorizar rosto', 'memorizar essa pessoa', 'memorizar esse rosto',
        'quero cadastrar', 'quero salvar', 'quero registrar',
        'adicionar rosto', 'adicionar essa pessoa', 'gravar rosto',
        'conhecer essa pessoa', 'lembrar essa pessoa', 'anotar essa pessoa',
    )
    if any(frase in t for frase in frases_diretas):
        return True
    if any(p in t for p in PALAVRAS_CADASTRO) and any(
        x in t for x in ('pessoa', 'rosto', 'face', 'ele', 'ela', 'essa', 'esse', 'frente', 'câmera', 'camera')
    ):
        return True
    return False


def detectar_sim(texto: str) -> Optional[bool]:
    t = texto.lower().strip()
    if any(p in t for p in PALAVRAS_SIM):
        return True
    if any(p in t for p in PALAVRAS_NAO):
        return False
    return None


def extrair_nome(texto: str) -> str:
    t = texto.strip()
    for prefixo in (
        r'^(o nome é|o nome e|nome é|nome e|se chama|chama|é|e)\s+',
        r'^(meu amigo|minha amiga|meu|minha)\s+',
    ):
        t = re.sub(prefixo, '', t, flags=re.IGNORECASE).strip()
    t = t.strip('.,!?\"\'')
    return t[:120] if t else texto.strip()[:120]


class FaceRegistry:
    """Gerencia cadastro conversacional e alertas de rostos conhecidos."""

    def __init__(self, face_detector=None, face_store=None):
        self.face_detector = face_detector
        self.face_store = face_store
        self._alertas_recentes: Dict[str, float] = {}
        self.alerta_cooldown = 25.0

    def recarregar_rostos(self) -> int:
        if not self.face_detector or not self.face_store:
            return 0
        try:
            encodings, nomes, _ = self.face_store.load_face_encodings()
            self.face_detector.carregar_encodings(encodings, nomes)
            logger.info(f'📥 {len(nomes)} rosto(s) carregado(s) do banco')
            return len(nomes)
        except Exception as e:
            logger.error(f'Erro ao recarregar rostos: {e}')
            return 0

    def contar_rostos_no_frame(self, frame) -> int:
        if not self.face_detector or not hasattr(self.face_detector, 'contar_rostos'):
            return 0
        return self.face_detector.contar_rostos(frame)

    def capturar_encoding(self, frame) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not self.face_detector:
            return None, 'Detector facial não disponível.'
        if not hasattr(self.face_detector, 'extrair_encoding_principal'):
            return None, 'Detector facial incompleto.'

        qtd = self.contar_rostos_no_frame(frame)
        if qtd == 0:
            return None, 'Não encontrei nenhum rosto na câmera. Posicione a pessoa de frente e tente de novo.'
        if qtd > 1:
            return None, (
                f'Estou vendo {qtd} rostos na câmera. '
                'Enquadre apenas uma pessoa para cadastrar e tente de novo.'
            )

        encoding, msg = self.face_detector.extrair_encoding_principal(frame)
        if encoding is None:
            return None, msg or 'Não encontrei um rosto claro na câmera. Posicione a pessoa de frente e tente de novo.'
        return encoding, None

    def processar_mensagem(self, sessao: CadastroSessao, texto: str, frame=None) -> Optional[str]:
        texto = (texto or '').strip()
        if not texto:
            return None

        if sessao.estado == 'idle':
            if not detectar_intencao_cadastro(texto):
                return None
            if frame is None:
                return 'Preciso da imagem da câmera para cadastrar. Mantenha a câmera ativa e tente de novo.'
            encoding, erro = self.capturar_encoding(frame)
            if erro:
                return erro
            if self.face_store and self.face_store.face_exists(encoding):
                return 'Essa pessoa já parece estar cadastrada. Quer cadastrar outra pessoa?'
            sessao.encoding = encoding
            sessao.estado = 'aguardando_nome'
            sessao.iniciado_em = time.time()
            return 'Perfeito! Qual o nome da pessoa à sua frente?'

        if sessao.estado == 'aguardando_nome':
            nome = extrair_nome(texto)
            if len(nome) < 2:
                return 'Não entendi o nome. Pode repetir?'
            sessao.nome = nome
            sessao.estado = 'aguardando_relacao'
            return f'Qual é a sua relação ou parentesco com {nome}?'

        if sessao.estado == 'aguardando_relacao':
            relacao = texto.strip()
            if len(relacao) < 2:
                return 'Pode me dizer como você conhece essa pessoa? Por exemplo: amigo, irmão, colega.'
            sessao.relacao = relacao
            sessao.estado = 'aguardando_aviso'
            return (
                f'Perfeito! {sessao.nome} será cadastrado(a) como seu {relacao}. '
                f'Quer que eu avise quando eu vê-lo(a) de novo?'
            )

        if sessao.estado == 'aguardando_aviso':
            decisao = detectar_sim(texto)
            if decisao is None:
                return 'Responda sim ou não: quer que eu avise quando eu vir essa pessoa de novo?'
            return self._finalizar_cadastro(sessao, avisar=decisao)

        return None

    def _finalizar_cadastro(self, sessao: CadastroSessao, avisar: bool) -> str:
        nome = sessao.nome
        relacao = sessao.relacao
        encoding = sessao.encoding

        if encoding is None or not nome:
            sessao.reset()
            return 'O cadastro foi interrompido. Podemos começar de novo quando quiser.'

        if not self.face_store:
            sessao.reset()
            return 'O banco de dados não está configurado. Configure as variáveis DB_* no servidor.'

        ok = self.face_store.save_face(
            face_encoding=encoding,
            nome=nome,
            relacao=relacao,
            label=nome,
            avisar=avisar,
        )
        sessao.reset()

        if not ok:
            return f'Não consegui salvar {nome} no banco. Tente novamente.'

        self.recarregar_rostos()
        if avisar:
            return f'Pronto! {nome} está cadastrado(a). Vou avisar quando eu vê-lo(a) de novo.'
        return f'Pronto! {nome} está cadastrado(a). Não vou avisar quando aparecer.'

    def cancelar(self, sessao: CadastroSessao) -> str:
        if sessao.em_andamento():
            sessao.reset()
            return 'Cadastro cancelado.'
        return ''

    def detectar_faces_stream(self, frame) -> List[Dict[str, Any]]:
        if not self.face_detector or not hasattr(self.face_detector, 'detectar_faces_bbox'):
            return []
        return self.face_detector.detectar_faces_bbox(frame)

    def verificar_alertas(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.face_store:
            return []
        agora = time.time()
        alertas = []
        nomes_aviso = self.face_store.get_nomes_com_aviso()

        for face in faces:
            nome = face.get('nome', '')
            if not nome or nome.lower() in ('desconhecido', 'unknown'):
                continue
            if nome not in nomes_aviso:
                continue
            ultimo = self._alertas_recentes.get(nome, 0)
            if agora - ultimo < self.alerta_cooldown:
                continue
            self._alertas_recentes[nome] = agora
            relacao = self.face_store.get_relacao(nome) or 'conhecido'
            alertas.append({
                'nome': nome,
                'relacao': relacao,
                'mensagem': f'Estou vendo {nome}, seu {relacao}.',
            })
        return alertas
