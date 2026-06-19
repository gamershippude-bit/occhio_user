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
CONFIRMACOES = (
    'sim', 's', 'pode', 'pode ser', 'quero', 'vai', 'claro',
    'beleza', 'tá', 'ta', 'ok', 'isso', 'isso mesmo', 'exato',
    'por favor', 'pfv', 'pf', 'vamo', 'vamos',
    'si', 'com certeza', 'positivo', 'confirmo', 'uhum', 'aham',
)
NEGACOES = (
    'não', 'nao', 'n', 'negativo', 'deixa', 'deixa pra la',
    'esquece', 'tanto faz', 'sem querer', 'não quero', 'nao quero',
    'cancela', 'dispensa', 'sem aviso', 'de jeito nenhum',
)
PALAVRAS_SIM = CONFIRMACOES
PALAVRAS_NAO = NEGACOES
RESPOSTAS_SIM_CURTAS = frozenset({'s', 'si', 'sim', 'n', 'nao', 'não'})
CADASTRO_TIMEOUT = 20.0


@dataclass
class CadastroSessao:
    estado: str = 'idle'
    encoding: Optional[np.ndarray] = None
    nome: str = ''
    relacao: str = ''
    iniciado_em: float = 0.0
    tentativas_aviso: int = 0

    def reset(self) -> None:
        self.estado = 'idle'
        self.encoding = None
        self.nome = ''
        self.relacao = ''
        self.iniciado_em = 0.0
        self.tentativas_aviso = 0

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
    t = re.sub(r'[^\w\sáàâãéêíóôõúüç]', ' ', texto.lower().strip())
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return None

    if t in RESPOSTAS_SIM_CURTAS:
        return t in ('s', 'si', 'sim')

    tokens = set(t.split())
    if tokens & {'sim', 'si', 's'} and not tokens & {'nao', 'não', 'n'}:
        return True
    if tokens & {'nao', 'não', 'n'} and not tokens & {'sim', 'si'}:
        return False

    if re.search(r'\bn[aã]o\b', t):
        return False
    if re.search(r'\bsim\b', t):
        return True

    for p in PALAVRAS_NAO:
        if p in t and len(p) >= 3:
            return False
    for p in PALAVRAS_SIM:
        if p in t and len(p) >= 2:
            return True
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
        self._catalog: Dict[str, dict] = {}
        self.alerta_cooldown = 25.0

    def recarregar_rostos(self) -> int:
        if not self.face_detector or not self.face_store:
            return 0
        try:
            encodings, nomes, _ = self.face_store.load_face_encodings()
            self.face_detector.carregar_encodings(encodings, nomes)
            if hasattr(self.face_store, 'get_faces_catalog'):
                self._catalog = self.face_store.get_faces_catalog()
            self._alertas_recentes.clear()
            logger.info(f'📥 {len(nomes)} rosto(s) do banco: {list(self._catalog.values())}')
            return len(nomes)
        except Exception as e:
            logger.error(f'Erro ao recarregar rostos: {e}')
            return 0

    def _meta_rosto(self, nome: str) -> Optional[dict]:
        if not nome:
            return None
        meta = self._catalog.get(nome.lower())
        if meta:
            return meta
        if self.face_store:
            rel = self.face_store.get_relacao(nome)
            if rel:
                return {'nome': nome, 'relacao': rel}
        return None

    def contar_rostos_no_frame(self, frame) -> int:
        if not self.face_detector or not hasattr(self.face_detector, 'contar_rostos'):
            return 0
        return self.face_detector.contar_rostos(frame)

    def capturar_encoding_do_frame_atual(self, occhio, timeout: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Solicita captura de encoding via loop de stream (evita dlib na thread de voz)."""
        if not occhio or not hasattr(occhio, '_encoding_request'):
            return None, 'Captura de rosto indisponível no momento.'

        with occhio._encoding_lock:
            occhio._encoding_result = None
            occhio._encoding_erro = None
        occhio._encoding_request.set()

        inicio = time.time()
        while time.time() - inicio < timeout:
            with occhio._encoding_lock:
                if occhio._encoding_result is not None:
                    return occhio._encoding_result, None
                if occhio._encoding_erro is not None:
                    return None, occhio._encoding_erro
            time.sleep(0.05)

        occhio._encoding_request.clear()
        return None, 'Timeout ao capturar rosto da câmera. Tente novamente.'

    def capturar_encoding(self, frame=None, occhio=None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if occhio is not None:
            return self.capturar_encoding_do_frame_atual(occhio)

        if not self.face_detector:
            return None, 'Detector facial não disponível.'
        if frame is None:
            return None, 'Preciso da imagem da câmera para cadastrar. Mantenha a câmera ativa e tente de novo.'
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

    def processar_mensagem(self, sessao: CadastroSessao, texto: str, frame=None, occhio=None) -> Optional[str]:
        texto = (texto or '').strip()
        if not texto:
            return None

        if sessao.em_andamento() and sessao.iniciado_em:
            if time.time() - sessao.iniciado_em > CADASTRO_TIMEOUT:
                sessao.reset()
                return None

        if detectar_intencao_cadastro(texto):
            if sessao.em_andamento():
                logger.warning('⚠️ Cadastro anterior incompleto detectado — resetando estado')
                sessao.reset()

        if sessao.estado == 'idle':
            if not detectar_intencao_cadastro(texto):
                return None
            encoding, erro = self.capturar_encoding(frame=frame, occhio=occhio)
            if erro:
                return erro
            if self.face_store and self.face_store.face_exists(encoding):
                existente = None
                if hasattr(self.face_store, 'find_existing_name'):
                    existente = self.face_store.find_existing_name(encoding)
                if existente:
                    meta = self._meta_rosto(existente)
                    rel = meta.get('relacao') if meta else None
                    if rel:
                        return f'{existente} já está cadastrado como {rel}.'
                    return f'{existente} já está cadastrado.'
                return 'Essa pessoa já parece estar cadastrada.'
            sessao.encoding = encoding
            sessao.estado = 'aguardando_nome'
            sessao.iniciado_em = time.time()
            return 'Qual o nome dessa pessoa?'

        if sessao.estado == 'aguardando_nome':
            nome = extrair_nome(texto)
            if len(nome) < 2:
                return 'Repita o nome, por favor.'
            sessao.nome = nome
            sessao.estado = 'aguardando_relacao'
            return f'Qual sua relação com {nome}?'

        if sessao.estado == 'aguardando_relacao':
            relacao = texto.strip()
            if len(relacao) < 2:
                return 'Exemplo: amigo, irmão ou colega.'
            sessao.relacao = relacao
            sessao.estado = 'aguardando_aviso'
            sessao.tentativas_aviso = 0
            return f'{sessao.nome} será salvo como {relacao}. Avisar quando aparecer?'

        if sessao.estado == 'aguardando_aviso':
            decisao = detectar_sim(texto)
            if decisao is None:
                sessao.tentativas_aviso += 1
                if sessao.tentativas_aviso >= 2:
                    return self._finalizar_cadastro(sessao, avisar=False)
                return 'Diga sim ou não.'
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
            return f'Pronto, vou lembrar de {nome} quando aparecer.'
        return f'Pronto, {nome} cadastrado.'

    def sugerir_cadastro_se_desconhecido(
        self, rostos: list, cadastro_sessao: CadastroSessao
    ) -> Optional[str]:
        """Sugere cadastro quando há rosto desconhecido e nenhum cadastro em andamento."""
        desconhecidos = [r for r in rostos if not r.get('conhecido')]
        if not desconhecidos or cadastro_sessao.em_andamento():
            return None
        return 'Não reconheço esse rosto. Quer que eu aprenda quem é?'

    def iniciar_cadastro_confirmado(
        self, sessao: CadastroSessao, frame=None, occhio=None
    ) -> Optional[str]:
        """Inicia cadastro após o usuário confirmar a sugestão."""
        encoding, erro = self.capturar_encoding(frame=frame, occhio=occhio)
        if erro:
            return erro
        if self.face_store and self.face_store.face_exists(encoding):
            existente = None
            if hasattr(self.face_store, 'find_existing_name'):
                existente = self.face_store.find_existing_name(encoding)
            if existente:
                return f'{existente} já está cadastrado.'
            return 'Essa pessoa já parece estar cadastrada.'
        sessao.encoding = encoding
        sessao.estado = 'aguardando_nome'
        sessao.iniciado_em = time.time()
        return 'Qual o nome dessa pessoa?'

    def remover_por_nome(self, nome: str) -> bool:
        """Remove rosto pelo nome (case-insensitive, busca parcial)."""
        if not self.face_store or not hasattr(self.face_store, 'list_faces'):
            return False
        nome_busca = nome.strip().lower().rstrip('?.!')
        if not nome_busca:
            return False
        faces = self.face_store.list_faces()
        removidos = 0
        for face in faces:
            fn = face.get('nome', '').lower()
            if nome_busca in fn or fn in nome_busca:
                if hasattr(self.face_store, 'delete_face_by_id'):
                    if self.face_store.delete_face_by_id(face['id']):
                        removidos += 1
                elif hasattr(self.face_store, 'delete_face'):
                    if self.face_store.delete_face(face.get('nome', '')):
                        removidos += 1
        if removidos:
            self.recarregar_rostos()
            return True
        return False

    def cancelar(self, sessao: CadastroSessao) -> str:
        if sessao.em_andamento():
            sessao.reset()
            return 'Cadastro cancelado.'
        return ''

    def detectar_faces_stream(self, frame) -> List[Dict[str, Any]]:
        if not self.face_detector or not hasattr(self.face_detector, 'detectar_faces_bbox'):
            return []
        rostos = self.face_detector.detectar_faces_bbox(frame)
        for rosto in rostos:
            if not rosto.get('conhecido'):
                continue
            meta = self._meta_rosto(rosto.get('nome', ''))
            if meta:
                rosto['relacao'] = meta.get('relacao')
                rosto['avisar'] = meta.get('avisar')
        return rostos

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
            meta = self._meta_rosto(nome) or {}
            alertas.append({
                'nome': nome,
                'mensagem': f'{nome} está aqui.',
            })
        return alertas

    def get_catalogo(self) -> Dict[str, dict]:
        """Retorna o catálogo completo de rostos cadastrados."""
        if self.face_store and hasattr(self.face_store, 'get_faces_catalog'):
            catalog = self.face_store.get_faces_catalog() or {}
            if catalog:
                self._catalog = catalog
            return catalog
        return dict(self._catalog)


def recarregar_estado_facial(occhio, cadastro_sessao: Optional['CadastroSessao'] = None) -> bool:
    """
    Recarrega completamente o estado de reconhecimento facial a partir do banco.
    Deve ser chamada após qualquer operação de escrita (salvar, deletar, renomear, atualizar).
    """
    try:
        if not occhio or not occhio.face_registry:
            return False

        qtd = occhio.face_registry.recarregar_rostos()
        occhio._last_rostos = []

        if cadastro_sessao and cadastro_sessao.em_andamento():
            cadastro_sessao.reset()

        logger.info(f'✅ Estado facial recarregado: {qtd} rosto(s)')
        return True
    except Exception as e:
        logger.error(f'❌ Erro ao recarregar estado facial: {e}')
        return False
