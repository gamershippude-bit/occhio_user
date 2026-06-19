from collections import deque
from typing import Optional
import time


class ConversationMemory:
    """
    Janela deslizante das últimas N trocas da sessão WebSocket.
    Uma instância por conexão WebSocket — criada dentro de _StreamState.
    """
    MAX_TURNS = 6  # ~30 segundos de conversa em ritmo normal

    def __init__(self):
        self._turns = deque(maxlen=self.MAX_TURNS)

    def adicionar_turno(
        self,
        pergunta: str,
        resposta: str,
        deteccoes: list,
        rostos: list,
    ) -> None:
        self._turns.append({
            "pergunta": pergunta,
            "resposta": resposta,
            "objetos": [d.get("nome", "") for d in deteccoes],
            "rostos": [r.get("nome", "") for r in rostos if r.get("conhecido")],
            "ts": time.time(),
        })

    def contexto_para_glm(self) -> list:
        """
        Retorna turns anteriores no formato messages do GLM.
        Injete isso ANTES do turn atual ao chamar glm_chat().
        Só inclui turnos dos últimos 45 segundos.
        """
        agora = time.time()
        msgs = []
        for t in self._turns:
            if agora - t["ts"] > 45:
                continue
            msgs.append({"role": "user", "content": t["pergunta"]})
            msgs.append({"role": "assistant", "content": t["resposta"]})
        return msgs

    def ultimo_assunto(self) -> Optional[str]:
        """Último objeto/rosto mencionado na resposta mais recente."""
        if not self._turns:
            return None
        return self._turns[-1]["resposta"]

    def objetos_recentes(self) -> list:
        """Objetos detectados no último turno — para resolver 'aquele objeto'."""
        if not self._turns:
            return []
        return self._turns[-1]["objetos"]

    def limpar(self) -> None:
        self._turns.clear()
