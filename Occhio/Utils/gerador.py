"""
Módulo de Geração de Descrições
Este módulo é responsável por gerar descrições textuais baseadas nas detecções
de objetos e faces, utilizando técnicas de processamento de linguagem natural.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class GeradorDescricao:
    def __init__(self):
        """
        Inicializa o gerador de descrições.
        """
        logger.info("GeradorDescricao inicializado")

    def gerar_descricao(self, detecao):
        """
        Gera uma descrição textual baseada nas detecções.
        
        Args:
            detecao (dict): Dicionário contendo informações de detecção
                - objetos: Contagem de objetos detectados
                - faces: Contagem de faces detectadas
                - tempo: Timestamp da detecção
                
        Returns:
            str: Descrição textual das detecções
        """
        try:
            # Converte detecção para dicionário se for apenas contagens
            if isinstance(detecao, tuple):
                detecao = {
                    "objetos": detecao[0],
                    "faces": detecao[1]
                }
            
            descricao = []
            
            # Adiciona descrição de objetos
            if detecao.get("objetos", 0) > 0:
                descricao.append(f"Detectado(s) {detecao['objetos']} objeto(s)")
            
            # Adiciona descrição de faces
            if detecao.get("faces", 0) > 0:
                descricao.append(f"Detectada(s) {detecao['faces']} face(s)")
            
            # Retorna descrição formatada
            if descricao:
                return " | ".join(descricao)
            return "Nenhum objeto ou face detectado"
            
        except Exception as e:
            logger.error(f"Erro ao gerar descrição: {e}")
            return "Erro ao gerar descrição" 