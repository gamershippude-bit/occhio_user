"""
Interpreter - Versão com GPT-4o Mini (Melhor e Mais Barato)
"""

import logging
from openai import OpenAI
import os
from collections import Counter

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(
        self,
        model_name="gpt-4o-mini",  # ← MELHOR CUSTO-BENEFÍCIO
        api_key=None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None

        if not self.api_key:
            logger.warning("Nenhuma API Key fornecida para OpenAI.")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✅ Cliente OpenAI configurado para {self.model_name} (Mais inteligente e econômico)")
            except Exception as e:
                logger.error(f"❌ Falha ao inicializar cliente OpenAI: {e}")
                self.client = None

        logger.info(f"🎯 Interpreter com {model_name} - Superior ao GPT-3.5 com melhor custo")

    def responder_pergunta(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        Gera respostas com modelo superior e mais econômico.
        """
        if not self.client:
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)

        contexto_limpo = self._criar_contexto_otimizado(objetos_detectados, faces_nomes)
        
        # Prompt mais eficiente para economizar tokens
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um assistente visual para cegos. Descreva o ambiente de forma "
                    "natural, direta e sem redundâncias. Apenas o que foi detectado. "
                    "Máximo 2 frases. Exemplo: 'Tem a Maria e 2 pessoas. Vejo um sofá e TV.'"
                )
            },
            {
                "role": "user", 
                "content": f"Pessoas: {self._formatar_pessoas(faces_nomes)}. "
                          f"Objetos: {self._formatar_objetos(objetos_detectados)}. "
                          f"Pergunta: {pergunta}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,  # Reduzido para economizar
                temperature=0.7,
            )
            
            resposta = response.choices[0].message.content.strip()
            logger.info(f"💬 {self.model_name} respondeu: {resposta}")
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro com {self.model_name}: {e}")
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)

    def _formatar_pessoas(self, faces_nomes):
        """Formata pessoas de forma eficiente"""
        if not faces_nomes:
            return "nenhuma pessoa"
        
        conhecidos = [nome for nome in faces_nomes if nome != "Desconhecido"]
        desconhecidos = faces_nomes.count("Desconhecido")
        
        if conhecidos:
            nomes = ", ".join(set(conhecidos))
            if desconhecidos > 0:
                return f"{nomes} e mais {desconhecidos} pessoas"
            return nomes
        return f"{len(faces_nomes)} pessoas"

    def _formatar_objetos(self, objetos_detectados):
        """Formata objetos de forma eficiente"""
        if not objetos_detectados:
            return "nada"
        
        contador = Counter(objetos_detectados)
        itens = []
        for obj, count in contador.items():
            if count == 1:
                itens.append(obj)
            else:
                plural = self._pluralizar_objeto(obj, count)
                itens.append(f"{count} {plural}")
        
        return ", ".join(itens) if itens else "nada"

    def _criar_contexto_otimizado(self, objetos_detectados, faces_nomes):
        """Cria contexto otimizado para economizar tokens"""
        return f"P: {self._formatar_pessoas(faces_nomes)} | O: {self._formatar_objetos(objetos_detectados)}"

    def _gerar_resposta_direta(self, objetos_detectados, faces_nomes):
        """Fallback direto"""
        # ... (mesmo código anterior)
        pass

    def _pluralizar_objeto(self, objeto, quantidade):
        """... (mesmo código anterior)"""
        pass

    def descrever_ambiente(self, objetos_detectados=None, faces_nomes=None):
        """Descrição econômica"""
        if not self.client:
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Descreva o ambiente em 1-2 frases. Seja natural."
                    },
                    {
                        "role": "user", 
                        "content": f"Pessoas: {self._formatar_pessoas(faces_nomes)}. "
                                  f"Objetos: {self._formatar_objetos(objetos_detectados)}. "
                                  f"Descreva:"
                    }
                ],
                max_tokens=100,  # Bem econômico
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Erro descrição: {e}")
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)