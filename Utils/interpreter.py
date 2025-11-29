"""
Interpreter - Versão Corrigida: Só menciona objetos específicos
"""

import logging
import os
from openai import OpenAI
from collections import Counter

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        api_key=None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        # Lista de objetos relevantes (APENAS objetos que sabemos identificar)
        self.objetos_relevantes = {
            'pessoas': ['person', 'people', 'human'],
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'dining table', 'desk'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'mobile phone', 'monitor', 'keyboard', 'mouse'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'umbrella', 'plate', 'bowl'],
            'animais': ['dog', 'cat', 'bird'],
            'veículos': ['car', 'bicycle', 'motorcycle'],
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe']
        }
        
        # Objetos que devem ser IGNORADOS completamente
        self.objetos_irrelevantes = {
            'partes_corpo': ['hand', 'head', 'face', 'arm', 'leg', 'foot', 'finger', 'eye', 'nose', 'mouth'],
            'elementos_cena': ['wall', 'floor', 'ceiling', 'sky', 'ground', 'tree', 'grass'],
            'objetos_vagos': ['stuff', 'things', 'item', 'object', 'unknown', 'furniture', 'vehicle', 'pet']
        }

        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY não encontrada")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✅ Cliente OpenAI configurado para {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Falha ao inicializar cliente OpenAI: {e}")
                self.client = None

    def _filtrar_objetos_relevantes(self, objetos_detectados):
        """Filtra objetos - APENAS os que sabemos identificar especificamente"""
        if not objetos_detectados:
            return []
        
        objetos_filtrados = []
        
        for obj in objetos_detectados:
            obj_lower = obj.lower().strip()
            
            # 1. PULAR objetos irrelevantes
            irrelevante = False
            for categoria, objetos in self.objetos_irrelevantes.items():
                if obj_lower in objetos:
                    irrelevante = True
                    logger.debug(f"❌ Objeto irrelevante filtrado: {obj}")
                    break
            
            if irrelevante:
                continue
                
            # 2. INCLUIR APENAS objetos que sabemos identificar especificamente
            relevante = False
            for categoria, objetos in self.objetos_relevantes.items():
                if obj_lower in objetos:
                    relevante = True
                    objetos_filtrados.append(obj)
                    logger.debug(f"✅ Objeto relevante incluído: {obj}")
                    break
            
            # 3. IGNORAR completamente objetos não classificados
            if not relevante:
                logger.debug(f"🚫 Objeto desconhecido ignorado: {obj}")
        
        return objetos_filtrados

    def _formatar_pessoas(self, faces_nomes):
        """Formata pessoas de forma precisa"""
        if not faces_nomes:
            return "nenhuma pessoa"
        
        # Contar pessoas conhecidas e desconhecidas
        conhecidos = [nome for nome in faces_nomes if nome and nome != "Desconhecido"]
        total_pessoas = len(faces_nomes)
        
        if conhecidos:
            nomes = ", ".join(conhecidos)
            if len(conhecidos) == total_pessoas:
                if total_pessoas == 1:
                    return f"1 pessoa ({nomes})"
                else:
                    return f"{total_pessoas} pessoas ({nomes})"
            else:
                desconhecidos_count = total_pessoas - len(conhecidos)
                return f"{nomes} e {desconhecidos_count} pessoa(s) não identificada(s)"
        else:
            # Apenas pessoas desconhecidas
            if total_pessoas == 1:
                return "1 pessoa"
            else:
                return f"{total_pessoas} pessoas"

    def _formatar_objetos(self, objetos_detectados):
        """Formata APENAS objetos específicos que sabemos identificar"""
        if not objetos_detectados:
            return "nenhum objeto específico"
        
        contador = Counter(objetos_detectados)
        itens = []
        
        for obj, count in contador.items():
            obj_formatado = self._traduzir_objeto(obj)
            if count == 1:
                itens.append(f"1 {obj_formatado}")
            else:
                plural = self._pluralizar_objeto(obj_formatado, count)
                itens.append(f"{count} {plural}")
        
        return ", ".join(itens)

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objetos do inglês para português"""
        traducoes = {
            'person': 'pessoa', 'people': 'pessoas', 'human': 'pessoa',
            'chair': 'cadeira', 'couch': 'sofá', 'sofa': 'sofá', 'bed': 'cama',
            'table': 'mesa', 'dining table': 'mesa de jantar', 'desk': 'escritório',
            'laptop': 'laptop', 'computer': 'computador', 'tv': 'televisão',
            'television': 'televisão', 'cell phone': 'celular', 'mobile phone': 'celular',
            'monitor': 'monitor', 'keyboard': 'teclado', 'mouse': 'mouse',
            'cup': 'copo', 'bottle': 'garrafa', 'book': 'livro', 'vase': 'vaso',
            'clock': 'relógio', 'umbrella': 'guarda-chuva', 'plate': 'prato',
            'bowl': 'tigela', 'dog': 'cachorro', 'cat': 'gato', 'bird': 'pássaro',
            'car': 'carro', 'bicycle': 'bicicleta', 'motorcycle': 'motocicleta',
            'backpack': 'mochila', 'handbag': 'bolsa', 'suitcase': 'mala',
            'tie': 'gravata', 'hat': 'chapéu', 'shoe': 'sapato'
        }
        
        return traducoes.get(objeto_ingles.lower(), objeto_ingles)

    def _pluralizar_objeto(self, objeto, quantidade):
        """Pluraliza objetos em português"""
        especiais = {
            'pessoa': 'pessoas', 'cachorro': 'cachorros', 'gato': 'gatos',
            'pássaro': 'pássaros', 'carro': 'carros', 'mochila': 'mochilas',
            'gravata': 'gravatas', 'chapéu': 'chapéus', 'sapato': 'sapatos'
        }
        
        if objeto in especiais:
            return especiais[objeto]
        
        if objeto.endswith(('r', 'z')):
            return objeto + 'es'
        elif objeto.endswith('l'):
            return objeto[:-1] + 'is'
        elif objeto.endswith('m'):
            return objeto[:-1] + 'ns'
        else:
            return objeto + 's'

    def responder_pergunta(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Gera respostas - APENAS menciona o que foi identificado especificamente"""
        if not self.client:
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)

        objetos_filtrados = self._filtrar_objetos_relevantes(objetos_detectados)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um assistente visual para cegos. Descreva APENAS o que foi identificado COM CLAREZA.\n\n"
                    "REGRAS:\n"
                    "1. Mencione APENAS pessoas e objetos específicos identificados\n"
                    "2. NUNCA mencione 'objetos', 'itens' ou 'coisas' genericamente\n"
                    "3. Se não identificou objetos específicos, mencione APENAS as pessoas\n"
                    "4. Seja PRECISO: '1 pessoa', '2 cadeiras'\n"
                    "5. Máximo 2 frases\n"
                )
            },
            {
                "role": "user", 
                "content": f"Pessoas: {self._formatar_pessoas(faces_nomes)}. "
                          f"Objetos específicos: {self._formatar_objetos(objetos_filtrados) if objetos_filtrados else 'NENHUM'}. "
                          f"Pergunta: {pergunta}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
            )
            
            resposta = response.choices[0].message.content.strip()
            return self._validar_resposta_rigorosa(resposta, objetos_filtrados)
            
        except Exception as e:
            logger.error(f"❌ Erro OpenAI: {e}")
            return self._gerar_resposta_direta(objetos_detectados, faces_nomes)

    def _validar_resposta_rigorosa(self, resposta, objetos_filtrados):
        """Validação RIGOROSA"""
        resposta_lower = resposta.lower()
        
        palavras_proibidas = [
            'objeto', 'objetos', 'item', 'itens', 'coisa', 'coisas', 
            'algo', 'alguma coisa', 'identificado', 'identificados'
        ]
        
        if not objetos_filtrados:
            for palavra in palavras_proibidas:
                if palavra in resposta_lower:
                    if 'pessoa' in resposta_lower or 'pessoas' in resposta_lower:
                        if ' e ' in resposta:
                            partes = resposta.split(' e ')
                            for parte in partes:
                                if 'pessoa' in parte.lower() or 'pessoas' in parte.lower():
                                    resposta = parte.strip()
                                    break
                    else:
                        resposta = "Não identifiquei nada específico no ambiente."
                    break
        
        for palavra in palavras_proibidas:
            if palavra in resposta_lower:
                resposta = resposta.replace(palavra, '')
                resposta = ' '.join(resposta.split())
        
        resposta = resposta.strip()
        if resposta and not resposta.endswith(('.', '!', '?')):
            resposta += '.'
            
        return resposta

    def _gerar_resposta_direta(self, objetos_detectados, faces_nomes):
        """Fallback direto"""
        objetos_filtrados = self._filtrar_objetos_relevantes(objetos_detectados)
        
        partes = []
        
        if faces_nomes:
            conhecidos = [nome for nome in faces_nomes if nome and nome != "Desconhecido"]
            total_pessoas = len(faces_nomes)
            
            if conhecidos:
                nomes = ", ".join(conhecidos)
                if len(conhecidos) == total_pessoas:
                    partes.append(f"Vejo {nomes}" if total_pessoas == 1 else f"Vejo {nomes}")
                else:
                    desconhecidos_count = total_pessoas - len(conhecidos)
                    partes.append(f"Vejo {nomes} e {desconhecidos_count} pessoa(s) não identificada(s)")
            else:
                partes.append("Vejo 1 pessoa" if total_pessoas == 1 else f"Vejo {total_pessoas} pessoas")
        else:
            partes.append("Não vejo ninguém")
        
        if objetos_filtrados:
            contador = Counter(objetos_filtrados)
            objetos_desc = []
            
            for obj, count in contador.items():
                obj_traduzido = self._traduzir_objeto(obj)
                if count == 1:
                    objetos_desc.append(f"1 {obj_traduzido}")
                else:
                    plural = self._pluralizar_objeto(obj_traduzido, count)
                    objetos_desc.append(f"{count} {plural}")
            
            if objetos_desc:
                if partes and "não vejo ninguém" not in partes[0].lower():
                    partes.append("e " + ", ".join(objetos_desc))
                else:
                    partes.append("Vejo " + ", ".join(objetos_desc))
        
        if not partes:
            return "Não identifiquei nada específico no ambiente."
        
        resposta = " ".join(partes) + "."
        return self._validar_resposta_rigorosa(resposta, objetos_filtrados)

    def descrever_ambiente(self, objetos_detectados=None, faces_nomes=None):
        """Descrição do ambiente"""
        return self.responder_pergunta("Descreva o ambiente", objetos_detectados, faces_nomes)