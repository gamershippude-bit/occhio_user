"""
Interpreter - Versão FINAL: 100% Honesto
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

        # Lista de objetos relevantes
        self.objetos_relevantes = {
            'pessoas': ['person', 'people', 'human'],
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'dining table', 'desk'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'mobile phone', 'monitor', 'keyboard', 'mouse'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'umbrella', 'plate', 'bowl'],
            'animais': ['dog', 'cat', 'bird'],
            'veículos': ['car', 'bicycle', 'motorcycle'],
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe']
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
        """Filtra objetos - APENAS os que sabemos identificar"""
        if not objetos_detectados:
            return []
        
        objetos_filtrados = []
        
        for obj in objetos_detectados:
            obj_lower = obj.lower().strip()
            
            for categoria, objetos in self.objetos_relevantes.items():
                if obj_lower in objetos:
                    objetos_filtrados.append(obj)
                    break
        
        return objetos_filtrados

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
        """Gera respostas 100% honestas"""
        logger.info(f"🔍 Interpreter recebendo - Pergunta: '{pergunta}'")
        logger.info(f"🔍 Faces: {faces_nomes}")
        logger.info(f"🔍 Objetos detectados: {objetos_detectados}")
        
        # Filtrar objetos
        objetos_filtrados = self._filtrar_objetos_relevantes(objetos_detectados)
        
        # Verificar se a pergunta é sobre a imagem
        pergunta_lower = pergunta.lower()
        
        # Lista de palavras que indicam pergunta sobre a imagem
        palavras_imagem = [
            'pessoa', 'pessoas', 'gente', 'humano', 'humanos',
            'cadeira', 'cadeiras', 'sofá', 'sofa', 'mesa', 'mesas', 
            'cama', 'camas', 'livro', 'livros', 'carro', 'carros', 
            'animal', 'animais', 'cachorro', 'gato', 'pássaro',
            'objeto', 'objetos', 'coisa', 'coisas', 'item', 'itens',
            'tem', 'tem alguma', 'tem algum', 'quantos', 'quantas', 
            'o que tem', 'descreva', 'descrever', 'mostre', 'mostrar', 
            'vejo', 'vê', 'enxerga', 'imagem', 'foto', 'fotografia',
            'ambiente', 'cena', 'cenário'
        ]
        
        # Lista de palavras que indicam pergunta geral (não sobre imagem)
        palavras_gerais = [
            'capital', 'horas', 'hora', 'tempo', 'clima', 'céu', 'ceu',
            'nuvem', 'nuvens', 'azul', 'vermelho', 'verde', 'cor', 'cores',
            'país', 'país', 'estado', 'cidade', 'presidente', 'governo',
            'comida', 'bebida', 'água', 'agua', 'música', 'musica', 'filme'
        ]
        
        # Decidir se é sobre imagem ou geral
        eh_sobre_imagem = any(palavra in pergunta_lower for palavra in palavras_imagem)
        eh_pergunta_geral = any(palavra in pergunta_lower for palavra in palavras_gerais)
        
        if eh_sobre_imagem:
            # RESPOSTA CONTROLADA - sem OpenAI
            resposta = self._gerar_resposta_controlada(objetos_filtrados, faces_nomes, pergunta)
            logger.info(f"🎯 Resposta controlada: {resposta}")
            return resposta
        else:
            # Pergunta geral - usar OpenAI
            return self._responder_pergunta_geral(pergunta)

    def _gerar_resposta_controlada(self, objetos_filtrados, faces_nomes, pergunta):
        """Gera resposta CONTROLADA - 100% baseada nos dados reais"""
        pergunta_lower = pergunta.lower()
        
        # Informações detectadas
        total_pessoas = len(faces_nomes)
        objetos_contador = Counter(objetos_filtrados)
        
        logger.info(f"🔍 Dados para resposta controlada:")
        logger.info(f"   - Pessoas: {total_pessoas}")
        logger.info(f"   - Objetos: {dict(objetos_contador)}")
        
        # PERGUNTAS SOBRE PESSOAS
        if any(palavra in pergunta_lower for palavra in ['pessoa', 'pessoas', 'gente', 'quantas pessoas', 'quantas gente']):
            if total_pessoas == 0:
                return "Não detectei pessoas na imagem."
            elif total_pessoas == 1:
                return "Detectei 1 pessoa."
            else:
                return f"Detectei {total_pessoas} pessoas."
        
        # PERGUNTAS SOBRE OBJETOS ESPECÍFICOS
        objetos_especificos = {
            'chair': 'cadeira', 'couch': 'sofá', 'sofa': 'sofá', 
            'bed': 'cama', 'table': 'mesa', 'book': 'livro', 
            'car': 'carro', 'dog': 'cachorro', 'cat': 'gato', 
            'bird': 'pássaro', 'laptop': 'laptop', 'computer': 'computador',
            'tv': 'televisão', 'cell phone': 'celular', 'bottle': 'garrafa', 
            'cup': 'copo', 'backpack': 'mochila'
        }
        
        for obj_ingles, obj_portugues in objetos_especificos.items():
            if obj_portugues in pergunta_lower:
                quantidade = objetos_contador.get(obj_ingles, 0)
                if quantidade == 0:
                    return f"Não detectei {obj_portugues}s na imagem."
                elif quantidade == 1:
                    return f"Detectei 1 {obj_portugues}."
                else:
                    return f"Detectei {quantidade} {obj_portugues}s."
        
        # PERGUNTA GENÉRICA "O QUE TEM" ou "DESCREVA"
        if any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva', 'mostre', 'vejo', 'descrever', 'ambiente', 'cena']):
            partes = []
            
            if total_pessoas > 0:
                if total_pessoas == 1:
                    partes.append("1 pessoa")
                else:
                    partes.append(f"{total_pessoas} pessoas")
            
            if objetos_contador:
                for obj, count in objetos_contador.items():
                    obj_traduzido = self._traduzir_objeto(obj)
                    if count == 1:
                        partes.append(f"1 {obj_traduzido}")
                    else:
                        plural = self._pluralizar_objeto(obj_traduzido, count)
                        partes.append(f"{count} {plural}")
            
            if not partes:
                return "Não detectei pessoas ou objetos específicos na imagem."
            
            # Juntar as partes de forma natural
            if len(partes) == 1:
                return f"Detectei {partes[0]}."
            elif len(partes) == 2:
                return f"Detectei {partes[0]} e {partes[1]}."
            else:
                return "Detectei " + ", ".join(partes[:-1]) + " e " + partes[-1] + "."
        
        # PERGUNTAS SOBRE CORES (não detectamos)
        if any(palavra in pergunta_lower for palavra in ['cor', 'cores', 'azul', 'vermelho', 'verde', 'amarelo']):
            return "Não posso detectar cores, apenas objetos e pessoas."
        
        # PERGUNTAS SOBRE CÉU/CLIMA (não detectamos)
        if any(palavra in pergunta_lower for palavra in ['céu', 'ceu', 'nuvem', 'nuvens', 'clima', 'tempo']):
            return "Não detectei informações sobre o céu ou clima na imagem."
        
        # FALLBACK - resposta direta baseada nos dados
        return self._gerar_resposta_direta(objetos_filtrados, faces_nomes)

    def _responder_pergunta_geral(self, pergunta):
        """Responde perguntas gerais usando OpenAI"""
        if not self.client:
            return "Desculpe, não posso responder perguntas gerais no momento."
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um assistente útil para pessoas com deficiência visual. "
                    "Responda perguntas gerais de forma clara e direta em português. "
                    "Seja conciso e objetivo."
                )
            },
            {
                "role": "user", 
                "content": pergunta
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
            logger.info(f"📥 Resposta geral do OpenAI: {resposta}")
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro OpenAI em pergunta geral: {e}")
            return "Desculpe, não consegui processar sua pergunta no momento."

    def _gerar_resposta_direta(self, objetos_detectados, faces_nomes):
        """Fallback direto - honesto e simples"""
        objetos_filtrados = self._filtrar_objetos_relevantes(objetos_detectados)
        
        partes = []
        
        if faces_nomes:
            total_pessoas = len(faces_nomes)
            if total_pessoas == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{total_pessoas} pessoas")
        
        if objetos_filtrados:
            contador = Counter(objetos_filtrados)
            for obj, count in contador.items():
                obj_traduzido = self._traduzir_objeto(obj)
                if count == 1:
                    partes.append(f"1 {obj_traduzido}")
                else:
                    plural = self._pluralizar_objeto(obj_traduzido, count)
                    partes.append(f"{count} {plural}")
        
        if not partes:
            return "Não detectei pessoas ou objetos específicos na imagem."
        
        if len(partes) == 1:
            return f"Detectei {partes[0]}."
        elif len(partes) == 2:
            return f"Detectei {partes[0]} e {partes[1]}."
        else:
            return "Detectei " + ", ".join(partes[:-1]) + " e " + partes[-1] + "."

    def descrever_ambiente(self, objetos_detectados=None, faces_nomes=None):
        """Descrição do ambiente"""
        return self.responder_pergunta("Descreva o que você vê nesta imagem", objetos_detectados, faces_nomes)