"""
Interpreter - VERSÃO FINAL SPECULA com respostas precisas e naturais
"""

import logging
import os
import time
import math
import re
from datetime import datetime
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

        # Dicionário de objetos que sabemos identificar
        self.objetos_conhecidos = {
            'pessoa': ['person', 'people', 'human'],
            'cadeira': ['chair'],
            'sofá': ['couch', 'sofa'],
            'mesa': ['table', 'dining table', 'desk'],
            'cama': ['bed'],
            'computador': ['laptop', 'computer', 'monitor'],
            'televisão': ['tv', 'television'],
            'celular': ['cell phone', 'mobile phone'],
            'livro': ['book'],
            'garrafa': ['bottle'],
            'copo': ['cup'],
            'prato': ['plate'],
            'vaso': ['vase'],
            'relógio': ['clock'],
            'cachorro': ['dog'],
            'gato': ['cat'],
            'carro': ['car'],
            'bicicleta': ['bicycle'],
            'mochila': ['backpack'],
            'bolsa': ['handbag'],
            'garfo': ['fork'],
            'faca': ['knife'],
            'colher': ['spoon'],
            'vaso sanitário': ['toilet'],
            'pia': ['sink'],
            'porta': ['door'],
            'janela': ['window'],
            'planta': ['potted plant'],
            'flor': ['flower'],
            'refrigerante': ['soda can', 'can'],
            'xicara': ['cup'],
            'mouse': ['mouse'],
            'teclado': ['keyboard'],
            'cadeira de escritório': ['office chair'],
            'abajur': ['lamp'],
            'quadro': ['picture', 'painting'],
            'bola': ['sports ball', 'ball'],
            'tenis': ['sneakers', 'shoe'],
            'chapéu': ['hat'],
            'guarda-chuva': ['umbrella'],
            'maleta': ['suitcase'],
            'frisbee': ['frisbee'],
            'neve': ['snowboard'],
            'garrafa esportiva': ['sports bottle']
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

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /processar - Gera descrição natural baseada APENAS nos dados reais
        """
        logger.info("🌄 Gerando descrição natural do ambiente")
        
        if not self.client:
            return "Vou descrever o ambiente para você. Sou a Specula, sua assistente visual."
        
        # Preparar dados REAIS
        objetos_filtrados = self._filtrar_objetos_relevantes(
            [obj.get('name', '') for obj in (objetos_detectados or [])]
        )
        
        contador_objetos = Counter(objetos_filtrados)
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Se temos poucos dados, usar resposta simples baseada APENAS no que tem
        if not objetos_filtrados and total_pessoas == 0:
            return "Olha, parece um ambiente bem simples ou vazio. Não estou identificando muitos objetos ou pessoas. Sou a Specula, sua assistente!"
        
        # Construir contexto APENAS com o que foi realmente detectado
        contexto_real = self._construir_contexto_real(contador_objetos, total_pessoas, faces_conhecidas)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""Você é a Specula, uma assistente que descreve ambientes para pessoas com deficiência visual.

                    INFORMAÇÕES REAIS DETECTADAS (APENAS ISSO FOI DETECTADO):
                    {contexto_real}

                    REGRAS ABSOLUTAS:
                    1. Descreva APENAS o que está listado acima
                    2. Não invente nada que não esteja na lista
                    3. Se algo não está na lista, não mencione
                    4. Seja natural, mas preciso
                    5. Use linguagem cotidiana como "tem", "vejo", "aqui tem"
                    6. Você é a Specula - seja acolhedora e útil

                    EXEMPLOS CORRETOS:
                    Se a lista diz "2 pessoas" → "Tem duas pessoas"
                    Se a lista diz "1 cadeira" → "Tem uma cadeira"
                    Se a lista diz "1 bola" → "Tem uma bola"
                    Se a lista está vazia → "Não tem muita coisa visível"

                    NUNCA INVENTE objetos que não estão na lista!"""
                },
                {
                    "role": "user", 
                    "content": "Descreva este ambiente naturalmente, baseado apenas nas informações acima:"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=180,
                temperature=0.5,
            )
            
            descricao = response.choices[0].message.content.strip()
            
            # Verificar se a IA não inventou nada
            descricao = self._verificar_e_corrigir_invencoes(descricao, contador_objetos, total_pessoas, faces_conhecidas)
            
            logger.info(f"✅ Descrição natural gerada (baseada em dados reais)")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição natural: {e}")
            return self._gerar_descricao_precisa(contador_objetos, total_pessoas, faces_conhecidas)

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat preciso baseado apenas nos dados detectados
        """
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Primeiro verificar se é pergunta sobre tempo/data
        resposta_tempo = self._verificar_pergunta_tempo(pergunta)
        if resposta_tempo:
            return {
                'sucesso': True,
                'timestamp': time.time(),
                'tempo_processamento': f"{time.time() - start_time:.2f}s",
                'pergunta': pergunta,
                'resposta': resposta_tempo,
                'tipo_pergunta': "geral",
                'correlacao_com_imagem': False,
                'dados_utilizados': "Pergunta geral (tempo/data)"
            }
        
        # Classificar o tipo de pergunta
        tipo_pergunta = self._classificar_tipo_pergunta(pergunta)
        logger.info(f"🔍 Pergunta classificada como: {tipo_pergunta}")
        
        if tipo_pergunta == "sobre_imagem":
            # Pergunta sobre a imagem - usar dados detectados
            resposta = self._responder_sobre_imagem_precisa(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
        else:
            # Pergunta geral - responder de forma natural
            resposta = self._responder_pergunta_geral_natural(pergunta)
            correlacao = False
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'pergunta': pergunta,
            'resposta': resposta,
            'tipo_pergunta': tipo_pergunta,
            'correlacao_com_imagem': correlacao,
            'dados_utilizados': self._formatar_dados_utilizados(objetos_detectados, faces_nomes) if correlacao else "Pergunta geral"
        }

    # ========== MÉTODO PARA VERIFICAR PERGUNTAS DE TEMPO ==========

    def _verificar_pergunta_tempo(self, pergunta):
        """Verifica se é pergunta sobre tempo/data e responde"""
        pergunta_lower = pergunta.lower()
        
        # Perguntas sobre horas
        if any(palavra in pergunta_lower for palavra in ['que horas', 'que hora', 'horas são', 'hora é', 'que horas são']):
            agora = datetime.now()
            hora_str = agora.strftime("%H:%M")
            return f"São {hora_str}."
        
        # Perguntas sobre data
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data', 'data de hoje', 'que dia estamos']):
            agora = datetime.now()
            data_str = agora.strftime("%d/%m/%Y")
            dia_semana = agora.strftime("%A")
            
            # Traduzir dia da semana
            dias_traduzidos = {
                "Monday": "segunda-feira",
                "Tuesday": "terça-feira",
                "Wednesday": "quarta-feira",
                "Thursday": "quinta-feira",
                "Friday": "sexta-feira",
                "Saturday": "sábado",
                "Sunday": "domingo"
            }
            dia_pt = dias_traduzidos.get(dia_semana, dia_semana)
            
            return f"Hoje é {dia_pt}, {data_str}."
        
        # Perguntas sobre dia da semana
        elif any(palavra in pergunta_lower for palavra in ['que dia é', 'dia da semana', 'que dia hoje', 'qual é o dia']):
            agora = datetime.now()
            dia_semana = agora.strftime("%A")
            
            dias_traduzidos = {
                "Monday": "segunda-feira",
                "Tuesday": "terça-feira",
                "Wednesday": "quarta-feira",
                "Thursday": "quinta-feira",
                "Friday": "sexta-feira",
                "Saturday": "sábado",
                "Sunday": "domingo"
            }
            dia_pt = dias_traduzidos.get(dia_semana, dia_semana)
            
            return f"Hoje é {dia_pt}."
        
        # Perguntas sobre ano
        elif any(palavra in pergunta_lower for palavra in ['que ano é', 'em que ano estamos', 'qual o ano']):
            agora = datetime.now()
            ano = agora.strftime("%Y")
            return f"Estamos em {ano}."
        
        return None

    # ========== MÉTODO PRECISO PARA RESPONDER SOBRE IMAGEM ==========

    def _responder_sobre_imagem_precisa(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde de forma PRECISA, baseada apenas nos dados reais"""
        # Preparar dados REAIS
        objetos_filtrados = []
        if objetos_detectados:
            for obj in objetos_detectados:
                nome = obj.get('name', '')
                count = obj.get('count', 1)
                for _ in range(count):
                    objetos_filtrados.append(nome)
        
        contador_objetos = Counter(objetos_filtrados)
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # DEBUG: Mostrar o que foi realmente detectado
        logger.info(f"📊 Dados reais detectados: {dict(contador_objetos)}, Pessoas: {total_pessoas}, Faces conhecidas: {faces_conhecidas}")
        
        # Se temos OpenAI, usar para resposta precisa
        if self.client:
            return self._responder_com_ia_precisa(pergunta, contador_objetos, total_pessoas, faces_conhecidas)
        else:
            return self._responder_base_precisa(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _responder_com_ia_precisa(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta precisa usando IA - baseada APENAS nos dados reais"""
        try:
            # Construir lista PRECISA do que foi detectado
            dados_reais = self._construir_dados_reais_precisos(contador_objetos, total_pessoas, faces_conhecidas)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""Você é a Specula, uma assistente que responde perguntas sobre imagens com base APENAS nos dados reais detectados.

                    DADOS REAIS DETECTADOS (APENAS ISTO FOI DETECTADO - NÃO INVENTE NADA):
                    {dados_reais}

                    REGRAS ABSOLUTAS:
                    1. Responda APENAS com base nos dados acima
                    2. Se algo não está na lista, NÃO EXISTE na imagem
                    3. Não invente, não deduza, não suponha
                    4. Seja natural mas preciso
                    5. Para perguntas sobre algo não detectado: "Não tem" ou "Não estou vendo"
                    6. Lembre: você é a Specula - seja útil e acolhedora

                    EXEMPLOS:
                    Pergunta: "Tem cadeira?" → Se cadeira está na lista: "Sim, tem uma cadeira" / Se não está: "Não, não tem cadeira"
                    Pergunta: "Quantas pessoas?" → Se pessoas estão na lista: "Tem X pessoas" / Se não está: "Não tem pessoas"
                    Pergunta: "Tem plantas?" → Se plantas não estão na lista: "Não, não tem plantas"

                    IMPORTANTE: Você só sabe o que está na lista acima. Se não está lá, não está na imagem!"""
                },
                {
                    "role": "user", 
                    "content": pergunta
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.3,  # Mais baixo para ser mais preciso
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # Verificar se a resposta é precisa
            resposta = self._verificar_precisao_resposta(resposta, contador_objetos, total_pessoas, faces_conhecidas)
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._responder_base_precisa(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _construir_dados_reais_precisos(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói lista precisa do que foi realmente detectado"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                partes.append(f"Pessoas identificadas: {', '.join(faces_conhecidas)}")
            else:
                partes.append(f"Pessoas: {total_detectado}")
        
        # Objetos detectados (traduzidos)
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_traduzidos = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                objetos_traduzidos.append(f"{quantidade} {obj_pt}")
            
            partes.append(f"Objetos detectados: {', '.join(objetos_traduzidos)}")
        
        if not partes:
            return "Nenhum objeto ou pessoa detectado."
        
        return " | ".join(partes)

    def _responder_base_precisa(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta base precisa sem IA - baseada apenas nos dados"""
        pergunta_lower = pergunta.lower()
        
        # Dados reais
        pessoas_yolo = contador_objetos.get('person', 0)
        total_pessoas_reais = max(total_pessoas, pessoas_yolo)
        objetos_reais = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        # Lista de objetos detectados (em português)
        objetos_pt = {self._traduzir_objeto(obj): qtd for obj, qtd in objetos_reais.items()}
        
        # Perguntas sobre pessoas
        if any(palavra in pergunta_lower for palavra in ['pessoa', 'pessoas', 'gente', 'alguém', 'quem']):
            if total_pessoas_reais > 0:
                if faces_conhecidas:
                    if len(faces_conhecidas) == 1:
                        return f"Sim, tem o {faces_conhecidas[0]}."
                    else:
                        return f"Sim, tem o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}."
                else:
                    if total_pessoas_reais == 1:
                        return "Sim, tem uma pessoa."
                    else:
                        return f"Sim, tem {total_pessoas_reais} pessoas."
            else:
                return "Não, não tem ninguém."
        
        # Perguntas sobre objetos específicos
        for objeto_pergunta in ['cadeira', 'sofá', 'mesa', 'cama', 'computador', 'tv', 'televisão', 
                               'celular', 'livro', 'garrafa', 'copo', 'prato', 'vaso', 'relógio',
                               'cachorro', 'gato', 'carro', 'bicicleta', 'mochila', 'bolsa',
                               'planta', 'flor', 'refrigerante', 'xicara', 'mouse', 'teclado',
                               'abajur', 'quadro', 'bola', 'tenis', 'chapéu', 'guarda-chuva',
                               'maleta', 'frisbee', 'neve', 'garrafa esportiva']:
            
            if objeto_pergunta in pergunta_lower:
                # Verificar se o objeto foi detectado
                objeto_detectado = False
                quantidade = 0
                
                for obj_pt, qtd in objetos_pt.items():
                    if objeto_pergunta in obj_pt.lower() or obj_pt.lower() in objeto_pergunta:
                        objeto_detectado = True
                        quantidade = qtd
                        break
                
                if objeto_detectado:
                    if quantidade == 1:
                        artigo = "uma" if objeto_pergunta in ['cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta'] else "um"
                        return f"Sim, tem {artigo} {objeto_pergunta}."
                    else:
                        return f"Sim, tem {quantidade} {objeto_pergunta}s."
                else:
                    return f"Não, não tem {objeto_pergunta}."
        
        # Perguntas sobre categorias de objetos
        if 'eletrônic' in pergunta_lower or 'eletronic' in pergunta_lower:
            eletronicos = [obj for obj in objetos_pt.keys() if obj in ['computador', 'tv', 'televisão', 'celular', 'monitor', 'mouse', 'teclado']]
            if eletronicos:
                return f"Sim, tem {', '.join(eletronicos)}."
            else:
                return "Não, não tem eletrônicos."
        
        if 'plant' in pergunta_lower or 'natureza' in pergunta_lower or 'verde' in pergunta_lower:
            plantas = [obj for obj in objetos_pt.keys() if obj in ['planta', 'flor']]
            if plantas:
                return f"Sim, tem {', '.join(plantas)}."
            else:
                return "Não, não tem plantas."
        
        if 'móve' in pergunta_lower or 'move' in pergunta_lower or 'mobília' in pergunta_lower:
            moveis = [obj for obj in objetos_pt.keys() if obj in ['cadeira', 'sofá', 'mesa', 'cama']]
            if moveis:
                return f"Sim, tem {', '.join(moveis)}."
            else:
                return "Não, não tem móveis."
        
        if 'animal' in pergunta_lower or 'cachorro' in pergunta_lower or 'gato' in pergunta_lower or 'pet' in pergunta_lower:
            animais = [obj for obj in objetos_pt.keys() if obj in ['cachorro', 'gato']]
            if animais:
                return f"Sim, tem {', '.join(animais)}."
            else:
                return "Não, não tem animais."
        
        if 'esport' in pergunta_lower or 'bola' in pergunta_lower:
            esportivos = [obj for obj in objetos_pt.keys() if obj in ['bola', 'tenis', 'frisbee', 'neve']]
            if esportivos:
                return f"Sim, tem {', '.join(esportivos)}."
            else:
                return "Não, não tem itens esportivos."
        
        # Pergunta: "O que tem?" ou "Descreva"
        if 'o que tem' in pergunta_lower or 'descreva' in pergunta_lower or 'o que você vê' in pergunta_lower:
            partes = []
            
            if total_pessoas_reais > 0:
                if faces_conhecidas:
                    if len(faces_conhecidas) == 1:
                        partes.append(f"o {faces_conhecidas[0]}")
                    else:
                        partes.append(f"o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
                else:
                    if total_pessoas_reais == 1:
                        partes.append("uma pessoa")
                    else:
                        partes.append(f"{total_pessoas_reais} pessoas")
            
            if objetos_pt:
                for obj_pt, qtd in objetos_pt.items():
                    if qtd == 1:
                        artigo = "uma" if obj_pt in ['cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta'] else "um"
                        partes.append(f"{artigo} {obj_pt}")
                    else:
                        partes.append(f"{qtd} {obj_pt}s")
            
            if partes:
                if len(partes) == 1:
                    return f"Tem {partes[0]}."
                else:
                    return f"Tem {', '.join(partes[:-1])} e {partes[-1]}."
            else:
                return "Não tem muita coisa visível."
        
        # Pergunta: "Quais objetos?"
        if 'quais objetos' in pergunta_lower or 'identifica' in pergunta_lower:
            if objetos_pt:
                lista = []
                for obj_pt, qtd in objetos_pt.items():
                    if qtd == 1:
                        artigo = "uma" if obj_pt in ['cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta'] else "um"
                        lista.append(f"{artigo} {obj_pt}")
                    else:
                        lista.append(f"{qtd} {obj_pt}s")
                
                return f"Tem {', '.join(lista)}."
            else:
                return "Não estou vendo objetos específicos."
        
        # Pergunta: "Interno ou externo?"
        if 'interno' in pergunta_lower or 'externo' in pergunta_lower or 'dentro' in pergunta_lower or 'fora' in pergunta_lower:
            # Inferir apenas com base nos objetos detectados
            objetos_detectados_lista = list(objetos_pt.keys())
            
            if any(obj in objetos_detectados_lista for obj in ['cadeira', 'mesa', 'cama', 'sofá', 'tv', 'computador']):
                return "Parece um ambiente interno."
            elif any(obj in objetos_detectados_lista for obj in ['carro', 'bicicleta', 'árvore']):
                return "Parece um ambiente externo."
            else:
                return "É difícil dizer só pelos objetos detectados."
        
        # Resposta padrão baseada nos dados
        if total_pessoas_reais > 0:
            return f"Tem {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}."
        elif objetos_pt:
            primeiro_obj = list(objetos_pt.items())[0]
            obj_nome = primeiro_obj[0]
            qtd = primeiro_obj[1]
            
            if qtd == 1:
                artigo = "uma" if obj_nome in ['cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta'] else "um"
                return f"Tem {artigo} {obj_nome}."
            else:
                return f"Tem {qtd} {obj_nome}s."
        else:
            return "Não estou identificando muitos detalhes no ambiente."

    # ========== MÉTODO PARA PERGUNTAS GERAIS NATURAIS ==========

    def _responder_pergunta_geral_natural(self, pergunta):
        """Responde perguntas gerais de forma natural e útil"""
        if not self.client:
            return "Olá! Sou a Specula, sua assistente. Podemos conversar sobre o ambiente que você está querendo entender?"
        
        # Primeiro verificar se é uma pergunta sobre o próprio assistente
        pergunta_lower = pergunta.lower()
        
        if any(palavra in pergunta_lower for palavra in ['quem é você', 'o que você é', 'qual seu nome', 'seu nome', 'como você se chama']):
            return "Eu sou a Specula, uma assistente para ajudar pessoas com deficiência visual a entender melhor seu ambiente através de imagens. Como posso te ajudar?"
        
        if any(palavra in pergunta_lower for palavra in ['obrigado', 'valeu', 'agradeço', 'thanks', 'obrigada']):
            return "Por nada! Estou aqui para ajudar. Sou a Specula, à sua disposição. Tem mais alguma coisa sobre o ambiente que você gostaria de saber?"
        
        if any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'hello', 'hi']):
            cumprimentos = {
                'manhã': "Bom dia!",
                'tarde': "Boa tarde!",
                'noite': "Boa noite!"
            }
            
            agora = datetime.now().hour
            if 5 <= agora < 12:
                periodo = 'manhã'
            elif 12 <= agora < 18:
                periodo = 'tarde'
            else:
                periodo = 'noite'
            
            return f"{cumprimentos[periodo]} Eu sou a Specula, sua assistente visual. Como posso te ajudar hoje?"
        
        if any(palavra in pergunta_lower for palavra in ['como você está', 'tudo bem', 'como vai', 'tudo bom']):
            return "Estou bem, obrigada! Pronta para te ajudar a entender melhor o ambiente ao seu redor. Sou a Specula. E você, como está?"
        
        if any(palavra in pergunta_lower for palavra in ['qual sua função', 'o que você faz', 'para que serve']):
            return "Sou a Specula, uma assistente visual. Minha função é ajudar pessoas com deficiência visual a entender melhor o ambiente ao seu redor através da análise de imagens. Posso descrever o que há numa foto, identificar objetos e pessoas, e responder perguntas sobre o ambiente!"
        
        # Perguntas sobre clima (resposta simples)
        if any(palavra in pergunta_lower for palavra in ['como está o tempo', 'qual a temperatura', 'está frio', 'está quente']):
            return "Infelizmente não tenho acesso a informações do tempo em tempo real no momento. Mas posso te ajudar a entender o ambiente nas imagens que você enviar!"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Você é a Specula, uma assistente amigável e útil para pessoas com deficiência visual.
                    
                    SEU ESTILO:
                    - Fale de forma natural, como em uma conversa
                    - Seja acolhedora e prestativa
                    - Use linguagem simples e clara
                    - Se não souber algo, admita de forma natural
                    - Mantenha as respostas razoavelmente curtas
                    - Lembre-se: você é a Specula!
                    
                    VOCÊ PODE:
                    - Responder perguntas sobre o mundo
                    - Dar explicações simples
                    - Conversar de forma geral
                    - Ajudar com perguntas cotidianas
                    
                    NÃO SE ESQUEÇA: Se o usuário quiser voltar a falar sobre a imagem, você pode ajudar com isso também."""
                },
                {
                    "role": "user", 
                    "content": pergunta
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.6,  # Natural mas não muito criativo
            )
            
            resposta = response.choices[0].message.content.strip()
            
            return resposta
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return "Desculpe, tive um problema técnico. Sou a Specula, vamos focar na imagem que você enviou?"

    # ========== MÉTODOS DE VERIFICAÇÃO DE PRECISÃO ==========

    def _verificar_e_corrigir_invencoes(self, descricao, contador_objetos, total_pessoas, faces_conhecidas):
        """Verifica se a IA inventou algo e corrige"""
        # Obter lista real de objetos detectados (em português)
        objetos_reais_pt = set()
        for obj_ingles, qtd in contador_objetos.items():
            if obj_ingles != 'person':  # Pessoas tratadas separadamente
                obj_pt = self._traduzir_objeto(obj_ingles)
                objetos_reais_pt.add(obj_pt.lower())
        
        # Verificar pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Lista de objetos mencionados na descrição
        descricao_lower = descricao.lower()
        
        # Verificar se mencionou pessoas quando não tem
        if total_detectado == 0 and any(palavra in descricao_lower for palavra in ['pessoa', 'pessoas', 'gente', 'alguém', 'homem', 'mulher']):
            # Corrigir: remover menção a pessoas
            descricao = re.sub(r'\b(?:tem|vejo|acho que tem|tem algumas?)\s+(?:umas?\s+)?\d*\s*(?:pessoas?|gente|alguém)\b', 
                             'não tem pessoas', descricao, flags=re.IGNORECASE)
        
        # Se a descrição parece muito vazia mas temos dados, adicionar contexto
        palavras_vazias = ['nada', 'vazio', 'não tem nada', 'sem nada', 'nenhum']
        if any(palavra in descricao_lower for palavra in palavras_vazias) and (total_detectado > 0 or objetos_reais_pt):
            # Substituir por descrição precisa
            return self._gerar_descricao_precisa(contador_objetos, total_pessoas, faces_conhecidas)
        
        return descricao

    def _verificar_precisao_resposta(self, resposta, contador_objetos, total_pessoas, faces_conhecidas):
        """Verifica a precisão da resposta e corrige se necessário"""
        resposta_lower = resposta.lower()
        
        # Dados reais
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        objetos_reais = {k: v for k, v in contador_objetos.items() if k != 'person'}
        objetos_pt = {self._traduzir_objeto(obj): qtd for obj, qtd in objetos_reais.items()}
        
        # Verificar afirmações incorretas sobre pessoas
        if total_detectado == 0:
            # Não deve afirmar que tem pessoas
            if any(palavra in resposta_lower for palavra in ['tem pessoa', 'tem pessoas', 'tem gente', 'tem alguém']):
                # Corrigir
                if 'não' not in resposta_lower[:50]:  # Se não está negando
                    return "Não, não tem pessoas."
        
        # Se a resposta parece inventar algo, usar resposta base
        palavras_suspeitas = ['provavelmente', 'deve ter', 'acho que', 'talvez', 'pode ser', 'imagino', 'suponho']
        if any(palavra in resposta_lower for palavra in palavras_suspeitas):
            # A IA está especulando - usar resposta precisa
            return self._responder_base_precisa("O que tem?", contador_objetos, total_pessoas, faces_conhecidas)
        
        return resposta

    # ========== MÉTODOS AUXILIARES ==========

    def _construir_contexto_real(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto apenas com dados reais"""
        partes = []
        
        # Pessoas reais
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                partes.append(f"Pessoas: {', '.join(faces_conhecidas)}")
            else:
                partes.append(f"Pessoas: {total_detectado}")
        
        # Objetos reais detectados
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_lista = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    objetos_lista.append(f"1 {obj_pt}")
                else:
                    objetos_lista.append(f"{quantidade} {obj_pt}s")
            
            partes.append(f"Objetos: {', '.join(objetos_lista)}")
        
        if not partes:
            return "Nada detectado."
        
        return " | ".join(partes)

    def _gerar_descricao_precisa(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição precisa baseada apenas nos dados"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"o {faces_conhecidas[0]}")
                else:
                    partes.append(f"o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
            else:
                if total_detectado == 1:
                    partes.append("uma pessoa")
                else:
                    partes.append(f"{total_detectado} pessoas")
        
        # Objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    artigo = "uma" if obj_pt in ['cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta'] else "um"
                    partes.append(f"{artigo} {obj_pt}")
                else:
                    partes.append(f"{quantidade} {obj_pt}s")
        
        if not partes:
            return "Olha, parece um ambiente bem simples. Não estou identificando objetos ou pessoas específicas. Sou a Specula, sua assistente!"
        
        if len(partes) == 1:
            return f"Tem {partes[0]}."
        else:
            return f"Tem {', '.join(partes[:-1])} e {partes[-1]}."

    def _classificar_tipo_pergunta(self, pergunta):
        """Classifica se a pergunta é sobre a imagem ou geral"""
        pergunta_lower = pergunta.lower().strip()
        
        # Palavras que indicam pergunta SOBRE A IMAGEM
        palavras_imagem = [
            'essa imagem', 'esta foto', 'na foto', 'na imagem',
            'o que tem', 'quem está', 'onde está', 'tem ', 'há ',
            'quantos', 'quantas', 'pessoa', 'pessoas',
            'descreva', 'analise', 'identifique', 'reconhece',
            'mostra', 'mostrar', 'o que você vê', 'que tem aí',
            'ambiente', 'lugar', 'sala', 'quarto', 'cozinha',
            'vejo', 'vê', 'consegue ver', 'está vendo',
            'identifica', 'reconhece', 'há alguma', 'tem algum',
            'objeto', 'objetos', 'coisa', 'coisas',
            'quais', 'qual'
        ]
        
        # Perguntas que são CLARAMENTE gerais (não sobre imagem)
        perguntas_gerais_claras = [
            'o que é', 'como funciona', 'quem foi',
            'história de', 'significado de', 'definição de',
            'explique', 'explicar', 'conceito de',
            'capital de', 'população de', 'onde fica',
            'conta uma piada', 'qual o sentido da vida',
            'que horas', 'que dia', 'data', 'horas são',
            'como você está', 'quem é você', 'qual seu nome',
            'obrigado', 'valeu', 'agradeço',
            'oi', 'olá', 'bom dia', 'boa tarde', 'boa noite',
            'tudo bem', 'como vai', 'hello', 'hi',
            'qual sua função', 'o que você faz',
            'como está o tempo', 'qual a temperatura'
        ]
        
        # Primeiro verificar se é claramente geral
        for palavra in perguntas_gerais_claras:
            if palavra in pergunta_lower:
                return "geral"
        
        # Verificar se é sobre a imagem
        for palavra in palavras_imagem:
            if palavra in pergunta_lower:
                return "sobre_imagem"
        
        # Se é uma pergunta curta com "?", provavelmente é sobre a imagem
        if '?' in pergunta and len(pergunta.split()) < 10:
            return "sobre_imagem"
        
        # Por padrão, assumir que é geral
        return "geral"

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objeto do inglês para português"""
        for pt, en_list in self.objetos_conhecidos.items():
            if objeto_ingles.lower() in en_list:
                return pt
        return objeto_ingles

    def _filtrar_objetos_relevantes(self, objetos_detectados):
        """Filtra apenas objetos que conhecemos"""
        objetos_filtrados = []
        todos_objetos = [obj for lista in self.objetos_conhecidos.values() for obj in lista]
        
        for obj in objetos_detectados:
            if obj.lower() in todos_objetos:
                objetos_filtrados.append(obj)
        
        return objetos_filtrados

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados para resposta"""
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        
        return f"{objetos_count} objetos e {faces_count} pessoas analisadas"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """
        PARA ROTA /estatistica - Dados técnicos detalhados
        """
        logger.info("📊 Gerando estatísticas técnicas")
        
        start_time = time.time()
        
        # Processar objetos detectados
        objetos_processados = self._processar_objetos_estatisticas(objetos_detectados)
        faces_processadas = self._processar_faces_estatisticas(faces_detectadas or [])
        
        # Calcular métricas de precisão
        metricas_precisao = self._calcular_metricas_precisao(objetos_detectados, faces_detectadas)
        
        # Gerar análise técnica
        analise_tecnica = self._gerar_analise_tecnica(objetos_processados, faces_processadas)
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'contagens': {
                'total_objetos': len(objetos_detectados),
                'total_faces': len(faces_detectadas or []),
                'objetos_por_categoria': self._agrupar_objetos_por_categoria(objetos_detectados),
                'faces_conhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') != 'Desconhecido']),
                'faces_desconhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') == 'Desconhecido'])
            },
            'precisao': metricas_precisao,
            'deteccoes_detalhadas': {
                'objetos': objetos_processados,
                'faces': faces_processadas
            },
            'analise_tecnica': analise_tecnica,
            'logs_diagnostico': self._gerar_logs_diagnostico(objetos_detectados, faces_detectadas)
        }

    # ========== MÉTODOS DE ESTATÍSTICAS ==========

    def _processar_objetos_estatisticas(self, objetos_detectados):
        """Processa objetos para estatísticas detalhadas"""
        objetos_processados = []
        
        for i, obj in enumerate(objetos_detectados, 1):
            nome_ingles = obj.get('name', 'desconhecido')
            confianca = obj.get('confidence', 0)
            bbox = obj.get('bbox', {})
            count = obj.get('count', 1)
            
            objeto_info = {
                'id': i,
                'nome_pt': self._traduzir_objeto(nome_ingles),
                'nome_en': nome_ingles,
                'confianca': confianca,
                'confianca_percentual': f"{confianca:.1%}",
                'quantidade': count,
                'categoria': self._classificar_categoria(nome_ingles),
                'coordenadas': bbox,
                'area': bbox.get('width', 0) * bbox.get('height', 0) if bbox else 0,
                'nivel_confianca': self._classificar_nivel_confianca(confianca)
            }
            objetos_processados.append(objeto_info)
        
        return objetos_processados

    def _processar_faces_estatisticas(self, faces_detectadas):
        """Processa faces para estatísticas detalhadas"""
        faces_processadas = []
        
        for i, face in enumerate(faces_detectadas, 1):
            nome = face.get('name', 'Desconhecido')
            confianca = face.get('confidence', 0)
            bbox = face.get('bbox', {})
            
            face_info = {
                'id': i,
                'nome': nome,
                'tipo': 'conhecida' if nome != 'Desconhecido' else 'desconhecida',
                'confianca': confianca,
                'confianca_percentual': f"{confianca:.1%}",
                'coordenadas': bbox,
                'area_rosto': bbox.get('width', 0) * bbox.get('height', 0) if bbox else 0,
                'nivel_confianca': self._classificar_nivel_confianca(confianca)
            }
            faces_processadas.append(face_info)
        
        return faces_processadas

    def _calcular_metricas_precisao(self, objetos_detectados, faces_detectadas):
        """Calcula métricas de precisão detalhadas"""
        confiancas_objetos = [obj.get('confidence', 0) for obj in objetos_detectados]
        confiancas_faces = [face.get('confidence', 0) for face in (faces_detectadas or [])]
        
        def calcular_media(lista):
            return sum(lista) / len(lista) if lista else 0
        
        def calcular_desvio_padrao(lista):
            if not lista or len(lista) < 2:
                return 0
            media = calcular_media(lista)
            variancia = sum((x - media) ** 2 for x in lista) / len(lista)
            return math.sqrt(variancia)
        
        return {
            'confianca_media_objetos': calcular_media(confiancas_objetos),
            'confianca_media_faces': calcular_media(confiancas_faces),
            'confianca_maxima_objetos': max(confiancas_objetos) if confiancas_objetos else 0,
            'confianca_minima_objetos': min(confiancas_objetos) if confiancas_objetos else 0,
            'desvio_padrao_objetos': calcular_desvio_padrao(confiancas_objetos),
            'objetos_alta_confianca': len([c for c in confiancas_objetos if c > 0.7]),
            'objetos_baixa_confianca': len([c for c in confiancas_objetos if c < 0.3])
        }

    def _gerar_analise_tecnica(self, objetos_processados, faces_processadas):
        """Gera análise técnica dos dados"""
        total_objetos = len(objetos_processados)
        total_faces = len(faces_processadas)
        
        if total_objetos == 0 and total_faces == 0:
            return "Nenhum objeto ou face detectado - verifique qualidade da imagem"
        
        analise = []
        
        if total_objetos > 0:
            categorias = Counter([obj['categoria'] for obj in objetos_processados])
            categoria_principal = categorias.most_common(1)[0][0] if categorias else "diversos"
            total_itens = sum(obj.get('quantidade', 1) for obj in objetos_processados)
            analise.append(f"{total_itens} itens detectados ({total_objetos} tipos), predominância: {categoria_principal}")
        
        if total_faces > 0:
            faces_conhecidas = len([f for f in faces_processadas if f['tipo'] == 'conhecida'])
            if faces_conhecidas > 0:
                analise.append(f"{faces_conhecidas} face(s) conhecida(s)")
            if total_faces - faces_conhecidas > 0:
                analise.append(f"{total_faces - faces_conhecidas} face(s) desconhecida(s)")
        
        return ". ".join(analise)

    def _agrupar_objetos_por_categoria(self, objetos_detectados):
        """Agrupa objetos por categoria"""
        categorias = {}
        for obj in objetos_detectados:
            nome = obj.get('name', 'desconhecido')
            count = obj.get('count', 1)
            categoria = self._classificar_categoria(nome)
            if categoria not in categorias:
                categorias[categoria] = 0
            categorias[categoria] += count
        return categorias

    def _classificar_categoria(self, objeto_ingles):
        """Classifica objeto em categoria"""
        categorias = {
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'dining table', 'desk', 'office chair'],
            'pessoas': ['person', 'people', 'human'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'mobile phone', 'monitor', 'mouse', 'keyboard'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'plate', 'bowl', 'fork', 'knife', 'spoon', 'soda can', 'can', 'lamp'],
            'animais': ['dog', 'cat', 'bird'],
            'veículos': ['car', 'bicycle', 'motorcycle'],
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe', 'sneakers'],
            'banheiro': ['toilet', 'sink'],
            'portas/janelas': ['door', 'window'],
            'plantas': ['potted plant', 'flower'],
            'decoração': ['picture', 'painting'],
            'esportes': ['sports ball', 'ball', 'frisbee', 'snowboard', 'sports bottle'],
            'acessórios': ['umbrella']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"

    def _classificar_nivel_confianca(self, confianca):
        """Classifica nível de confiança"""
        if confianca >= 0.8:
            return "alta"
        elif confianca >= 0.5:
            return "media"
        else:
            return "baixa"

    def _gerar_logs_diagnostico(self, objetos_detectados, faces_detectadas):
        """Gera logs para diagnóstico técnico"""
        logs = []
        
        confiancas_obj = [obj.get('confidence', 0) for obj in objetos_detectados]
        if confiancas_obj:
            avg = sum(confiancas_obj)/len(confiancas_obj) if confiancas_obj else 0
            logs.append(f"Confiança objetos: min={min(confiancas_obj):.2f}, max={max(confiancas_obj):.2f}, avg={avg:.2f}")
        
        tipos_objetos = Counter([obj.get('name', 'desconhecido') for obj in objetos_detectados])
        if tipos_objetos:
            logs.append(f"Tipos objetos: {dict(tipos_objetos)}")
        
        if faces_detectadas:
            faces_conhecidas = len([f for f in faces_detectadas if f.get('name', 'Desconhecido') != 'Desconhecido'])
            logs.append(f"Faces: {faces_conhecidas} conhecidas, {len(faces_detectadas) - faces_conhecidas} desconhecidas")
        
        return logs