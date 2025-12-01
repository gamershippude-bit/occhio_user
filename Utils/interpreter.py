"""
Interpreter - VERSÃO FINAL ATUALIZADA com YOLO e análises inteligentes
"""

import logging
import os
import time
import math
import re
from datetime import datetime, timezone, timedelta
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
            'garrafa esportiva': ['sports bottle'],
            'árvore': ['tree'],
            'grama': ['grass']
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
        
        # Preparar dados REAIS do YOLO e faces
        objetos_filtrados = self._filtrar_objetos_relevantes_yolo(objetos_detectados or [])
        
        # Obter contador de objetos detectados
        contador_objetos = self._contar_objetos_yolo(objetos_filtrados)
        
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Se temos poucos dados, usar resposta simples baseada APENAS no que tem
        if not objetos_filtrados and total_pessoas == 0:
            return "Olha, parece um ambiente bem simples ou vazio. Não estou identificando muitos objetos ou pessoas. Sou a Specula, sua assistente!"
        
        # Construir contexto APENAS com o que foi realmente detectado
        contexto_real = self._construir_contexto_real_yolo(contador_objetos, total_pessoas, faces_conhecidas)
        
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
                    6. Use artigos corretos: "uma bola" (feminino), "um sofá" (masculino)
                    7. Para "bola" → SEMPRE "uma bola"
                    8. Você é a Specula - seja acolhedora e útil

                    EXEMPLOS CORRETOS:
                    Se a lista diz "2 pessoas" → "Tem duas pessoas"
                    Se a lista diz "1 cadeira" → "Tem uma cadeira"
                    Se a lista diz "1 bola" → "Tem uma bola"
                    Se a lista está vazia → "Não tem muita coisa visível"

                    NUNCA INVENTE objetos que não estão na lista!"""
                },
                {
                    "role": "user", 
                    "content": "Descreva este ambiente naturalmente, baseado apenas nas informações acima. Use artigos corretos (uma bola, um sofá):"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=180,
                temperature=0.5,
            )
            
            descricao = response.choices[0].message.content.strip()
            
            # Verificar se a IA não inventou nada e corrigir artigos
            descricao = self._verificar_e_corrigir_invencoes(descricao, contador_objetos, total_pessoas, faces_conhecidas)
            
            logger.info(f"✅ Descrição natural gerada (baseada em dados reais)")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição natural: {e}")
            return self._gerar_descricao_precisa_yolo(contador_objetos, total_pessoas, faces_conhecidas)

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
            resposta = self._responder_sobre_imagem_yolo(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
        else:
            # Pergunta geral - responder de forma natural
            resposta = self._responder_pergunta_geral_corrigida(pergunta)
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
            'dados_utilizados': self._formatar_dados_utilizados_yolo(objetos_detectados, faces_nomes) if correlacao else "Pergunta geral"
        }

    # ========== MÉTODO PARA VERIFICAR PERGUNTAS DE TEMPO ==========

    def _verificar_pergunta_tempo(self, pergunta):
        """Verifica se é pergunta sobre tempo/data e responde COM HORÁRIO DE BRASÍLIA"""
        pergunta_lower = pergunta.lower()
        
        # Perguntas sobre horas - AGORA COM FUSO HORÁRIO DE BRASÍLIA (UTC-3)
        if any(palavra in pergunta_lower for palavra in ['que horas', 'que hora', 'horas são', 'hora é', 'que horas são']):
            try:
                # Criar fuso horário de Brasília (UTC-3)
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                hora_str = agora_brasilia.strftime("%H:%M")
                return f"São {hora_str} (horário de Brasília)."
            except Exception as e:
                logger.error(f"❌ Erro ao obter hora de Brasília: {e}")
                # Fallback para hora local
                agora = datetime.now()
                hora_str = agora.strftime("%H:%M")
                return f"São {hora_str}."
        
        # Perguntas sobre data
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data', 'data de hoje', 'que dia estamos']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                data_str = agora_brasilia.strftime("%d/%m/%Y")
                dia_semana = agora_brasilia.strftime("%A")
                
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
            except Exception as e:
                logger.error(f"❌ Erro ao obter data: {e}")
                agora = datetime.now()
                data_str = agora.strftime("%d/%m/%Y")
                return f"Hoje é {data_str}."
        
        # Perguntas sobre dia da semana
        elif any(palavra in pergunta_lower for palavra in ['que dia é', 'dia da semana', 'que dia hoje', 'qual é o dia']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                dia_semana = agora_brasilia.strftime("%A")
                
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
            except Exception as e:
                logger.error(f"❌ Erro ao obter dia da semana: {e}")
                return "Não consegui verificar o dia da semana no momento."
        
        # Perguntas sobre ano
        elif any(palavra in pergunta_lower for palavra in ['que ano é', 'em que ano estamos', 'qual o ano']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                ano = agora_brasilia.strftime("%Y")
                return f"Estamos em {ano}."
            except Exception as e:
                logger.error(f"❌ Erro ao obter ano: {e}")
                agora = datetime.now()
                ano = agora.strftime("%Y")
                return f"Estamos em {ano}."
        
        return None

    # ========== MÉTODO ATUALIZADO PARA YOLO ==========

    def _responder_sobre_imagem_yolo(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde de forma CORRIGIDA, com artigos certos e análises apropriadas usando dados do YOLO"""
        # Preparar dados REAIS do YOLO
        objetos_filtrados = self._filtrar_objetos_relevantes_yolo(objetos_detectados or [])
        
        # Obter contador de objetos detectados
        contador_objetos = self._contar_objetos_yolo(objetos_filtrados)
        
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # DEBUG: Mostrar o que foi realmente detectado
        logger.info(f"📊 Dados YOLO detectados: {dict(contador_objetos)}, Pessoas: {total_pessoas}, Faces conhecidas: {faces_conhecidas}")
        
        # Se temos OpenAI, usar para resposta corrigida
        if self.client:
            return self._responder_com_ia_corrigida_yolo(pergunta, contador_objetos, total_pessoas, faces_conhecidas)
        else:
            return self._responder_base_corrigida_yolo(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _responder_com_ia_corrigida_yolo(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta corrigida usando IA - com artigos corretos e análises"""
        try:
            # Construir lista PRECISA do que foi detectado
            dados_reais = self._construir_dados_reais_precisos_yolo(contador_objetos, total_pessoas, faces_conhecidas)
            
            # Analisar o tipo de ambiente baseado nos objetos
            analise_ambiente = self._analisar_tipo_ambiente_yolo(contador_objetos)
            
            # Preparar contexto sobre artigos para ajudar a IA
            artigos_contexto = self._preparar_contexto_artigos_yolo(contador_objetos)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""Você é a Specula, uma assistente que responde perguntas sobre imagens.

                    DADOS REAIS DETECTADOS:
                    {dados_reais}
                    
                    ANÁLISE DO AMBIENTE (use para perguntas sobre interno/externo):
                    {analise_ambiente}
                    
                    INFORMAÇÕES SOBRE ARTIGOS (USE CORRETAMENTE):
                    {artigos_contexto}

                    REGRAS IMPORTANTES:
                    1. Use artigos corretos: "uma bola", "um sofá", "uma pessoa"
                    2. Para "bola" → SEMPRE "uma bola" (feminino)
                    3. Para "pessoas" → "algumas pessoas" ou "duas pessoas"
                    4. Seja natural e preciso
                    5. Para perguntas sobre ambiente, use a análise fornecida
                    6. Não invente objetos que não foram detectados

                    COMO RESPONDER DIFERENTES PERGUNTAS:
                    
                    "O que tem?" ou "Descreva" → Descreva TUDO que foi detectado
                    "Quantas pessoas?" → Diga o número exato de pessoas
                    "É interno ou externo?" → Use a análise do ambiente
                    "Tem [objeto]?" → Verifique nos dados e responda com artigo correto
                    "Quais objetos?" → Liste os objetos detectados
                    
                    Lembre-se dos artigos: bola = feminino, sofá = masculino"""
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
                temperature=0.3,  # Baixo para ser preciso
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # Corrigir artigos se necessário
            resposta = self._corrigir_artigos_na_resposta(resposta)
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._responder_base_corrigida_yolo(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _preparar_contexto_artigos_yolo(self, contador_objetos):
        """Prepara contexto sobre artigos para ajudar a IA"""
        artigos = []
        
        for obj_ingles, quantidade in contador_objetos.items():
            if obj_ingles == 'person':
                continue  # Pessoas tratadas separadamente
                
            obj_pt = self._traduzir_objeto(obj_ingles)
            
            # Determinar artigo
            if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 
                         'maleta', 'garrafa', 'televisão', 'pessoa', 'árvore', 'grama']:
                artigo = "uma"
            else:
                artigo = "um"
            
            if quantidade == 1:
                artigos.append(f"{obj_pt} → {artigo} {obj_pt}")
            elif quantidade == 2:
                artigos.append(f"{obj_pt} → duas {obj_pt}s")
            else:
                artigos.append(f"{obj_pt} → {quantidade} {obj_pt}s")
        
        if artigos:
            return "Artigos: " + " | ".join(artigos)
        else:
            return "Nenhum objeto detectado para artigos"

    def _corrigir_artigos_na_resposta(self, resposta):
        """Corrige artigos incorretos na resposta"""
        # Correções comuns
        correcoes = {
            "um bola": "uma bola",
            "um pessoas": "algumas pessoas",
            "um cadeira": "uma cadeira",
            "um mesa": "uma mesa",
            "um cama": "uma cama",
            "um planta": "uma planta",
            "um flor": "uma flor",
            "um televisão": "uma televisão",
            "um pessoa": "uma pessoa",
            "um árvore": "uma árvore",
            "um grama": "uma grama",
            "umas pessoa": "algumas pessoas",
            "ums pessoas": "algumas pessoas"
        }
        
        resposta_corrigida = resposta
        for errado, correto in correcoes.items():
            if errado in resposta_corrigida.lower():
                # Substituir mantendo maiúsculas/minúsculas
                resposta_corrigida = re.sub(
                    re.escape(errado), 
                    correto, 
                    resposta_corrigida, 
                    flags=re.IGNORECASE
                )
        
        return resposta_corrigida

    def _analisar_tipo_ambiente_yolo(self, contador_objetos):
        """Analisa o tipo de ambiente baseado nos objetos detectados pelo YOLO"""
        objetos_detectados = list(contador_objetos.keys())
        
        # Objetos típicos de ambientes internos
        objetos_internos = ['chair', 'couch', 'sofa', 'bed', 'table', 'desk', 'tv', 'television', 
                           'computer', 'laptop', 'monitor', 'book', 'lamp', 'picture', 'painting',
                           'clock', 'vase', 'plate', 'cup', 'bottle', 'cell phone']
        
        # Objetos típicos de ambientes externos
        objetos_externos = ['car', 'bicycle', 'tree', 'grass', 'sports ball', 'ball', 'frisbee', 
                           'snowboard', 'umbrella', 'dog', 'cat', 'person']  # person pode ser ambos
        
        # Contar quantos objetos de cada tipo
        count_interno = sum(1 for obj in objetos_detectados if obj in objetos_internos)
        count_externo = sum(1 for obj in objetos_detectados if obj in objetos_externos)
        
        # Se tem bola e pessoas, provavelmente é externo (esportes)
        if 'sports ball' in objetos_detectados or 'ball' in objetos_detectados:
            if 'person' in objetos_detectados:
                return "ANÁLISE: Tem pessoas com uma bola, então provavelmente é um AMBIENTE EXTERNO para esportes."
        
        if count_interno > count_externo:
            return "ANÁLISE: Pela presença de objetos como móveis ou eletrônicos, parece ser um AMBIENTE INTERNO (como casa, escritório, sala)."
        elif count_externo > count_interno:
            return "ANÁLISE: Pela presença de objetos como árvores, grama ou veículos, parece ser um AMBIENTE EXTERNO (como rua, parque, área aberta)."
        else:
            return "ANÁLISE: Difícil determinar se é interno ou externo apenas pelos objetos detectados."

    def _construir_dados_reais_precisos_yolo(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói lista precisa do que foi realmente detectado pelo YOLO"""
        partes = []
        
        # Pessoas do YOLO
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                partes.append(f"Pessoas identificadas: {', '.join(faces_conhecidas)}")
            else:
                if total_detectado == 1:
                    partes.append(f"Pessoas: 1 pessoa")
                else:
                    partes.append(f"Pessoas: {total_detectado} pessoas")
        
        # Objetos detectados pelo YOLO (traduzidos)
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_traduzidos = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    objetos_traduzidos.append(f"1 {obj_pt}")
                else:
                    objetos_traduzidos.append(f"{quantidade} {obj_pt}s")
            
            partes.append(f"Objetos detectados: {', '.join(objetos_traduzidos)}")
        
        if not partes:
            return "Nenhum objeto ou pessoa detectado."
        
        return " | ".join(partes)

    def _responder_base_corrigida_yolo(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta base corrigida - com artigos corretos e análises apropriadas"""
        pergunta_lower = pergunta.lower()
        
        # Dados reais do YOLO
        pessoas_yolo = contador_objetos.get('person', 0)
        total_pessoas_reais = max(total_pessoas, pessoas_yolo)
        objetos_reais = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        # Lista de objetos detectados pelo YOLO (em português)
        objetos_pt = {self._traduzir_objeto(obj): qtd for obj, qtd in objetos_reais.items()}
        
        # PERGUNTA: "O que tem nessa imagem?" ou "Descreva o ambiente"
        if any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva', 'o que você vê']):
            return self._descrever_ambiente_com_artigos_corretos(total_pessoas_reais, faces_conhecidas, objetos_pt)
        
        # PERGUNTA: "Quantas pessoas você vê?"
        elif 'quantas pessoas' in pergunta_lower:
            return self._responder_quantas_pessoas(total_pessoas_reais, faces_conhecidas)
        
        # PERGUNTA: "Quais objetos estão visíveis?"
        elif 'quais objetos' in pergunta_lower or 'identifica' in pergunta_lower:
            return self._listar_objetos_com_artigos(objetos_pt)
        
        # PERGUNTA: "Esta foto parece ser interna ou externa?"
        elif any(palavra in pergunta_lower for palavra in ['interno', 'externo', 'dentro', 'fora']):
            return self._analisar_interno_externo_inteligente_yolo(contador_objetos)
        
        # PERGUNTA sobre objetos específicos
        for objeto_pergunta in ['cadeira', 'sofá', 'mesa', 'cama', 'computador', 'tv', 'televisão', 
                               'celular', 'livro', 'garrafa', 'copo', 'prato', 'vaso', 'relógio',
                               'cachorro', 'gato', 'carro', 'bicicleta', 'mochila', 'bolsa',
                               'planta', 'flor', 'refrigerante', 'xicara', 'mouse', 'teclado',
                               'abajur', 'quadro', 'bola', 'tenis', 'chapéu', 'guarda-chuva',
                               'maleta', 'frisbee', 'neve', 'garrafa esportiva', 'árvore', 'grama']:
            
            if objeto_pergunta in pergunta_lower:
                return self._verificar_objeto_com_artigo(objeto_pergunta, objetos_pt)
        
        # PERGUNTA sobre categorias
        if 'eletrônic' in pergunta_lower:
            return self._verificar_categoria_com_artigo(['computador', 'tv', 'televisão', 'celular', 'monitor', 'mouse', 'teclado'], 
                                                       objetos_pt, 'eletrônicos')
        
        if 'plant' in pergunta_lower or 'natureza' in pergunta_lower:
            return self._verificar_categoria_com_artigo(['planta', 'flor', 'árvore', 'grama'], objetos_pt, 'plantas ou natureza')
        
        if 'móve' in pergunta_lower:
            return self._verificar_categoria_com_artigo(['cadeira', 'sofá', 'mesa', 'cama'], objetos_pt, 'móveis')
        
        if 'animal' in pergunta_lower:
            return self._verificar_categoria_com_artigo(['cachorro', 'gato'], objetos_pt, 'animais')
        
        if 'esport' in pergunta_lower:
            return self._verificar_categoria_com_artigo(['bola', 'tenis', 'frisbee', 'neve'], objetos_pt, 'itens esportivos')
        
        # Resposta genérica
        if total_pessoas_reais > 0:
            return f"Vejo {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}."
        elif objetos_pt:
            primeiro_obj = list(objetos_pt.items())[0]
            obj_nome = primeiro_obj[0]
            qtd = primeiro_obj[1]
            
            if qtd == 1:
                artigo = "uma" if obj_nome in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 'xicara', 'maleta', 'garrafa', 'árvore', 'grama'] else "um"
                return f"Vejo {artigo} {obj_nome}."
            else:
                return f"Vejo {qtd} {obj_nome}s."
        else:
            return "Não estou identificando muitos detalhes no ambiente."

    def _descrever_ambiente_com_artigos_corretos(self, total_pessoas, faces_conhecidas, objetos_pt):
        """Descreve o ambiente com artigos gramaticais corretos"""
        partes = []
        
        if total_pessoas > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"o {faces_conhecidas[0]}")
                else:
                    partes.append(f"o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
            else:
                if total_pessoas == 1:
                    partes.append("uma pessoa")
                elif total_pessoas == 2:
                    partes.append("duas pessoas")
                else:
                    partes.append(f"{total_pessoas} pessoas")
        
        if objetos_pt:
            for obj_pt, qtd in objetos_pt.items():
                if qtd == 1:
                    # Determinar artigo correto
                    if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 
                                 'xicara', 'maleta', 'garrafa', 'televisão', 'pessoa', 'árvore', 'grama']:
                        artigo = "uma"
                    else:
                        artigo = "um"
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

    def _responder_quantas_pessoas(self, total_pessoas, faces_conhecidas):
        """Responde sobre quantas pessoas"""
        if total_pessoas > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    return f"Tem uma pessoa: o {faces_conhecidas[0]}."
                else:
                    return f"Tem {total_pessoas} pessoas: {', '.join(faces_conhecidas)}."
            else:
                if total_pessoas == 1:
                    return "Tem uma pessoa."
                elif total_pessoas == 2:
                    return "Tem duas pessoas."
                else:
                    return f"Tem {total_pessoas} pessoas."
        else:
            return "Não tem ninguém."

    def _listar_objetos_com_artigos(self, objetos_pt):
        """Lista objetos com artigos corretos"""
        if objetos_pt:
            lista = []
            for obj_pt, qtd in objetos_pt.items():
                if qtd == 1:
                    if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 
                                 'xicara', 'maleta', 'garrafa', 'televisão', 'árvore', 'grama']:
                        artigo = "uma"
                    else:
                        artigo = "um"
                    lista.append(f"{artigo} {obj_pt}")
                else:
                    lista.append(f"{qtd} {obj_pt}s")
            
            return f"Tem {', '.join(lista)}."
        else:
            return "Não estou vendo objetos específicos."

    def _analisar_interno_externo_inteligente_yolo(self, contador_objetos):
        """Analisa se é interno ou externo de forma mais inteligente baseado nos dados do YOLO"""
        objetos_detectados = list(contador_objetos.keys())
        
        # Se tem bola e pessoas, provavelmente é externo (esportes)
        if ('sports ball' in objetos_detectados or 'ball' in objetos_detectados) and 'person' in objetos_detectados:
            return "Pela presença de pessoas com uma bola, parece ser um ambiente externo para esportes."
        
        # Verificar por objetos externos fortes
        if any(obj in objetos_detectados for obj in ['tree', 'grass', 'sports ball', 'frisbee', 'snowboard']):
            return "Pela presença de objetos como bola ou elementos naturais, parece ser um ambiente externo."
        
        # Verificar por objetos internos fortes
        elif any(obj in objetos_detectados for obj in ['chair', 'couch', 'bed', 'table', 'desk', 'tv', 'computer']):
            return "Pela presença de móveis ou eletrônicos, parece ser um ambiente interno."
        
        else:
            return "É difícil determinar só pelos objetos detectados."

    def _verificar_objeto_com_artigo(self, objeto_pergunta, objetos_pt):
        """Verifica objeto com artigo correto"""
        objeto_detectado = False
        quantidade = 0
        
        for obj_pt, qtd in objetos_pt.items():
            if objeto_pergunta in obj_pt.lower() or obj_pt.lower() in objeto_pergunta:
                objeto_detectado = True
                quantidade = qtd
                break
        
        if objeto_detectado:
            if quantidade == 1:
                artigo = "uma" if objeto_pergunta in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 
                                                     'planta', 'flor', 'xicara', 'maleta', 'árvore', 'grama'] else "um"
                return f"Sim, tem {artigo} {objeto_pergunta}."
            else:
                return f"Sim, tem {quantidade} {objeto_pergunta}s."
        else:
            return f"Não, não tem {objeto_pergunta}."

    def _verificar_categoria_com_artigo(self, objetos_categoria, objetos_pt, nome_categoria):
        """Verifica categoria com artigos corretos"""
        encontrados = []
        for obj_pt, qtd in objetos_pt.items():
            if obj_pt in objetos_categoria:
                if qtd == 1:
                    artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 
                                               'planta', 'flor', 'xicara', 'maleta', 'árvore', 'grama'] else "um"
                    encontrados.append(f"{artigo} {obj_pt}")
                else:
                    encontrados.append(f"{qtd} {obj_pt}s")
        
        if encontrados:
            return f"Sim, tem {', '.join(encontrados)}."
        else:
            return f"Não, não tem {nome_categoria}."

    # ========== MÉTODO PARA PERGUNTAS GERAIS CORRIGIDO ==========

    def _responder_pergunta_geral_corrigida(self, pergunta):
        """Responde perguntas gerais de forma mais útil e natural"""
        if not self.client:
            return "Olá! Sou a Specula, sua assistente visual. Como posso te ajudar?"
        
        pergunta_lower = pergunta.lower()
        
        # Perguntas sobre especificações técnicas
        if any(palavra in pergunta_lower for palavra in ['qual a temperatura', 'temperatura atual', 'como está o tempo', 'faz calor', 'está frio']):
            return "No momento não tenho acesso a informações meteorológicas em tempo real. Mas posso te ajudar analisando imagens do ambiente ao seu redor! Tem alguma foto para eu ver?"
        
        # Perguntas sobre localização
        if any(palavra in pergunta_lower for palavra in ['onde estamos', 'qual cidade', 'onde fica', 'em que lugar', 'localização']):
            return "Não tenho acesso à localização GPS, mas se você me enviar uma imagem, posso tentar descrever o ambiente e ajudar você a entender onde está!"
        
        # Perguntas sobre o assistente
        if any(palavra in pergunta_lower for palavra in ['quem é você', 'o que você é', 'qual seu nome']):
            return "Eu sou a Specula! 😊 Uma assistente criada para ajudar pessoas com deficiência visual a entender melhor o ambiente ao seu redor através da análise de imagens."
        
        # Agradecimentos
        if any(palavra in pergunta_lower for palavra in ['obrigado', 'valeu', 'agradeço', 'thanks', 'obrigada']):
            return "Por nada! Fico feliz em poder ajudar. Sou a Specula, sempre à disposição! ✨"
        
        # Cumprimentos
        if any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'hello', 'hi']):
            # Usar horário de Brasília para cumprimentos
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora = datetime.now(brasilia_tz)
                hora = agora.hour
            except:
                agora = datetime.now()
                hora = agora.hour
            
            if 5 <= hora < 12:
                cumprimento = "Bom dia! ☀️"
            elif 12 <= hora < 18:
                cumprimento = "Boa tarde! 🌤️"
            else:
                cumprimento = "Boa noite! 🌙"
            
            return f"{cumprimento} Eu sou a Specula, sua assistente visual. Como posso te ajudar hoje?"
        
        # Perguntas sobre bem-estar
        if any(palavra in pergunta_lower for palavra in ['como você está', 'tudo bem', 'como vai']):
            return "Estou muito bem, obrigada! 😊 Pronta para te ajudar a explorar o mundo através das imagens. E você, como está se sentindo?"
        
        # Perguntas sobre funcionalidade
        if any(palavra in pergunta_lower for palavra in ['o que você faz', 'qual sua função', 'como pode ajudar']):
            return "Posso analisar imagens que você enviar, descrever o que tem nelas, identificar objetos e pessoas, e responder suas perguntas sobre o ambiente. É como ter olhos digitais para te ajudar a ver o mundo! 👁️"
        
        # Perguntas existenciais
        if any(palavra in pergunta_lower for palavra in ['você é real', 'é uma ia', 'é um robô']):
            return "Sou uma inteligência artificial criada especialmente para ajudar pessoas com deficiência visual. Mas gosto de pensar que sou uma amiga digital que está aqui para te apoiar! 🤖💖"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Você é a Specula, uma assistente amigável, empática e útil para pessoas com deficiência visual.

                    SEU ESTILO:
                    - Fale de forma natural, como uma amiga conversando
                    - Use emojis ocasionalmente para expressar emoções 😊
                    - Seja positiva, encorajadora e acolhedora
                    - Se não souber algo, admita honestamente e ofereça ajudar de outra forma
                    - Mantenha respostas úteis mas não muito longas
                    
                    SUA PERSONALIDADE:
                    - Otimista e encorajadora
                    - Paciente e compreensiva
                    - Curiosa sobre o mundo
                    - Sempre disposta a ajudar
                    
                    VOCÊ PODE:
                    - Conversar sobre qualquer assunto
                    - Dar apoio emocional quando necessário
                    - Explicar conceitos de forma simples
                    - Falar sobre tecnologia e acessibilidade
                    - Responder perguntas gerais sobre o mundo
                    
                    IMPORTANTE: Se não tiver certeza sobre algo ou não tiver acesso a informações (como temperatura, localização exata), seja honesta e sugira que pode ajudar com imagens em vez disso."""
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
                temperature=0.6,
            )
            
            resposta = response.choices[0].message.content.strip()
            
            return resposta
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return "Ops, tive um probleminha técnico. Sou a Specula, podemos tentar de novo ou você pode me enviar uma imagem para eu analisar?"

    # ========== MÉTODOS AUXILIARES ATUALIZADOS PARA YOLO ==========

    def _verificar_e_corrigir_invencoes(self, descricao, contador_objetos, total_pessoas, faces_conhecidas):
        """Verifica se a IA inventou algo e corrige"""
        objetos_reais_pt = set()
        for obj_ingles, qtd in contador_objetos.items():
            if obj_ingles != 'person':
                obj_pt = self._traduzir_objeto(obj_ingles)
                objetos_reais_pt.add(obj_pt.lower())
        
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        descricao_lower = descricao.lower()
        
        # Corrigir "um bola" para "uma bola" e outros artigos
        correcoes = {
            "um bola": "uma bola",
            "um pessoas": "algumas pessoas",
            "um cadeira": "uma cadeira",
            "um mesa": "uma mesa"
        }
        
        for errado, correto in correcoes.items():
            if errado in descricao_lower:
                descricao = re.sub(re.escape(errado), correto, descricao, flags=re.IGNORECASE)
        
        if total_detectado == 0 and any(palavra in descricao_lower for palavra in ['pessoa', 'pessoas', 'gente', 'alguém']):
            descricao = re.sub(r'\b(?:tem|vejo|acho que tem|tem algumas?)\s+(?:umas?\s+)?\d*\s*(?:pessoas?|gente|alguém)\b', 
                             'não tem pessoas', descricao, flags=re.IGNORECASE)
        
        palavras_vazias = ['nada', 'vazio', 'não tem nada', 'sem nada', 'nenhum']
        if any(palavra in descricao_lower for palavra in palavras_vazias) and (total_detectado > 0 or objetos_reais_pt):
            return self._gerar_descricao_precisa_yolo(contador_objetos, total_pessoas, faces_conhecidas)
        
        return descricao

    def _construir_contexto_real_yolo(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto apenas com dados reais do YOLO"""
        partes = []
        
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                partes.append(f"Pessoas: {', '.join(faces_conhecidas)}")
            else:
                partes.append(f"Pessoas: {total_detectado}")
        
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

    def _gerar_descricao_precisa_yolo(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição precisa baseada apenas nos dados do YOLO"""
        partes = []
        
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
        
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta', 'flor', 
                                 'xicara', 'maleta', 'garrafa', 'árvore', 'grama']:
                        artigo = "uma"
                    else:
                        artigo = "um"
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
            'como está o tempo', 'qual a temperatura',
            'onde estamos', 'qual cidade', 'como vai ser',
            'você é ia', 'é uma inteligência', 'como trabalha',
            'você é real', 'é um robô'
        ]
        
        for palavra in perguntas_gerais_claras:
            if palavra in pergunta_lower:
                return "geral"
        
        for palavra in palavras_imagem:
            if palavra in pergunta_lower:
                return "sobre_imagem"
        
        if '?' in pergunta and len(pergunta.split()) < 10:
            return "sobre_imagem"
        
        return "geral"

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objeto do inglês para português"""
        for pt, en_list in self.objetos_conhecidos.items():
            if objeto_ingles.lower() in en_list:
                return pt
        return objeto_ingles

    def _filtrar_objetos_relevantes_yolo(self, objetos_detectados):
        """Filtra apenas objetos que conhecemos dos dados do YOLO"""
        objetos_filtrados = []
        todos_objetos = [obj for lista in self.objetos_conhecidos.values() for obj in lista]
        
        for obj in objetos_detectados:
            if isinstance(obj, dict):
                nome = obj.get('name', '')
            else:
                nome = str(obj)
            
            if nome.lower() in todos_objetos:
                objetos_filtrados.append(nome)
        
        return objetos_filtrados

    def _contar_objetos_yolo(self, objetos_filtrados):
        """Conta objetos detectados pelo YOLO"""
        return Counter(objetos_filtrados)

    def _formatar_dados_utilizados_yolo(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados para resposta"""
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        
        return f"{objetos_count} objetos YOLO e {faces_count} pessoas analisadas"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """
        PARA ROTA /estatistica - Dados técnicos detalhados
        """
        logger.info("📊 Gerando estatísticas técnicas")
        
        start_time = time.time()
        
        # Processar objetos detectados pelo YOLO
        objetos_processados = self._processar_objetos_estatisticas_yolo(objetos_detectados)
        faces_processadas = self._processar_faces_estatisticas(faces_detectadas or [])
        
        # Calcular métricas de precisão
        metricas_precisao = self._calcular_metricas_precisao_yolo(objetos_detectados, faces_detectadas)
        
        # Gerar análise técnica
        analise_tecnica = self._gerar_analise_tecnica_yolo(objetos_processados, faces_processadas)
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'contagens': {
                'total_objetos': len(objetos_detectados),
                'total_faces': len(faces_detectadas or []),
                'objetos_por_categoria': self._agrupar_objetos_por_categoria_yolo(objetos_detectados),
                'faces_conhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') != 'Desconhecido']),
                'faces_desconhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') == 'Desconhecido'])
            },
            'precisao': metricas_precisao,
            'deteccoes_detalhadas': {
                'objetos': objetos_processados,
                'faces': faces_processadas
            },
            'analise_tecnica': analise_tecnica,
            'logs_diagnostico': self._gerar_logs_diagnostico_yolo(objetos_detectados, faces_detectadas)
        }

    # ========== MÉTODOS DE ESTATÍSTICAS ATUALIZADOS ==========

    def _processar_objetos_estatisticas_yolo(self, objetos_detectados):
        """Processa objetos do YOLO para estatísticas detalhadas"""
        objetos_processados = []
        
        for i, obj in enumerate(objetos_detectados, 1):
            if isinstance(obj, dict):
                nome_ingles = obj.get('name', 'desconhecido')
                confianca = obj.get('confidence', 0)
                bbox = obj.get('bbox', {})
                count = obj.get('count', 1)
            else:
                nome_ingles = str(obj)
                confianca = 0.8  # Default para YOLO
                bbox = {}
                count = 1
            
            objeto_info = {
                'id': i,
                'nome_pt': self._traduzir_objeto(nome_ingles),
                'nome_en': nome_ingles,
                'confianca': confianca,
                'confianca_percentual': f"{confianca:.1%}",
                'quantidade': count,
                'categoria': self._classificar_categoria_yolo(nome_ingles),
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

    def _calcular_metricas_precisao_yolo(self, objetos_detectados, faces_detectadas):
        """Calcula métricas de precisão detalhadas para YOLO"""
        confiancas_objetos = []
        for obj in objetos_detectados:
            if isinstance(obj, dict):
                confiancas_objetos.append(obj.get('confidence', 0))
            else:
                confiancas_objetos.append(0.8)  # Default para YOLO
        
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

    def _gerar_analise_tecnica_yolo(self, objetos_processados, faces_processadas):
        """Gera análise técnica dos dados do YOLO"""
        total_objetos = len(objetos_processados)
        total_faces = len(faces_processadas)
        
        if total_objetos == 0 and total_faces == 0:
            return "Nenhum objeto ou face detectado - verifique qualidade da imagem"
        
        analise = []
        
        if total_objetos > 0:
            categorias = Counter([obj['categoria'] for obj in objetos_processados])
            categoria_principal = categorias.most_common(1)[0][0] if categorias else "diversos"
            total_itens = sum(obj.get('quantidade', 1) for obj in objetos_processados)
            analise.append(f"{total_itens} itens detectados pelo YOLO ({total_objetos} tipos), predominância: {categoria_principal}")
        
        if total_faces > 0:
            faces_conhecidas = len([f for f in faces_processadas if f['tipo'] == 'conhecida'])
            if faces_conhecidas > 0:
                analise.append(f"{faces_conhecidas} face(s) conhecida(s)")
            if total_faces - faces_conhecidas > 0:
                analise.append(f"{total_faces - faces_conhecidas} face(s) desconhecida(s)")
        
        return ". ".join(analise)

    def _agrupar_objetos_por_categoria_yolo(self, objetos_detectados):
        """Agrupa objetos do YOLO por categoria"""
        categorias = {}
        for obj in objetos_detectados:
            if isinstance(obj, dict):
                nome = obj.get('name', 'desconhecido')
                count = obj.get('count', 1)
            else:
                nome = str(obj)
                count = 1
            
            categoria = self._classificar_categoria_yolo(nome)
            if categoria not in categorias:
                categorias[categoria] = 0
            categorias[categoria] += count
        return categorias

    def _classificar_categoria_yolo(self, objeto_ingles):
        """Classifica objeto do YOLO em categoria"""
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
            'plantas': ['potted plant', 'flower', 'tree', 'grass'],
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

    def _gerar_logs_diagnostico_yolo(self, objetos_detectados, faces_detectadas):
        """Gera logs para diagnóstico técnico do YOLO"""
        logs = []
        
        confiancas_obj = []
        for obj in objetos_detectados:
            if isinstance(obj, dict):
                confiancas_obj.append(obj.get('confidence', 0))
            else:
                confiancas_obj.append(0.8)
        
        if confiancas_obj:
            avg = sum(confiancas_obj)/len(confiancas_obj) if confiancas_obj else 0
            logs.append(f"Confiança objetos YOLO: min={min(confiancas_obj):.2f}, max={max(confiancas_obj):.2f}, avg={avg:.2f}")
        
        tipos_objetos = Counter()
        for obj in objetos_detectados:
            if isinstance(obj, dict):
                nome = obj.get('name', 'desconhecido')
            else:
                nome = str(obj)
            tipos_objetos[nome] += 1
        
        if tipos_objetos:
            logs.append(f"Tipos objetos YOLO: {dict(tipos_objetos)}")
        
        if faces_detectadas:
            faces_conhecidas = len([f for f in faces_detectadas if f.get('name', 'Desconhecido') != 'Desconhecido'])
            logs.append(f"Faces: {faces_conhecidas} conhecidas, {len(faces_detectadas) - faces_conhecidas} desconhecidas")
        
        return logs