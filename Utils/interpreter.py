"""
Interpreter - VERSÃO FINAL CORRIGIDA com respostas variadas e classificação perfeita
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
            return self._gerar_descricao_precisa(contador_objetos, total_pessoas, faces_conhecidas)

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat preciso baseado apenas nos dados detectados
        """
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Primeiro verificar se é pergunta sobre tempo/data - AGORA ANTES DE CLASSIFICAR
        resposta_tempo = self._verificar_pergunta_tempo(pergunta)
        if resposta_tempo:
            processing_time = time.time() - start_time
            return {
                'sucesso': True,
                'timestamp': time.time(),
                'tempo_processamento': f"{processing_time:.2f}s",
                'pergunta': pergunta,
                'resposta': resposta_tempo,
                'tipo_pergunta': "geral",
                'correlacao_com_imagem': False,
                'dados_utilizados': "Pergunta geral (tempo/data)"
            }
        
        # Classificar o tipo de pergunta - USANDO MÉTODO CORRETO
        tipo_pergunta = self._classificar_tipo_pergunta(pergunta)
        logger.info(f"🔍 Pergunta classificada como: {tipo_pergunta}")
        
        if tipo_pergunta == "sobre_imagem":
            # Pergunta sobre a imagem - usar dados detectados
            resposta = self._responder_sobre_imagem_corrigida(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
            dados_utilizados = self._formatar_dados_utilizados(objetos_detectados, faces_nomes)
        else:
            # Pergunta geral - responder de forma natural
            resposta = self._responder_pergunta_geral_corrigida(pergunta)
            correlacao = False
            dados_utilizados = "Pergunta geral"
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'pergunta': pergunta,
            'resposta': resposta,
            'tipo_pergunta': tipo_pergunta,
            'correlacao_com_imagem': correlacao,
            'dados_utilizados': dados_utilizados
        }

    # ========== CLASSIFICAÇÃO DE PERGUNTAS CORRIGIDA ==========

    def _classificar_tipo_pergunta(self, pergunta):
        """Classifica se a pergunta é sobre a imagem ou geral - VERSÃO SIMPLES E EFETIVA"""
        pergunta_lower = pergunta.lower().strip()
        
        # PERGUNTAS CLARAMENTE GERAIS (não sobre imagem)
        gerais_absolutas = [
            # Perguntas sobre conceitos/definições
            'oque é', 'o que é', 'qual é ', 'quem foi ', 'quem é ',
            'história de', 'significado de', 'definição de',
            'explique ', 'explicar ', 'conceito de',
            
            # Informações geográficas
            'capital de', 'população de', 'onde fica ',
            'quantos habitantes', 'qual a população',
            
            # Piadas e filosofia
            'conta uma piada', 'piada sobre', 'piada do',
            'qual o sentido da vida', 'filosofia',
            
            # Perguntas sobre o assistente
            'quem é você', 'qual seu nome', 'o que você faz',
            'qual sua função', 'você é ', 'é uma ia', 'é um robô',
            'como trabalha', 'como funciona',
            
            # Agradecimentos
            'obrigado', 'valeu', 'agradeço', 'thanks',
            
            # Cumprimentos
            'oi', 'olá', 'bom dia', 'boa tarde', 'boa noite',
            'hello', 'hi', 'hey',
            
            # Perguntas sobre condições físicas
            'qual a temperatura', 'temperatura atual', 'faz calor',
            'está frio', 'como está o tempo', 'previsão do tempo',
            'está chovendo', 'faz sol',
            
            # Perguntas sobre localização
            'onde estamos', 'qual cidade', 'onde fica',
            'em que lugar', 'localização', 'em qual cidade',
            'em qual estado', 'em qual país',
            
            # Ano/Data (já tratado separadamente)
            'que ano é', 'em que ano', 'qual o ano',
            
            # Perguntas com definições
            'batata', 'abacaxi', 'computador', 'carro', 'casa',
            'animais', 'planetas', 'universo',
            
            # Perguntas que pedem explicações
            'como funciona', 'para que serve', 'para que é usado',
            
            # Perguntas sobre sentimentos/emoções
            'como está se sentindo', 'está feliz', 'está triste',
            'como você se sente',
            
            # Perguntas sobre bem-estar do assistente
            'como você está', 'tudo bem', 'como vai'
        ]
        
        # Verificar se é pergunta geral absoluta
        for padrao in gerais_absolutas:
            if padrao in pergunta_lower:
                return "geral"
        
        # PERGUNTAS CLARAMENTE SOBRE IMAGEM
        imagem_absolutas = [
            # Pronomes demonstrativos + imagem/foto
            'essa imagem', 'esta foto', 'na foto', 'na imagem',
            'nesta imagem', 'nesta foto', 'dessa imagem', 'dessa foto',
            'essa foto', 'esta imagem',
            
            # Perguntas sobre visão/detecção
            'o que você vê', 'o que está vendo', 'o que consegue ver',
            'o que identifica', 'o que reconhece', 'o que detecta',
            'está vendo', 'consegue ver', 'pode ver',
            
            # Descrever/analisar imagem
            'descreva a imagem', 'descreva a foto', 'analise a imagem',
            'analise a foto', 'descreva o que vê', 'descreva o ambiente',
            
            # Perguntas específicas sobre conteúdo
            'quem está na', 'onde está na', 
            'qual a cor do', 'qual a cor da', 'qual cor tem',
            
            # Ambiente interno/externo
            'é interno ou externo', 'está dentro ou fora',
            'parece ser interno', 'parece ser externo',
            'interno ou externo'
        ]
        
        # Verificar se é pergunta sobre imagem absoluta
        for padrao in imagem_absolutas:
            if padrao in pergunta_lower:
                return "sobre_imagem"
        
        # PERGUNTAS COM PALAVRAS-CHAVE DE IMAGEM
        palavras_chave_imagem = [
            # Palavras que indicam objetos visíveis
            'tem ', 'há ', 'vejo', 'vê', 'visualizo',
            'pessoa', 'pessoas', 'gente', 'alguém',
            'objeto', 'objetos', 'coisa', 'coisas',
            
            # Nomes de objetos específicos
            'cadeira', 'mesa', 'sofá', 'cama', 'computador',
            'tv', 'televisão', 'celular', 'planta', 'árvore',
            'animal', 'cachorro', 'gato', 'carro', 'livro',
            'garrafa', 'copo', 'prato', 'vaso', 'relógio',
            'mochila', 'bolsa', 'quadro', 'bola', 'tenis',
            'chapéu', 'guarda-chuva',
            
            # Palavras sobre ambiente
            'ambiente', 'lugar', 'sala', 'quarto', 'cozinha',
            'banheiro', 'escritório', 'parque', 'rua', 'jardim'
        ]
        
        # Se tem palavras-chave de imagem, é sobre imagem
        for palavra in palavras_chave_imagem:
            if palavra in pergunta_lower:
                # Verificar se não é um falso positivo
                if not self._e_falso_positivo_imagem(pergunta_lower, palavra):
                    return "sobre_imagem"
        
        # Se chegou aqui e tem interrogação, assume que é sobre imagem
        if '?' in pergunta:
            # Mas se for muito curta (1-3 palavras), pode ser geral
            palavras = pergunta_lower.split()
            if len(palavras) <= 3:
                return "geral"
            return "sobre_imagem"
        
        # Default para geral
        return "geral"
    
    def _e_falso_positivo_imagem(self, pergunta_lower, palavra_imagem):
        """Verifica se uma palavra de imagem é um falso positivo"""
        falsos_positivos = {
            'tem': ['tem perigo', 'tem problema', 'tem alguma dica', 'tem como', 'tem certeza'],
            'pessoa': ['pessoa famosa', 'pessoa importante', 'pessoa histórica'],
            'objeto': ['objeto de estudo', 'objeto histórico', 'objeto filosófico'],
            'ambiente': ['ambiente de trabalho', 'ambiente escolar', 'ambiente familiar'],
            'lugar': ['lugar turístico', 'lugar histórico', 'lugar famoso'],
            'animal': ['animal em extinção', 'animal selvagem', 'animal doméstico'],
            'planta': ['planta medicinal', 'planta ornamental', 'planta comestível']
        }
        
        if palavra_imagem in falsos_positivos:
            for falso_positivo in falsos_positivos[palavra_imagem]:
                if falso_positivo in pergunta_lower:
                    return True
        
        return False

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

    # ========== MÉTODO CORRIGIDO PARA RESPONDER SOBRE IMAGEM ==========

    def _responder_sobre_imagem_corrigida(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde de forma CORRIGIDA, com respostas variadas e precisas"""
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
        
        logger.info(f"📊 Dados reais: {dict(contador_objetos)}, Pessoas: {total_pessoas}")
        
        # Se temos OpenAI, usar para resposta
        if self.client:
            return self._responder_com_ia_melhorada(pergunta, contador_objetos, total_pessoas, faces_conhecidas)
        else:
            return self._responder_base_melhorada(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _responder_com_ia_melhorada(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta usando IA - com respostas muito mais variadas"""
        try:
            # Construir dados reais
            dados_reais = self._construir_dados_reais_detalhados(contador_objetos, total_pessoas, faces_conhecidas)
            
            # Analisar contexto
            analise_contexto = self._analisar_contexto_detalhado(contador_objetos, total_pessoas)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""Você é a Specula, uma assistente que responde perguntas sobre imagens.

                    DADOS REAIS DETECTADOS NA IMAGEM:
                    {dados_reais}
                    
                    ANÁLISE DE CONTEXTO:
                    {analise_contexto}

                    REGRAS IMPORTANTES:
                    1. Baseie sua resposta APENAS nos dados acima
                    2. Seja natural e variada nas respostas
                    3. Use artigos corretos: "uma bola", "um sofá", "uma pessoa"
                    4. Não invente objetos que não foram detectados
                    5. Seja útil e acolhedora

                    INSTRUÇÕES ESPECÍFICAS:
                    - Para "O que tem nessa imagem?": Descreva TUDO que foi detectado
                    - Para "Quantas pessoas você vê?": Diga APENAS o número de pessoas
                    - Para "Descreva o ambiente": Descreva o ambiente de forma completa
                    - Para "Quais objetos estão visíveis?": Liste apenas os objetos
                    - Para "É interno ou externo?": Use a análise de contexto
                    - Para perguntas do tipo "Tem [objeto]?": Responda especificamente sobre aquele objeto
                    
                    VARIEDADE NAS RESPOSTAS:
                    - Use diferentes formas de começar: "Na imagem...", "Vejo que...", "Analisando..."
                    - Para a mesma pergunta, dê respostas com estruturas diferentes
                    - Seja natural como uma conversa real"""
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
                temperature=0.7,  # Mais alto para variedade
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # Corrigir artigos básicos
            resposta = self._corrigir_artigos_basicos(resposta)
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._responder_base_melhorada(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _construir_dados_reais_detalhados(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói dados detalhados para a IA"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"1 pessoa identificada: {faces_conhecidas[0]}")
                else:
                    partes.append(f"{total_detectado} pessoas identificadas: {', '.join(faces_conhecidas)}")
            else:
                if total_detectado == 1:
                    partes.append("1 pessoa não identificada")
                else:
                    partes.append(f"{total_detectado} pessoas não identificadas")
        else:
            partes.append("0 pessoas")
        
        # Objetos detectados
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_lista = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                objetos_lista.append(f"{quantidade} {obj_pt}{'s' if quantidade > 1 else ''}")
            
            partes.append(f"Objetos: {', '.join(objetos_lista)}")
        else:
            partes.append("0 objetos específicos")
        
        return " | ".join(partes)

    def _analisar_contexto_detalhado(self, contador_objetos, total_pessoas):
        """Analisa contexto detalhado"""
        objetos_detectados = list(contador_objetos.keys())
        
        analises = []
        
        # Análise de pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if total_detectado == 1:
                analises.append("Há uma pessoa na imagem")
            elif total_detectado == 2:
                analises.append("Há duas pessoas na imagem, pode ser um encontro ou conversa")
            elif total_detectado > 2:
                analises.append(f"Há {total_detectado} pessoas, pode ser um grupo ou reunião")
        
        # Análise de ambiente
        objetos_internos = ['chair', 'couch', 'sofa', 'bed', 'table', 'desk', 'tv', 'television', 
                           'computer', 'laptop', 'monitor', 'book', 'lamp']
        objetos_externos = ['car', 'bicycle', 'tree', 'grass', 'sports ball', 'ball']
        
        count_interno = sum(1 for obj in objetos_detectados if obj in objetos_internos)
        count_externo = sum(1 for obj in objetos_detectados if obj in objetos_externos)
        
        if count_interno > count_externo:
            analises.append("O ambiente parece ser interno (casa, escritório, sala)")
        elif count_externo > count_interno:
            analises.append("O ambiente parece ser externo (rua, parque, área aberta)")
        
        # Análise de objetos específicos
        if 'chair' in objetos_detectados:
            analises.append("Tem cadeiras, pode ser um ambiente de descanso ou trabalho")
        if 'table' in objetos_detectados:
            analises.append("Tem mesas, pode ser para refeições ou trabalho")
        if 'laptop' in objetos_detectados or 'computer' in objetos_detectados:
            analises.append("Tem computadores, ambiente de trabalho ou estudo")
        
        return ". ".join(analises)

    def _corrigir_artigos_basicos(self, resposta):
        """Corrige apenas os artigos mais básicos"""
        correcoes = {
            "um bola": "uma bola",
            "um cadeira": "uma cadeira",
            "um mesa": "uma mesa",
            "um cama": "uma cama",
            "um planta": "uma planta",
            "um pessoa": "uma pessoa"
        }
        
        resposta_corrigida = resposta
        for errado, correto in correcoes.items():
            if errado in resposta_corrigida.lower():
                resposta_corrigida = re.sub(re.escape(errado), correto, resposta_corrigida, flags=re.IGNORECASE)
        
        return resposta_corrigida

    def _responder_base_melhorada(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta base muito melhorada - com respostas variadas"""
        pergunta_lower = pergunta.lower()
        
        # Dados reais
        pessoas_yolo = contador_objetos.get('person', 0)
        total_pessoas_reais = max(total_pessoas, pessoas_yolo)
        objetos_reais = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        # Lista de objetos detectados (em português)
        objetos_pt = {self._traduzir_objeto(obj): qtd for obj, qtd in objetos_reais.items()}
        
        # GERAR RESPOSTAS VARIADAS
        respostas_variadas = {
            'descrever': [
                "Na imagem, ",
                "Analisando a foto, ",
                "Vejo que ",
                "Observando a imagem, ",
                "Na foto, "
            ],
            'pessoas': [
                f"tem {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}",
                f"identifico {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}",
                f"consigo ver {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}",
                f"percebo {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}",
                f"{total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''} visível{'is' if total_pessoas_reais > 1 else ''}"
            ]
        }
        
        import random
        
        # PERGUNTA: "O que tem nessa imagem?" ou similar
        if any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva', 'o que você vê']):
            if total_pessoas_reais > 0 or objetos_pt:
                inicio = random.choice(respostas_variadas['descrever'])
                
                if total_pessoas_reais > 0:
                    pessoas_texto = random.choice(respostas_variadas['pessoas'])
                    
                    if objetos_pt:
                        # Tem pessoas e objetos
                        objetos_lista = []
                        for obj_pt, qtd in objetos_pt.items():
                            if qtd == 1:
                                artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta'] else "um"
                                objetos_lista.append(f"{artigo} {obj_pt}")
                            else:
                                objetos_lista.append(f"{qtd} {obj_pt}s")
                        
                        if len(objetos_lista) > 1:
                            objetos_texto = f"{', '.join(objetos_lista[:-1])} e {objetos_lista[-1]}"
                        else:
                            objetos_texto = objetos_lista[0]
                        
                        return f"{inicio}{pessoas_texto} e também {objetos_texto}."
                    else:
                        # Só tem pessoas
                        return f"{inicio}{pessoas_texto}."
                else:
                    # Só tem objetos
                    objetos_lista = []
                    for obj_pt, qtd in objetos_pt.items():
                        if qtd == 1:
                            artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta'] else "um"
                            objetos_lista.append(f"{artigo} {obj_pt}")
                        else:
                            objetos_lista.append(f"{qtd} {obj_pt}s")
                    
                    if len(objetos_lista) > 1:
                        objetos_texto = f"{', '.join(objetos_lista[:-1])} e {objetos_lista[-1]}"
                    else:
                        objetos_texto = objetos_lista[0]
                    
                    return f"{inicio}{objetos_texto}."
            else:
                return "Não estou identificando elementos específicos nesta imagem."
        
        # PERGUNTA: "Quantas pessoas você vê?"
        elif 'quantas pessoas' in pergunta_lower:
            if total_pessoas_reais > 0:
                respostas = [
                    f"Tem {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}.",
                    f"Vejo {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}.",
                    f"Identifico {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}.",
                    f"Na imagem, há {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}."
                ]
                return random.choice(respostas)
            else:
                return "Não tem pessoas visíveis na imagem."
        
        # PERGUNTA: "Descreva o ambiente"
        elif 'descreva o ambiente' in pergunta_lower:
            if total_pessoas_reais > 0:
                respostas = [
                    f"O ambiente tem {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}. É difícil determinar mais detalhes sem objetos específicos visíveis.",
                    f"Vejo {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''} no ambiente. Parece um espaço simples.",
                    f"O ambiente contém {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}. Não identifico muitos elementos adicionais."
                ]
                return random.choice(respostas)
            else:
                return "O ambiente parece simples, sem muitos elementos visíveis."
        
        # PERGUNTA: "Quais objetos estão visíveis?"
        elif 'quais objetos' in pergunta_lower:
            if objetos_pt:
                lista_objetos = []
                for obj_pt, qtd in objetos_pt.items():
                    if qtd == 1:
                        artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta'] else "um"
                        lista_objetos.append(f"{artigo} {obj_pt}")
                    else:
                        lista_objetos.append(f"{qtd} {obj_pt}s")
                
                if len(lista_objetos) > 1:
                    objetos_texto = f"{', '.join(lista_objetos[:-1])} e {lista_objetos[-1]}"
                else:
                    objetos_texto = lista_objetos[0]
                
                return f"Os objetos visíveis são: {objetos_texto}."
            else:
                return "Não estou identificando objetos específicos na imagem."
        
        # PERGUNTA: "Esta foto parece ser interna ou externa?"
        elif any(palavra in pergunta_lower for palavra in ['interno', 'externo', 'dentro', 'fora']):
            analise = self._analisar_interno_externo_simples(contador_objetos)
            respostas = [
                f"{analise}",
                f"Pela análise, {analise.lower()}",
                f"Baseado no que vejo, {analise.lower()}"
            ]
            return random.choice(respostas)
        
        # PERGUNTA sobre objetos específicos
        for objeto_pergunta in ['cadeira', 'sofá', 'mesa', 'cama', 'computador', 'tv', 'televisão', 
                               'celular', 'livro', 'garrafa', 'copo', 'prato', 'vaso', 'relógio',
                               'cachorro', 'gato', 'carro', 'bicicleta', 'planta', 'flor', 'árvore']:
            
            if objeto_pergunta in pergunta_lower:
                objeto_detectado = False
                quantidade = 0
                
                for obj_pt, qtd in objetos_pt.items():
                    if objeto_pergunta in obj_pt.lower() or obj_pt.lower() in objeto_pergunta:
                        objeto_detectado = True
                        quantidade = qtd
                        break
                
                if objeto_detectado:
                    if quantidade == 1:
                        artigo = "uma" if objeto_pergunta in ['cadeira', 'mesa', 'cama', 'planta', 'flor'] else "um"
                        respostas_sim = [
                            f"Sim, tem {artigo} {objeto_pergunta}.",
                            f"Sim, identifico {artigo} {objeto_pergunta}.",
                            f"Sim, vejo {artigo} {objeto_pergunta} na imagem."
                        ]
                        return random.choice(respostas_sim)
                    else:
                        return f"Sim, tem {quantidade} {objeto_pergunta}s."
                else:
                    respostas_nao = [
                        f"Não, não tem {objeto_pergunta}.",
                        f"Não estou vendo {objeto_pergunta}.",
                        f"Não identifico {objeto_pergunta} na imagem."
                    ]
                    return random.choice(respostas_nao)
        
        # PERGUNTA sobre categorias
        if 'eletrônic' in pergunta_lower:
            return self._verificar_categoria(['computador', 'tv', 'televisão', 'celular', 'monitor'], 
                                           objetos_pt, 'eletrônicos')
        
        if 'plant' in pergunta_lower or 'natureza' in pergunta_lower:
            return self._verificar_categoria(['planta', 'flor', 'árvore', 'grama'], objetos_pt, 'plantas ou natureza')
        
        if 'móve' in pergunta_lower:
            return self._verificar_categoria(['cadeira', 'sofá', 'mesa', 'cama'], objetos_pt, 'móveis')
        
        # Resposta genérica para perguntas sobre imagem não cobertas
        if total_pessoas_reais > 0:
            respostas = [
                f"Vejo {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''} na imagem.",
                f"Na foto, há {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''}.",
                f"Identifico {total_pessoas_reais} pessoa{'s' if total_pessoas_reais > 1 else ''} visível{'is' if total_pessoas_reais > 1 else ''}."
            ]
            return random.choice(respostas)
        elif objetos_pt:
            primeiro_obj = list(objetos_pt.items())[0]
            obj_nome = primeiro_obj[0]
            qtd = primeiro_obj[1]
            
            if qtd == 1:
                artigo = "uma" if obj_nome in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta'] else "um"
                return f"Vejo {artigo} {obj_nome}."
            else:
                return f"Vejo {qtd} {obj_nome}s."
        else:
            return "Não estou identificando muitos detalhes específicos no ambiente."

    def _analisar_interno_externo_simples(self, contador_objetos):
        """Análise simples de interno/externo"""
        objetos_detectados = list(contador_objetos.keys())
        
        objetos_internos = ['chair', 'couch', 'sofa', 'bed', 'table', 'desk', 'tv', 'television', 'laptop', 'computer']
        objetos_externos = ['car', 'bicycle', 'tree', 'grass', 'sports ball', 'ball']
        
        count_interno = sum(1 for obj in objetos_detectados if obj in objetos_internos)
        count_externo = sum(1 for obj in objetos_detectados if obj in objetos_externos)
        
        if count_interno > count_externo:
            return "Parece ser um ambiente interno."
        elif count_externo > count_interno:
            return "Parece ser um ambiente externo."
        else:
            return "É difícil determinar se é interno ou externo."

    def _verificar_categoria(self, objetos_categoria, objetos_pt, nome_categoria):
        """Verifica categoria"""
        import random
        
        encontrados = []
        for obj_pt, qtd in objetos_pt.items():
            if obj_pt in objetos_categoria:
                if qtd == 1:
                    artigo = "uma" if obj_pt in ['planta', 'flor', 'árvore'] else "um"
                    encontrados.append(f"{artigo} {obj_pt}")
                else:
                    encontrados.append(f"{qtd} {obj_pt}s")
        
        if encontrados:
            if len(encontrados) > 1:
                objetos_texto = f"{', '.join(encontrados[:-1])} e {encontrados[-1]}"
            else:
                objetos_texto = encontrados[0]
            
            respostas = [
                f"Sim, tem {objetos_texto}.",
                f"Sim, identifico {objetos_texto}.",
                f"Sim, vejo {objetos_texto} na imagem."
            ]
            return random.choice(respostas)
        else:
            respostas = [
                f"Não, não tem {nome_categoria}.",
                f"Não estou vendo {nome_categoria}.",
                f"Não identifico {nome_categoria} na imagem."
            ]
            return random.choice(respostas)

    # ========== MÉTODO PARA PERGUNTAS GERAIS CORRIGIDO ==========

    def _responder_pergunta_geral_corrigida(self, pergunta):
        """Responde perguntas gerais de forma muito melhor"""
        if not self.client:
            return self._responder_pergunta_geral_local(pergunta)
        
        pergunta_lower = pergunta.lower()
        
        # RESPOSTAS PRÉ-DEFINIDAS PARA PERGUNTAS COMUNS
        respostas_pre_definidas = {
            # Temperatura
            'qual a temperatura': "🌡️ No momento não tenho acesso a informações meteorológicas em tempo real. Mas posso te ajudar analisando imagens do ambiente ao seu redor! Tem alguma foto para eu ver?",
            'temperatura atual': "🌡️ Não consigo verificar a temperatura agora, mas posso analisar imagens para te ajudar a entender melhor seu ambiente!",
            'como está o tempo': "⛅ Infelizmente não tenho acesso a dados meteorológicos. Mas se você me enviar uma imagem, posso descrever o que está vendo!",
            'faz calor': "🔥 Não sei dizer exatamente, mas posso te ajudar analisando imagens do ambiente. Que tal me enviar uma foto?",
            
            # Localização
            'onde estamos': "📍 Não tenho acesso à localização GPS, mas se você me enviar uma imagem, posso tentar descrever o ambiente e ajudar você a entender onde está!",
            'qual cidade': "🏙️ Não consigo determinar a cidade, mas posso analisar imagens para te ajudar a explorar o ambiente!",
            
            # Assistente
            'quem é você': "👋 Eu sou a Specula! 😊 Uma assistente criada para ajudar pessoas com deficiência visual a entender melhor o ambiente ao seu redor através da análise de imagens.",
            'qual seu nome': "😊 Meu nome é Specula! Sou sua assistente visual, pronta para te ajudar a explorar o mundo através das imagens.",
            'o que você faz': "👁️ Posso analisar imagens que você enviar, descrever o que tem nelas, identificar objetos e pessoas, e responder suas perguntas sobre o ambiente. É como ter olhos digitais para te ajudar a ver o mundo!",
            
            # Agradecimentos
            'obrigado': "✨ Por nada! Fico feliz em poder ajudar. Sou a Specula, sempre à disposição!",
            'valeu': "😊 De nada! Estou aqui para te ajudar. Sou a Specula!",
            
            # Cumprimentos
            'oi': self._gerar_cumprimento(),
            'olá': self._gerar_cumprimento(),
            'bom dia': self._gerar_cumprimento(),
            'boa tarde': self._gerar_cumprimento(),
            'boa noite': self._gerar_cumprimento(),
            
            # Bem-estar
            'como você está': "😊 Estou muito bem, obrigada! Pronta para te ajudar a explorar o mundo através das imagens. E você, como está se sentindo?",
            'tudo bem': "👍 Tudo ótimo por aqui! Pronta para te ajudar com análise de imagens. E com você, tudo bem?",
            
            # Conceitos
            'oque é batata': "🍠 A batata é um tubérculo comestível, originário da América do Sul. É um alimento básico em muitas culturas e pode ser preparado de diversas formas: cozida, frita, assada ou purê. Rico em carboidratos, vitaminas e minerais!",
            'o que é batata': "🍠 A batata é um tubérculo comestível super versátil! Pode ser frita (batata frita), cozida, assada, ou em purê. É um alimento muito popular no mundo todo!",
            
            # Funcionalidade
            'como pode ajudar': "🤔 Posso analisar qualquer imagem que você enviar! Me mande uma foto e eu descrevo o que tem nela, identifico objetos e pessoas, e respondo suas perguntas sobre o ambiente.",
            
            # IA/Robô
            'você é real': "🤖 Sou uma inteligência artificial criada especialmente para ajudar pessoas com deficiência visual. Mas gosto de pensar que sou uma amiga digital que está aqui para te apoiar! 💖",
            'é uma ia': "💻 Sim! Sou uma inteligência artificial chamada Specula, criada para ser seus olhos digitais e te ajudar a entender melhor o mundo ao seu redor.",
            
            # Eletrônicos (pergunta geral, não sobre imagem)
            'você identifica algum eletrônico': "💡 Quando você me envia uma imagem, sim! Consigo identificar eletrônicos como computadores, celulares, TVs e muito mais. Quer testar? Me envie uma foto!",
            'identifica eletrônico': "📱 Sim, quando analiso imagens, consigo identificar vários tipos de eletrônicos. Me envie uma foto com eletrônicos para eu mostrar como funciona!"
        }
        
        # Verificar respostas pré-definidas
        for chave, resposta in respostas_pre_definidas.items():
            if chave in pergunta_lower:
                return resposta
        
        # Se não encontrou resposta pré-definida, usar IA
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Você é a Specula, uma assistente amigável e útil para pessoas com deficiência visual.

                    SEU ESTILO:
                    - Fale de forma natural e acolhedora
                    - Use emojis ocasionalmente 😊
                    - Seja positiva e encorajadora
                    - Se não souber algo, seja honesta
                    - Ofereça ajudar com imagens quando possível
                    
                    SUA ESPECIALIDADE:
                    - Análise de imagens para pessoas com deficiência visual
                    - Descrição de ambientes
                    - Identificação de objetos e pessoas
                    
                    QUANDO NÃO SOUBER:
                    - Admita honestamente
                    - Sugira que pode ajudar com imagens
                    - Mantenha-se útil e positiva"""
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
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return self._responder_pergunta_geral_local(pergunta)

    def _gerar_cumprimento(self):
        """Gera cumprimento baseado no horário"""
        try:
            brasilia_tz = timezone(timedelta(hours=-3))
            agora = datetime.now(brasilia_tz)
            hora = agora.hour
        except:
            agora = datetime.now()
            hora = agora.hour
        
        if 5 <= hora < 12:
            return "☀️ Bom dia! Eu sou a Specula, sua assistente visual. Como posso te ajudar hoje?"
        elif 12 <= hora < 18:
            return "🌤️ Boa tarde! Eu sou a Specula, pronta para te ajudar com análise de imagens. O que precisa?"
        else:
            return "🌙 Boa noite! Sou a Specula, sua assistente visual. Como posso te ajudar nesta noite?"

    def _responder_pergunta_geral_local(self, pergunta):
        """Resposta local para perguntas gerais"""
        pergunta_lower = pergunta.lower()
        
        if 'batata' in pergunta_lower:
            return "🍠 A batata é um tubérculo comestível muito versátil! Pode ser frita, cozida, assada... É um alimento básico em muitas culturas!"
        
        if any(palavra in pergunta_lower for palavra in ['temperatura', 'calor', 'frio']):
            return "🌡️ Não tenho acesso a informações de temperatura, mas posso te ajudar analisando imagens do ambiente!"
        
        if any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde']):
            return self._gerar_cumprimento()
        
        if 'quem é você' in pergunta_lower:
            return "👋 Eu sou a Specula, sua assistente visual! Posso analisar imagens para te ajudar a entender melhor o ambiente."
        
        return "Olá! Sou a Specula, sua assistente visual. Posso te ajudar analisando imagens e descrevendo o que tem nelas. Tem alguma foto para eu ver?"

    # ========== MÉTODOS AUXILIARES ==========

    # ... (MANTENHA TODOS OS OUTROS MÉTODOS AUXILIARES DO SEU CÓDIGO ANTERIOR) ...
    # _verificar_e_corrigir_invencoes, _construir_contexto_real, _gerar_descricao_precisa,
    # _traduzir_objeto, _filtrar_objetos_relevantes, _formatar_dados_utilizados,
    # obter_estatisticas, e todos os métodos de estatísticas

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
        
        # Correções básicas de artigos
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
            return self._gerar_descricao_precisa(contador_objetos, total_pessoas, faces_conhecidas)
        
        return descricao

    def _construir_contexto_real(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto apenas com dados reais"""
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

    def _gerar_descricao_precisa(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição precisa baseada apenas nos dados"""
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
                elif total_detectado == 2:
                    partes.append("duas pessoas")
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