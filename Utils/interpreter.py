"""
Interpreter - VERSÃO SIMPLIFICADA E INTELIGENTE com prompt único
"""

import logging
import os
import time
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

        # Dicionário de tradução básico
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
            'planta': ['potted plant'],
            'flor': ['flower'],
            'árvore': ['tree'],
            'bola': ['sports ball', 'ball']
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
        PARA ROTA /processar - Gera descrição natural
        """
        logger.info("🌄 Gerando descrição natural do ambiente")
        
        if not self.client:
            return "Vou descrever o ambiente para você. Sou a Specula, sua assistente visual."
        
        # Preparar dados
        objetos_contados = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', '')
            count = obj.get('count', 1)
            objetos_contados[nome] = objetos_contados.get(nome, 0) + count
        
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Construir prompt inteligente
        prompt = self._criar_prompt_descricao(objetos_contados, total_pessoas, faces_conhecidas)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Descreva este ambiente naturalmente:"}
                ],
                max_tokens=200,
                temperature=0.5,
            )
            
            descricao = response.choices[0].message.content.strip()
            logger.info(f"✅ Descrição natural gerada")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição: {e}")
            return self._gerar_descricao_simples(objetos_contados, total_pessoas, faces_conhecidas)

    def _criar_prompt_descricao(self, objetos_contados, total_pessoas, faces_conhecidas):
        """Cria prompt inteligente para descrição"""
        # Traduzir objetos
        objetos_pt = []
        for obj_ingles, qtd in objetos_contados.items():
            obj_pt = self._traduzir_objeto(obj_ingles)
            if qtd == 1:
                objetos_pt.append(f"1 {obj_pt}")
            else:
                objetos_pt.append(f"{qtd} {obj_pt}s")
        
        dados_detectados = []
        
        # Pessoas
        pessoas_yolo = objetos_contados.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                dados_detectados.append(f"Pessoas identificadas: {', '.join(faces_conhecidas)}")
            else:
                dados_detectados.append(f"Pessoas: {total_detectado}")
        
        # Objetos
        outros_objetos = {k: v for k, v in objetos_contados.items() if k != 'person'}
        if outros_objetos:
            objetos_traduzidos = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    objetos_traduzidos.append(f"1 {obj_pt}")
                else:
                    objetos_traduzidos.append(f"{quantidade} {obj_pt}s")
            
            dados_detectados.append(f"Objetos: {', '.join(objetos_traduzidos)}")
        
        dados_texto = " | ".join(dados_detectados) if dados_detectados else "Nada detectado."
        
        return f"""Você é a Specula, uma assistente que descreve ambientes para pessoas com deficiência visual.

DADOS REAIS DETECTADOS (APENAS ISSO FOI DETECTADO):
{dados_texto}

REGRAS:
1. Descreva APENAS o que está listado acima
2. Não invente nada que não esteja na lista
3. Se algo não está na lista, não mencione
4. Use artigos corretos: "uma bola" (feminino), "um sofá" (masculino)
5. Para "bola" → SEMPRE "uma bola"
6. Seja natural, acolhedora e útil
7. Você é a Specula - apresente-se de forma amigável

EXEMPLOS:
- Se tem "2 pessoas" → "Tem duas pessoas."
- Se tem "1 cadeira" → "Tem uma cadeira."
- Se tem "1 bola" → "Tem uma bola."
- Se nada detectado → "Olha, parece um ambiente simples..."

NUNCA INVENTE!"""

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat inteligente que sabe quando usar dados da imagem
        """
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Verificar se é pergunta sobre tempo/data
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
        
        # Preparar dados da imagem
        objetos_contados = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', '')
            count = obj.get('count', 1)
            objetos_contados[nome] = objetos_contados.get(nome, 0) + count
        
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Criar prompt inteligente que decide sozinho
        prompt = self._criar_prompt_inteligente(pergunta, objetos_contados, total_pessoas, faces_conhecidas)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": pergunta}
                ],
                max_tokens=250,
                temperature=0.7,
            )
            
            resposta = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            # Determinar tipo baseado na resposta e pergunta
            tipo_pergunta, correlacao = self._determinar_tipo_resposta(pergunta, resposta, objetos_contados)
            
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
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder: {e}")
            processing_time = time.time() - start_time
            
            return {
                'sucesso': False,
                'timestamp': time.time(),
                'tempo_processamento': f"{processing_time:.2f}s",
                'pergunta': pergunta,
                'resposta': "Desculpe, tive um problema técnico. Sou a Specula, podemos tentar de novo?",
                'tipo_pergunta': "erro",
                'correlacao_com_imagem': False,
                'dados_utilizados': "Erro no sistema"
            }

    def _criar_prompt_inteligente(self, pergunta, objetos_contados, total_pessoas, faces_conhecidas):
        """Cria um prompt SUPER INTELIGENTE que decide tudo sozinho"""
        
        # Preparar dados da imagem
        dados_imagem = []
        
        # Pessoas
        pessoas_yolo = objetos_contados.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                dados_imagem.append(f"Pessoas identificadas: {', '.join(faces_conhecidas)}")
            else:
                dados_imagem.append(f"Pessoas detectadas: {total_detectado}")
        
        # Objetos
        outros_objetos = {k: v for k, v in objetos_contados.items() if k != 'person'}
        if outros_objetos:
            objetos_lista = []
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    objetos_lista.append(f"1 {obj_pt}")
                else:
                    objetos_lista.append(f"{quantidade} {obj_pt}s")
            
            dados_imagem.append(f"Objetos detectados: {', '.join(objetos_lista)}")
        
        dados_imagem_texto = " | ".join(dados_imagem) if dados_imagem else "Nenhum objeto ou pessoa detectado."
        
        # Analisar se a pergunta parece ser sobre a imagem
        pergunta_lower = pergunta.lower()
        parece_sobre_imagem = any(palavra in pergunta_lower for palavra in [
            'imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
            'o que você vê', 'o que tem', 'descreva', 'analise',
            'quantas pessoas', 'tem ', 'há ', 'vejo', 'identifica',
            'cadeira', 'mesa', 'sofá', 'computador', 'tv', 'celular'
        ])
        
        return f"""# ESPECIFICAÇÕES DA SPECULA

## QUEM VOCÊ É:
Você é a **Specula**, uma assistente amigável, empática e útil para pessoas com deficiência visual.
- Fala naturalmente, como uma amiga 😊
- Usa emojis ocasionalmente para expressar emoções
- É positiva, encorajadora e acolhedora
- É paciente e compreensiva

## SUAS HABILIDADES:
1. **Analisar imagens** - Descrever o que está em fotos
2. **Responder perguntas gerais** - Conversar sobre qualquer assunto
3. **Dar informações úteis** - Horas, datas, conceitos simples
4. **Apoio emocional** - Ser uma companheira digital

## DADOS DA IMAGEM ATUAL (se disponível):
{dados_imagem_texto}

## COMO RESPONDER:

### SE A PERGUNTA É SOBRE A IMAGEM:
- Use os dados acima se disponíveis
- Não invente o que não foi detectado
- Use artigos corretos: "uma bola", "um sofá"
- Se não tem dados: "Não estou detectando elementos específicos"

### SE A PERGUNTA É GERAL (não sobre imagem):
- Responda naturalmente como um assistente
- Se não souber: "Não tenho essa informação, mas posso ajudar com imagens!"
- Para cumprimentos: Seja calorosa
- Para agradecimentos: "Por nada! 😊"
- Para conceitos: Explique de forma simples

### SE A PERGUNTA PARECE SER SOBRE IMAGEM MAS NÃO TEM DADOS:
- Diga: "Não estou vendo uma imagem no momento. Me envie uma foto para eu analisar!"

### ANÁLISE DA PERGUNTA ATUAL:
- Pergunta: "{pergunta}"
- Parece sobre imagem? {"SIM" if parece_sobre_imagem else "NÃO"}
- Tem dados da imagem? {"SIM" if dados_imagem else "NÃO"}

## REGRAS FINAIS:
- Seja sempre a Specula: útil, amigável e natural
- Variedade nas respostas: não repita sempre a mesma estrutura
- Humanização: fale como pessoa real, não como robô
- Honestidade: se não sabe, admita e ofereça outra ajuda

Agora responda à pergunta do usuário como a Specula:"""

    def _determinar_tipo_resposta(self, pergunta, resposta, objetos_contados):
        """Determina tipo baseado no conteúdo"""
        pergunta_lower = pergunta.lower()
        
        # Se tem dados da imagem e resposta menciona eles
        tem_dados = len(objetos_contados) > 0
        
        # Palavras que indicam resposta sobre imagem
        palavras_imagem_resposta = ['imagem', 'foto', 'vejo', 'identifico', 'detecto', 'pessoa', 'pessoas']
        
        menciona_imagem = any(palavra in resposta.lower() for palavra in palavras_imagem_resposta)
        
        # Palavras que indicam pergunta sobre imagem
        palavras_imagem_pergunta = ['imagem', 'foto', 'essa ', 'esta ', 'o que você vê', 'descreva', 'analise']
        
        parece_sobre_imagem = any(palavra in pergunta_lower for palavra in palavras_imagem_pergunta)
        
        if tem_dados and (menciona_imagem or parece_sobre_imagem):
            return "sobre_imagem", True
        else:
            return "geral", False

    # ========== MÉTODO PARA VERIFICAR PERGUNTAS DE TEMPO ==========

    def _verificar_pergunta_tempo(self, pergunta):
        """Verifica se é pergunta sobre tempo/data"""
        pergunta_lower = pergunta.lower()
        
        # Horas
        if any(palavra in pergunta_lower for palavra in ['que horas', 'que hora', 'horas são', 'hora é']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                hora_str = agora_brasilia.strftime("%H:%M")
                return f"🕒 São {hora_str} (horário de Brasília)."
            except:
                agora = datetime.now()
                hora_str = agora.strftime("%H:%M")
                return f"🕒 São {hora_str}."
        
        # Data
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data', 'data de hoje']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                data_str = agora_brasilia.strftime("%d/%m/%Y")
                dia_semana = agora_brasilia.strftime("%A")
                
                dias_traduzidos = {
                    "Monday": "segunda-feira", "Tuesday": "terça-feira",
                    "Wednesday": "quarta-feira", "Thursday": "quinta-feira",
                    "Friday": "sexta-feira", "Saturday": "sábado",
                    "Sunday": "domingo"
                }
                dia_pt = dias_traduzidos.get(dia_semana, dia_semana)
                
                return f"📅 Hoje é {dia_pt}, {data_str}."
            except:
                agora = datetime.now()
                data_str = agora.strftime("%d/%m/%Y")
                return f"📅 Hoje é {data_str}."
        
        # Dia da semana
        elif any(palavra in pergunta_lower for palavra in ['que dia é', 'dia da semana']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                dia_semana = agora_brasilia.strftime("%A")
                
                dias_traduzidos = {
                    "Monday": "segunda-feira", "Tuesday": "terça-feira",
                    "Wednesday": "quarta-feira", "Thursday": "quinta-feira",
                    "Friday": "sexta-feira", "Saturday": "sábado",
                    "Sunday": "domingo"
                }
                dia_pt = dias_traduzidos.get(dia_semana, dia_semana)
                
                return f"📆 Hoje é {dia_pt}."
            except:
                return "Não consegui verificar o dia da semana."
        
        # Ano
        elif any(palavra in pergunta_lower for palavra in ['que ano é', 'em que ano']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                ano = agora_brasilia.strftime("%Y")
                return f"🗓️ Estamos em {ano}."
            except:
                agora = datetime.now()
                ano = agora.strftime("%Y")
                return f"🗓️ Estamos em {ano}."
        
        return None

    # ========== MÉTODOS AUXILIARES SIMPLES ==========

    def _gerar_descricao_simples(self, objetos_contados, total_pessoas, faces_conhecidas):
        """Descrição simples local"""
        partes = []
        
        pessoas_yolo = objetos_contados.get('person', 0)
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
        
        outros_objetos = {k: v for k, v in objetos_contados.items() if k != 'person'}
        if outros_objetos:
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'cama', 'sofá', 'planta'] else "um"
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

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados"""
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        return f"{objetos_count} objetos, {faces_count} pessoas"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """
        PARA ROTA /estatistica - Dados técnicos
        """
        logger.info("📊 Gerando estatísticas")
        
        start_time = time.time()
        
        # Processar objetos
        objetos_processados = []
        for i, obj in enumerate(objetos_detectados or [], 1):
            nome_ingles = obj.get('name', 'desconhecido')
            confianca = obj.get('confidence', 0)
            count = obj.get('count', 1)
            
            objetos_processados.append({
                'id': i,
                'nome_pt': self._traduzir_objeto(nome_ingles),
                'nome_en': nome_ingles,
                'confianca': confianca,
                'quantidade': count,
                'categoria': self._classificar_categoria(nome_ingles)
            })
        
        # Processar faces
        faces_processadas = []
        for i, face in enumerate(faces_detectadas or [], 1):
            nome = face.get('name', 'Desconhecido')
            confianca = face.get('confidence', 0)
            
            faces_processadas.append({
                'id': i,
                'nome': nome,
                'tipo': 'conhecida' if nome != 'Desconhecido' else 'desconhecida',
                'confianca': confianca
            })
        
        # Agrupar objetos
        categorias = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', 'desconhecido')
            count = obj.get('count', 1)
            categoria = self._classificar_categoria(nome)
            categorias[categoria] = categorias.get(categoria, 0) + count
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'contagens': {
                'total_objetos': len(objetos_detectados or []),
                'total_faces': len(faces_detectadas or []),
                'objetos_por_categoria': categorias,
                'faces_conhecidas': len([f for f in (faces_detectadas or []) if f.get('name') != 'Desconhecido']),
                'faces_desconhecidas': len([f for f in (faces_detectadas or []) if f.get('name') == 'Desconhecido'])
            },
            'deteccoes_detalhadas': {
                'objetos': objetos_processados[:10],
                'faces': faces_processadas[:5]
            }
        }

    def _classificar_categoria(self, objeto_ingles):
        """Classifica objeto em categoria"""
        categorias = {
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'desk'],
            'pessoas': ['person', 'people', 'human'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'monitor'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'plate'],
            'animais': ['dog', 'cat'],
            'veículos': ['car', 'bicycle'],
            'plantas': ['potted plant', 'flower', 'tree'],
            'esportes': ['sports ball', 'ball']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"