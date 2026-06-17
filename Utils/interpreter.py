"""
Interpreter
"""

import logging
import os
import time
import numpy as np
from datetime import datetime, timezone, timedelta

from Utils.glm_client import chat as glm_chat, glm_disponivel as glm_api_disponivel

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or os.getenv('GLM_MODEL', 'glm-5')
        self.glm_disponivel = False
        
        logger.info("🔄 Inicializando Interpreter...")
        
        if glm_api_disponivel():
            logger.info("📦 ZAI API key encontrada")
            try:
                logger.info("🧪 Testando conexão GLM...")
                glm_chat(
                    messages=[{"role": "user", "content": "Teste"}],
                    model=self.model_name,
                    max_tokens=5,
                )
                self.glm_disponivel = True
                logger.info("✅ GLM conectado")
            except Exception as e:
                logger.error(f"❌ Erro GLM: {e}")
                self.glm_disponivel = False
        else:
            logger.warning("⚠️ Sem ZAI_API_KEY válida - modo local")
            self.glm_disponivel = False
        
        # Dicionário de tradução
        self.traducoes = {
            'person': 'pessoa',
            'chair': 'cadeira', 
            'table': 'mesa',
            'laptop': 'computador',
            'tv': 'televisão',
            'cell phone': 'celular',
            'book': 'livro',
            'bottle': 'garrafa',
            'cup': 'copo',
            'car': 'carro',
            'bicycle': 'bicicleta',
            'dog': 'cachorro',
            'cat': 'gato',
            'potted plant': 'planta',
            'clock': 'relógio',
            'vase': 'vaso',
            'tie': 'gravata',
            'truck': 'caminhão',
            'backpack': 'mochila',
            'handbag': 'bolsa',
            'remote': 'controle remoto',
            'keyboard': 'teclado',
            'mouse': 'mouse',
            'dining table': 'mesa de jantar',
            'couch': 'sofá',
            'bed': 'cama',
            'toilet': 'vaso sanitário',
            'refrigerator': 'geladeira',
            'microwave': 'microondas',
            'oven': 'forno',
            'sink': 'pia',
            'sports ball': 'bola esportiva',
            'frisbee': 'frisbee',
            'skateboard': 'skate',
            'surfboard': 'prancha de surf',
            'tennis racket': 'raquete de tênis',
            'wine glass': 'taça de vinho',
            'fork': 'garfo',
            'knife': 'faca',
            'spoon': 'colher',
            'bowl': 'tigela',
            'banana': 'banana',
            'apple': 'maçã',
            'sandwich': 'sanduíche',
            'orange': 'laranja',
            'pizza': 'pizza',
            'donut': 'rosquinha',
            'cake': 'bolo'
        }

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """Gera descrição natural - compatível com main.py v5"""
        logger.info("🌄 Gerando descrição natural")
        
        # Preparar dados no formato correto
        dados_texto = self._formatar_dados_compativel(objetos_detectados)
        
        if self.glm_disponivel:
            try:
                return glm_chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Você é a Specula, assistente amigável para deficientes visuais.

DADOS DETECTADOS:
{dados_texto}

INSTRUÇÕES:
- Baseie-se APENAS nos dados acima
- Seja natural e amigável
- Traduza nomes para português
- Não invente posições ou quantidades
- Usuário é deficiente visual
- NUNCA peça para descrever algo"""
                        },
                        {"role": "user", "content": "Descreva o que vê"}
                    ],
                    max_tokens=120,
                    temperature=0.6,
                )
                
            except Exception as e:
                logger.error(f"❌ Erro GLM: {e}")
                return self._descricao_local(objetos_detectados)
        else:
            return self._descricao_local(objetos_detectados)

    def _descricao_local(self, objetos_detectados):
        """Descrição local de fallback"""
        if not objetos_detectados:
            return "Olá! Sou a Specula. Não estou identificando objetos claramente."
        
        # Contar objetos únicos
        contagem = {}
        for obj in objetos_detectados:
            nome = self._obter_nome_portugues(obj)
            contagem[nome] = contagem.get(nome, 0) + obj.get('quantidade', 1)
        
        # Construir descrição
        itens = []
        for nome, qtd in contagem.items():
            if qtd == 1:
                itens.append(f"1 {nome}")
            else:
                itens.append(f"{qtd} {nome}s")
        
        if itens:
            return f"Olá! Vejo: {', '.join(itens)}."
        else:
            return "Olá! Sou a Specula."

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas - RETORNO COMPATÍVEL com main.py v5"""
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        inicio_tempo = time.time()
        
        # Verificar perguntas de tempo/data
        resposta_tempo = self._verificar_pergunta_tempo_data(pergunta)
        if resposta_tempo:
            tempo_ms = int((time.time() - inicio_tempo) * 1000)
            return {
                'sucesso': True,
                'timestamp': int(time.time() * 1000),  # millis
                'tempo_processamento_ms': tempo_ms,    # ms
                'dados': {
                    'pergunta': pergunta,
                    'resposta': resposta_tempo,
                    'correlacao_com_imagem': False,
                    'confianca_resposta': 1.0
                }
            }
        
        # Preparar dados
        dados_texto = self._formatar_dados_compativel(objetos_detectados)
        
        if self.glm_disponivel:
            try:
                # Analisar se é sobre imagem
                pergunta_lower = pergunta.lower()
                sobre_imagem = any(palavra in pergunta_lower for palavra in [
                    'imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
                    'o que você vê', 'o que tem', 'descreva', 'analise',
                    'quantas pessoas', 'tem ', 'há ', 'vejo', 'identifica',
                    'quantos ', 'tem alguma', 'há alguma', 'o que está'
                ])
                
                prompt = f"""Você é a Specula, assistente para deficientes visuais.

DADOS DA IMAGEM (APENAS ESTES):
{dados_texto}

PERGUNTA: "{pergunta}"

REGRAS:
1. Baseie-se APENAS nos dados acima para perguntas sobre imagem
2. Não invente objetos, posições ou quantidades
3. Usuário é deficiente visual - NUNCA peça para descrever
4. Seja natural e amigável
5. Para perguntas gerais, responda normalmente
6. Se não detecta algo: "Não estou detectando isso"

RESPONDA:"""
                
                resposta = glm_chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": pergunta}
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )
                tempo_ms = int((time.time() - inicio_tempo) * 1000)
                
                # Detecções relevantes (top 3 por confiança)
                deteccoes_relevantes = []
                if objetos_detectados and sobre_imagem:
                    # Ordenar por confiança
                    objetos_ordenados = sorted(objetos_detectados, 
                                              key=lambda x: x.get('confianca', 0), 
                                              reverse=True)[:3]
                    
                    for obj in objetos_ordenados:
                        deteccoes_relevantes.append({
                            'nome': obj.get('nome', 'desconhecido'),
                            'confianca': obj.get('confianca', 0)
                        })
                
                return {
                    'sucesso': True,
                    'timestamp': int(time.time() * 1000),
                    'tempo_processamento_ms': tempo_ms,
                    'dados': {
                        'pergunta': pergunta,
                        'resposta': resposta,
                        'deteccoes_relevantes': deteccoes_relevantes,
                        'correlacao_com_imagem': sobre_imagem,
                        'confianca_resposta': 0.8 if sobre_imagem else 1.0,
                        'total_deteccoes': len(objetos_detectados or [])
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Erro GLM: {e}")
                return self._resposta_local(pergunta, objetos_detectados, inicio_tempo)
        else:
            return self._resposta_local(pergunta, objetos_detectados, inicio_tempo)

    def _verificar_pergunta_tempo_data(self, pergunta):
        """Verifica perguntas sobre tempo/data"""
        pergunta_lower = pergunta.lower()
        
        if any(palavra in pergunta_lower for palavra in ['que horas', 'que hora', 'horas são', 'hora é']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora = datetime.now(brasilia_tz)
                hora = agora.strftime("%H:%M")
                return f"🕒 São {hora} (horário de Brasília)."
            except:
                agora = datetime.now()
                hora = agora.strftime("%H:%M")
                return f"🕒 São {hora}."
        
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data', 'data de hoje']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora = datetime.now(brasilia_tz)
                data = agora.strftime("%d/%m/%Y")
                return f"📅 Hoje é {data}."
            except:
                agora = datetime.now()
                data = agora.strftime("%d/%m/%Y")
                return f"📅 Hoje é {data}."
        
        return None

    def _resposta_local(self, pergunta, objetos_detectados, inicio_tempo):
        """Resposta local de fallback"""
        pergunta_lower = pergunta.lower()
        
        # Respostas básicas
        if 'quem é você' in pergunta_lower or 'seu nome' in pergunta_lower:
            resposta = "👋 Eu sou a Specula, sua assistente visual!"
        elif 'oi' in pergunta_lower or 'olá' in pergunta_lower:
            hora = datetime.now().hour
            if 5 <= hora < 12:
                resposta = "☀️ Bom dia! Sou a Specula."
            elif 12 <= hora < 18:
                resposta = "🌤️ Boa tarde! Sou a Specula."
            else:
                resposta = "🌙 Boa noite! Sou a Specula."
        elif 'batata' in pergunta_lower:
            resposta = "🍠 A batata é um tubérculo comestível muito versátil!"
        elif objetos_detectados and any(p in pergunta_lower for p in ['o que tem', 'o que vê', 'quantas', 'quantos']):
            # Tentar responder baseado nos objetos
            pessoas = sum(obj.get('quantidade', 1) for obj in objetos_detectados 
                         if obj.get('nome', '').lower() == 'person')
            
            if 'pessoa' in pergunta_lower or 'pessoas' in pergunta_lower:
                if pessoas == 0:
                    resposta = "Não estou detectando pessoas."
                elif pessoas == 1:
                    resposta = "Vejo 1 pessoa."
                else:
                    resposta = f"Vejo {pessoas} pessoas."
            else:
                resposta = "Vejo alguns objetos na imagem."
        else:
            resposta = "Olá! Sou a Specula. Como posso ajudar?"
        
        tempo_ms = int((time.time() - inicio_tempo) * 1000)
        
        return {
            'sucesso': True,
            'timestamp': int(time.time() * 1000),
            'tempo_processamento_ms': tempo_ms,
            'dados': {
                'pergunta': pergunta,
                'resposta': resposta,
                'deteccoes_relevantes': [],
                'correlacao_com_imagem': False,
                'confianca_resposta': 0.5
            }
        }

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados=None, faces_detectadas=None):
        """Estatísticas - RETORNO COMPATÍVEL com main.py v5"""
        logger.info("📊 Gerando estatísticas")
        inicio_tempo = time.time()
        
        # Processar objetos
        objetos_info = []
        contagem_tipos = {}
        confiancas = []
        
        for obj in (objetos_detectados or []):
            nome_ingles = obj.get('nome', '')
            nome_pt = self._obter_nome_portugues(obj)
            confianca = obj.get('confianca', 0)
            quantidade = obj.get('quantidade', 1)
            
            # Adicionar à lista de objetos
            objetos_info.append({
                'nome': nome_pt,
                'confianca': confianca,
                'quantidade': quantidade
            })
            
            # Contar por tipo
            contagem_tipos[nome_pt] = contagem_tipos.get(nome_pt, 0) + quantidade
            
            # Coletar confianças
            confiancas.append(confianca)
        
        # Calcular estatísticas de confiança
        estatisticas_confianca = {
            'media': 0,
            'maxima': 0,
            'minima': 0,
            'mediana': 0
        }
        
        if confiancas:
            estatisticas_confianca = {
                'media': round(float(np.mean(confiancas)), 3),
                'maxima': round(float(np.max(confiancas)), 3),
                'minima': round(float(np.min(confiancas)), 3),
                'mediana': round(float(np.median(confiancas)), 3)
            }
        
        tempo_ms = int((time.time() - inicio_tempo) * 1000)
        
        return {
            'sucesso': True,
            'timestamp': int(time.time() * 1000),
            'tempo_processamento_ms': tempo_ms,
            'dados': {
                'resumo': {
                    'total_objetos': len(objetos_detectados or []),
                    'objetos_unicos': len(contagem_tipos)
                },
                'contagem_objetos': contagem_tipos,
                'estatisticas_confianca': estatisticas_confianca,
                'amostra_deteccoes': objetos_info[:5]
            }
        }

    # ========== MÉTODOS AUXILIARES ==========

    def _formatar_dados_compativel(self, objetos_detectados):
        """Formata dados no formato correto para GPT"""
        if not objetos_detectados:
            return "Nenhum objeto detectado."
        
        # Agrupar por tipo
        grupos = {}
        for obj in objetos_detectados:
            nome_pt = self._obter_nome_portugues(obj)
            quantidade = obj.get('quantidade', 1)
            grupos[nome_pt] = grupos.get(nome_pt, 0) + quantidade
        
        # Construir texto
        itens = []
        for nome, qtd in grupos.items():
            if qtd == 1:
                itens.append(f"1 {nome}")
            else:
                itens.append(f"{qtd} {nome}s")
        
        return "Detectado: " + ", ".join(itens)

    def _obter_nome_portugues(self, objeto):
        """Obtém nome em português do objeto"""
        nome_ingles = objeto.get('nome', '')
        if nome_ingles in self.traducoes:
            return self.traducoes[nome_ingles]
        return nome_ingles

    def _classificar_categoria(self, nome_ingles):
        """Classifica objeto em categoria"""
        categorias = {
            'móveis': ['chair', 'table', 'couch', 'bed', 'sofa'],
            'eletrônicos': ['laptop', 'tv', 'cell phone', 'monitor'],
            'pessoas': ['person'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock'],
            'veículos': ['car', 'bicycle', 'truck'],
            'animais': ['dog', 'cat'],
            'alimentos': ['banana', 'apple', 'orange', 'pizza', 'cake'],
            'plantas': ['potted plant']
        }
        
        for categoria, itens in categorias.items():
            if nome_ingles in itens:
                return categoria
        return 'outros'