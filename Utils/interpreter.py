"""
Interpreter - VERSÃO INTELIGENTE E NATURAL
Apenas GPT-4o-mini com instruções claras, sem respostas pré-definidas
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        self.model_name = model_name
        self.openai_available = False
        
        # LOG DE INICIALIZAÇÃO
        logger.info("=" * 60)
        logger.info("🔧 INICIANDO INTERPRETER - VERSÃO INTELIGENTE")
        logger.info("=" * 60)
        
        # Verificar API key
        logger.info(f"📦 API key recebida: {'✅ SIM' if api_key else '❌ NÃO'}")
        
        if api_key and isinstance(api_key, str) and api_key.strip():
            logger.info(f"📦 Tamanho da API key: {len(api_key)} caracteres")
            logger.info(f"📦 Prefixo: {api_key[:8]}...")
            
            if api_key.startswith('sk-'):
                logger.info("✅ Formato OpenAI válido (sk-...)")
            else:
                logger.warning("⚠️ API key não começa com 'sk-', tentando mesmo assim")
            
            # TENTAR IMPORTAR E CONFIGURAR OPENAI
            try:
                logger.info("🔄 Importando biblioteca OpenAI...")
                import openai
                
                # Configurar API key (forma antiga da 0.28.1)
                openai.api_key = api_key
                
                # TESTAR A CONEXÃO
                logger.info("🧪 Testando conexão com OpenAI...")
                try:
                    # Teste simples
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Teste"}],
                        max_tokens=5
                    )
                    
                    logger.info(f"✅ OpenAI CONECTADO COM SUCESSO!")
                    logger.info(f"✅ Modelo: {self.model_name}")
                    self.openai_available = True
                    
                except Exception as test_e:
                    logger.error(f"❌ Teste de conexão falhou: {test_e}")
                    self.openai_available = False
                    
            except ImportError as e:
                logger.error(f"❌ Biblioteca 'openai' não instalada: {e}")
                self.openai_available = False
            except Exception as e:
                logger.error(f"❌ Erro ao configurar OpenAI: {str(e)[:200]}")
                self.openai_available = False
        else:
            logger.warning("⚠️ Nenhuma API key válida recebida")
            logger.warning("⚠️ Usando MODO LOCAL SIMPLES")
            self.openai_available = False
        
        logger.info(f"🎯 Estado final: {'OPENAI ATIVO' if self.openai_available else 'MODO LOCAL'}")
        logger.info("=" * 60)
        
        # Dicionário de tradução para ajudar o GPT
        self.objetos_traduzidos = {
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
            'flower': 'flor',
            'tree': 'árvore',
            'sports ball': 'bola',
            'clock': 'relógio',
            'vase': 'vaso',
            'tie': 'gravata',
            'truck': 'caminhão',
            'backpack': 'mochila',
            'handbag': 'bolsa',
            'remote': 'controle remoto'
        }

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """Gera descrição natural usando apenas GPT"""
        logger.info("🌄 Gerando descrição natural")
        
        # Preparar dados SIMPLES para o GPT
        dados_texto = self._formatar_dados_simples(objetos_detectados, faces_nomes)
        
        if self.openai_available:
            try:
                import openai
                
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Você é a Specula, uma assistente amigável que descreve ambientes para pessoas com deficiência visual.

DADOS DETECTADOS NA IMAGEM (APENAS ESTES):
{dados_texto}

INSTRUÇÕES:
1. Baseie-se APENAS nos dados acima - não invente nada
2. Seja natural, amigável e use emojis 😊
3. Fale na primeira pessoa: "Vejo", "Estou detectando"
4. Se não há dados: "Não estou identificando elementos específicos"
5. Traduza para português se necessário
6. NUNCA peça para o usuário descrever algo (ele é deficiente visual)
7. NUNCA invente posições (esquerda/direita/centro)
8. NUNCA invente objetos não listados

EXEMPLOS:
- Dados: "1 pessoa" → "Vejo uma pessoa na imagem. 😊"
- Dados: "2 pessoas, 1 cadeira" → "Estou detectando duas pessoas e uma cadeira."
- Dados: "Nenhum objeto" → "Não estou identificando elementos específicos no momento."

Agora descreva o que está vendo:"""
                        },
                        {"role": "user", "content": "O que você está vendo na imagem?"}
                    ],
                    max_tokens=120,
                    temperature=0.6,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"❌ Erro OpenAI na descrição: {e}")
                # Fallback muito simples
                return self._descricao_local_simples(objetos_detectados)
        else:
            # Modo local muito simples
            return self._descricao_local_simples(objetos_detectados)

    def _descricao_local_simples(self, objetos_detectados):
        """Fallback local muito simples"""
        if not objetos_detectados:
            return "Olá! Sou a Specula. Não estou identificando elementos específicos."
        
        pessoas = sum(1 for obj in objetos_detectados if obj.get('name') == 'person')
        outros = [obj for obj in objetos_detectados if obj.get('name') != 'person']
        
        if pessoas > 0 and not outros:
            if pessoas == 1:
                return "Vejo uma pessoa."
            else:
                return f"Vejo {pessoas} pessoas."
        elif pessoas > 0 and outros:
            return f"Vejo {pessoas} pessoa(s) e outros objetos."
        else:
            return "Vejo alguns objetos na imagem."

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas usando apenas GPT com instruções claras"""
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Verificar se é pergunta sobre tempo (único tratamento especial necessário)
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
        
        # Preparar dados SIMPLES para o GPT
        dados_texto = self._formatar_dados_simples(objetos_detectados, faces_nomes)
        
        if self.openai_available:
            try:
                import openai
                
                # Analisar se a pergunta é sobre a imagem
                pergunta_lower = pergunta.lower()
                parece_sobre_imagem = any(palavra in pergunta_lower for palavra in [
                    'imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
                    'o que você vê', 'o que tem', 'descreva', 'analise',
                    'quantas pessoas', 'tem ', 'há ', 'vejo', 'identifica',
                    'quantos ', 'tem alguma', 'há alguma'
                ])
                
                # PROMPT INTELIGENTE E CLARO
                prompt = f"""# ESPECIFICAÇÕES DA SPECULA

## CONTEXTO:
Você é a Specula, uma assistente amigável e útil. O usuário é uma pessoa com deficiência visual.

## DADOS DA IMAGEM ATUAL (APENAS ESTES - NADA MAIS):
{dados_texto}

## PERGUNTA DO USUÁRIO:
"{pergunta}"

## REGRAS FUNDAMENTAIS:

### SOBRE OS DADOS DA IMAGEM:
1. BASEIE-SE APENAS nos dados acima - não invente nada
2. Se algo NÃO está nos dados, diga "Não estou detectando [isso]"
3. Não invente posições (esquerda/direita/centro)
4. Não invente quantidades diferentes
5. Não invente objetos não listados
6. Traduza para português se necessário

### SOBRE O USUÁRIO:
7. O usuário é DEFICIENTE VISUAL - NUNCA peça para ele descrever algo
8. NUNCA diga "se você puder me dar mais detalhes"
9. NUNCA diga "descreva o que está vendo"

### SOBRE SUAS RESPOSTAS:
10. Seja NATURAL, AMIGÁVEL e útil 😊
11. Use emojis ocasionalmente para tom amigável
12. Seja clara e direta
13. Para perguntas gerais (não sobre imagem), responda normalmente
14. Se não sabe algo: "Não tenho essa informação no momento"

### EXEMPLOS CORRETOS:
- Dados: "1 pessoa" | Pergunta: "Quantas pessoas?" → "Vejo uma pessoa. 😊"
- Dados: "1 pessoa" | Pergunta: "Tem cadeira?" → "Não estou detectando cadeiras."
- Dados: "Nenhum objeto" | Pergunta: "O que tem?" → "Não estou identificando elementos específicos."
- Pergunta: "Que horas são?" → "🕒 São 10:30 (horário de Brasília)."
- Pergunta: "O que é batata?" → "🍠 A batata é um tubérculo comestível..."

### EXEMPLOS INCORRETOS:
- "Uma pessoa à esquerda" ❌ (inventou posição)
- "Parece ter duas pessoas" ❌ (inventou quantidade)
- "Se você puder descrever..." ❌ (usuário é deficiente visual)
- "Me dê mais detalhes..." ❌ (usuário é deficiente visual)

## AGORA RESPONDA:
Baseando-se APENAS nos dados quando for sobre a imagem, sendo natural e amigável:"""
                
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": pergunta}
                    ],
                    max_tokens=200,
                    temperature=0.7,  # Temperatura natural
                )
                
                resposta = response.choices[0].message.content.strip()
                processing_time = time.time() - start_time
                
                # Determinar tipo (apenas para logging)
                tipo = "sobre_imagem" if parece_sobre_imagem else "geral"
                correlacao = tipo == "sobre_imagem"
                
                return {
                    'sucesso': True,
                    'timestamp': time.time(),
                    'tempo_processamento': f"{processing_time:.2f}s",
                    'pergunta': pergunta,
                    'resposta': resposta,
                    'tipo_pergunta': tipo,
                    'correlacao_com_imagem': correlacao,
                    'dados_utilizados': self._contar_dados(objetos_detectados) if correlacao else "Pergunta geral"
                }
                
            except Exception as e:
                logger.error(f"❌ Erro OpenAI na pergunta: {e}")
                # Fallback muito simples
                return self._resposta_local_simples(pergunta, objetos_detectados, start_time)
        else:
            # Modo local muito simples
            return self._resposta_local_simples(pergunta, objetos_detectados, start_time)

    def _verificar_pergunta_tempo(self, pergunta):
        """Verifica pergunta sobre tempo (única exceção necessária)"""
        pergunta_lower = pergunta.lower()
        
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
        
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data', 'data de hoje']):
            try:
                brasilia_tz = timezone(timedelta(hours=-3))
                agora_brasilia = datetime.now(brasilia_tz)
                data_str = agora_brasilia.strftime("%d/%m/%Y")
                return f"📅 Hoje é {data_str}."
            except:
                agora = datetime.now()
                data_str = agora.strftime("%d/%m/%Y")
                return f"📅 Hoje é {data_str}."
        
        return None

    def _resposta_local_simples(self, pergunta, objetos_detectados, start_time):
        """Fallback local muito simples"""
        pergunta_lower = pergunta.lower()
        
        # Apenas respostas muito básicas
        if 'quem é você' in pergunta_lower:
            resposta = "👋 Eu sou a Specula, sua assistente visual!"
        elif 'batata' in pergunta_lower:
            resposta = "🍠 A batata é um alimento muito versátil!"
        elif 'temperatura' in pergunta_lower:
            resposta = "🌡️ Não tenho acesso a informações meteorológicas."
        elif any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']):
            hora = datetime.now().hour
            if 5 <= hora < 12:
                resposta = "☀️ Bom dia! Sou a Specula."
            elif 12 <= hora < 18:
                resposta = "🌤️ Boa tarde! Sou a Specula."
            else:
                resposta = "🌙 Boa noite! Sou a Specula."
        else:
            resposta = "Olá! Sou a Specula. Como posso te ajudar?"
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'pergunta': pergunta,
            'resposta': resposta,
            'tipo_pergunta': "geral",
            'correlacao_com_imagem': False,
            'dados_utilizados': 'modo local'
        }

    # ========== MÉTODOS AUXILIARES SIMPLES ==========

    def _formatar_dados_simples(self, objetos_detectados, faces_nomes):
        """Formata dados de forma SIMPLES para o GPT"""
        if not objetos_detectados:
            return "Nenhum objeto ou pessoa detectado na imagem."
        
        # Contar pessoas
        pessoas = sum(obj.get('count', 1) for obj in objetos_detectados if obj.get('name') == 'person')
        
        # Listar outros objetos
        outros_objetos = []
        for obj in objetos_detectados:
            nome = obj.get('name', '')
            if nome != 'person':
                quantidade = obj.get('count', 1)
                nome_pt = self.objetos_traduzidos.get(nome, nome)
                if quantidade == 1:
                    outros_objetos.append(f"1 {nome_pt}")
                else:
                    outros_objetos.append(f"{quantidade} {nome_pt}s")
        
        # Construir texto
        partes = []
        
        if pessoas > 0:
            if pessoas == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{pessoas} pessoas")
        
        if outros_objetos:
            partes.extend(outros_objetos)
        
        if not partes:
            return "Nenhum objeto ou pessoa detectado na imagem."
        
        return "Detectado: " + ", ".join(partes)

    def _contar_dados(self, objetos_detectados):
        """Conta dados para logging"""
        if not objetos_detectados:
            return "0 objetos"
        
        pessoas = sum(1 for obj in objetos_detectados if obj.get('name') == 'person')
        outros = len(objetos_detectados) - pessoas
        
        if pessoas > 0 and outros > 0:
            return f"{pessoas} pessoas, {outros} objetos"
        elif pessoas > 0:
            return f"{pessoas} pessoas"
        else:
            return f"{outros} objetos"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """Estatísticas - mantido simples"""
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
                'nome_pt': self.objetos_traduzidos.get(nome_ingles, nome_ingles),
                'nome_en': nome_ingles,
                'confianca': confianca,
                'quantidade': count,
                'categoria': self._classificar_categoria(nome_ingles)
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
                'faces': (faces_detectadas or [])[:5]
            }
        }

    def _classificar_categoria(self, objeto_ingles):
        """Classifica categoria"""
        categorias = {
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'desk'],
            'pessoas': ['person', 'people', 'human'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'monitor'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'plate'],
            'animais': ['dog', 'cat'],
            'veículos': ['car', 'bicycle'],
            'plantas': ['potted plant', 'flower', 'tree'],
            'vestuário': ['tie', 'handbag', 'backpack']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"