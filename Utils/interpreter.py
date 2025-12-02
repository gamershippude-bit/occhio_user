"""
Interpreter - VERSÃO FINAL COM OPENAI CORRIGIDO
"""

import logging
import os
import time
import random
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        self.model_name = model_name
        self.client = None
        
        # LOG DE INICIALIZAÇÃO DETALHADO
        logger.info("=" * 60)
        logger.info("🔧 INICIANDO INTERPRETER - VERSÃO FINAL")
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
                from openai import OpenAI
                
                logger.info("🔄 Configurando cliente OpenAI...")
                self.client = OpenAI(api_key=api_key)
                
                # TESTAR A CONEXÃO
                logger.info("🧪 Testando conexão com OpenAI...")
                try:
                    # Teste simples - listar modelos
                    models = self.client.models.list(limit=1)
                    logger.info(f"✅ OpenAI CONECTADO COM SUCESSO!")
                    logger.info(f"✅ Modelo padrão: {self.model_name}")
                    
                except Exception as test_e:
                    logger.error(f"❌ Teste de conexão falhou: {test_e}")
                    logger.warning("⚠️ Usando modo local devido a erro de conexão")
                    self.client = None
                    
            except ImportError as e:
                logger.error(f"❌ Biblioteca 'openai' não instalada: {e}")
                logger.error("❌ Execute: pip install openai")
                self.client = None
            except Exception as e:
                logger.error(f"❌ Erro ao configurar OpenAI: {str(e)[:200]}")
                self.client = None
        else:
            logger.warning("⚠️ Nenhuma API key válida recebida")
            logger.warning("⚠️ Usando MODO LOCAL")
            self.client = None
        
        logger.info(f"🎯 Estado final do Interpreter: {'OPENAI ATIVO' if self.client else 'MODO LOCAL'}")
        logger.info("=" * 60)
        
        # Dicionário de tradução
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
            'vase': 'vaso'
        }

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """Gera descrição natural"""
        logger.info("🌄 Gerando descrição natural")
        
        # Preparar dados
        contador_objetos = self._contar_objetos(objetos_detectados)
        total_pessoas = len(faces_nomes or [])
        
        # Se temos OpenAI client, usar IA
        if self.client:
            try:
                # Preparar dados para o prompt
                dados_texto = self._formatar_dados_para_prompt(contador_objetos, total_pessoas, faces_nomes)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Você é a Specula, uma assistente que descreve ambientes para pessoas com deficiência visual.

DADOS DETECTADOS NA IMAGEM:
{dados_texto}

REGRAS:
1. Descreva APENAS o que está nos dados acima
2. Não invente nada que não foi detectado
3. Use artigos corretos: "uma bola", "um sofá"
4. Seja natural e acolhedora
5. Você é a Specula - apresente-se brevemente

EXEMPLOS:
- Se tem "2 pessoas" → "Tem duas pessoas."
- Se tem "1 cadeira" → "Tem uma cadeira."
- Se nada detectado → "Não estou identificando elementos específicos."

Responda naturalmente:"""
                        },
                        {"role": "user", "content": "Descreva o ambiente que está vendo:"}
                    ],
                    max_tokens=150,
                    temperature=0.5,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"❌ Erro OpenAI na descrição: {e}")
                # Fallback para modo local
                return self._gerar_descricao_local(contador_objetos, total_pessoas, faces_nomes)
        else:
            # Modo local
            return self._gerar_descricao_local(contador_objetos, total_pessoas, faces_nomes)

    def _formatar_dados_para_prompt(self, contador_objetos, total_pessoas, faces_nomes):
        """Formata dados para o prompt"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_nomes and any(n != 'Desconhecido' for n in faces_nomes):
                conhecidas = [n for n in faces_nomes if n != 'Desconhecido']
                if conhecidas:
                    partes.append(f"Pessoas identificadas: {', '.join(conhecidas)}")
            else:
                partes.append(f"Pessoas: {total_detectado}")
        
        # Objetos
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
            return "Nenhum objeto ou pessoa detectado."
        
        return " | ".join(partes)

    def _gerar_descricao_local(self, contador_objetos, total_pessoas, faces_nomes):
        """Descrição local"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_nomes and any(n != 'Desconhecido' for n in faces_nomes):
                conhecidas = [n for n in faces_nomes if n != 'Desconhecido']
                if len(conhecidas) == 1:
                    partes.append(f"o {conhecidas[0]}")
                else:
                    partes.append(f"o {', '.join(conhecidas[:-1])} e o {conhecidas[-1]}")
            else:
                if total_detectado == 1:
                    partes.append("uma pessoa")
                elif total_detectado == 2:
                    partes.append("duas pessoas")
                else:
                    partes.append(f"{total_detectado} pessoas")
        
        # Objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            for obj_ingles, quantidade in outros_objetos.items():
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'planta', 'flor', 'árvore'] else "um"
                    partes.append(f"{artigo} {obj_pt}")
                else:
                    partes.append(f"{quantidade} {obj_pt}s")
        
        if not partes:
            return "Olá! Sou a Specula. Não estou identificando elementos específicos nesta imagem."
        
        inicios = ["Na imagem, ", "Vejo que ", "Analisando, ", "Na foto, "]
        inicio = random.choice(inicios)
        
        if len(partes) == 1:
            return f"{inicio}tem {partes[0]}."
        else:
            return f"{inicio}tem {', '.join(partes[:-1])} e {partes[-1]}."

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas"""
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Verificar se é pergunta sobre tempo (tratamento especial)
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
        
        # Preparar dados
        contador_objetos = self._contar_objetos(objetos_detectados)
        total_pessoas = len(faces_nomes or [])
        
        # Se temos OpenAI client, usar IA inteligente
        if self.client:
            try:
                # Preparar dados
                dados_texto = self._formatar_dados_para_prompt(contador_objetos, total_pessoas, faces_nomes)
                
                # Analisar se a pergunta é sobre a imagem
                pergunta_lower = pergunta.lower()
                parece_sobre_imagem = any(palavra in pergunta_lower for palavra in [
                    'imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
                    'o que você vê', 'o que tem', 'descreva', 'analise',
                    'quantas pessoas', 'tem ', 'há ', 'vejo', 'identifica'
                ])
                
                # Criar prompt inteligente
                prompt = f"""# ESPECIFICAÇÕES DA SPECULA

## QUEM VOCÊ É:
Você é a Specula, uma assistente amigável, empática e útil.

## DADOS DA IMAGEM ATUAL:
{dados_texto}

## CONTEXTO DA PERGUNTA:
- Pergunta: "{pergunta}"
- Parece ser sobre a imagem? {"SIM" if parece_sobre_imagem else "NÃO"}

## COMO RESPONDER:

### SE A PERGUNTA É SOBRE A IMAGEM:
- Use os dados acima se disponíveis
- Não invente o que não foi detectado
- Use artigos corretos: "uma bola", "um sofá"
- Se não tem dados: "Não estou detectando..."

### SE A PERGUNTA É GERAL:
- Responda naturalmente como um assistente
- Seja útil e amigável
- Use emojis ocasionalmente 😊
- Se não souber: "Não tenho essa informação, mas posso ajudar com imagens!"

### SEMPRE:
- Seja a Specula: acolhedora, paciente e útil
- Variedade nas respostas
- Humanização: fale como pessoa real

Agora responda à pergunta como a Specula:"""
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": pergunta}
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )
                
                resposta = response.choices[0].message.content.strip()
                processing_time = time.time() - start_time
                
                # Determinar tipo
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
                    'dados_utilizados': self._formatar_dados_utilizados(objetos_detectados, faces_nomes) if correlacao else "Pergunta geral"
                }
                
            except Exception as e:
                logger.error(f"❌ Erro OpenAI na pergunta: {e}")
                # Fallback para modo local
                return self._responder_local(pergunta, contador_objetos, total_pessoas, faces_nomes, start_time)
        else:
            # Modo local
            return self._responder_local(pergunta, contador_objetos, total_pessoas, faces_nomes, start_time)

    def _responder_local(self, pergunta, contador_objetos, total_pessoas, faces_nomes, start_time):
        """Resposta local"""
        pergunta_lower = pergunta.lower()
        
        # Respostas pré-definidas
        if 'oque é batata' in pergunta_lower or 'o que é batata' in pergunta_lower:
            resposta = "🍠 A batata é um tubérculo comestível muito versátil! Pode ser frita, cozida, assada... É um alimento básico em muitas culturas!"
            tipo = "geral"
        
        elif 'quem é você' in pergunta_lower:
            resposta = "👋 Eu sou a Specula! Uma assistente criada para ajudar pessoas a entender melhor o ambiente através da análise de imagens."
            tipo = "geral"
        
        elif any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']):
            resposta = self._gerar_cumprimento()
            tipo = "geral"
        
        elif 'temperatura' in pergunta_lower:
            resposta = "🌡️ No momento não tenho acesso a informações meteorológicas em tempo real. Mas posso te ajudar analisando imagens!"
            tipo = "geral"
        
        # Perguntas sobre imagem
        elif any(palavra in pergunta_lower for palavra in ['imagem', 'foto', 'essa ', 'esta ', 'o que tem', 'descreva', 'quantas pessoas']):
            # Usar dados da imagem
            resposta = self._gerar_resposta_sobre_imagem_local(pergunta, contador_objetos, total_pessoas, faces_nomes)
            tipo = "sobre_imagem"
        
        else:
            resposta = "Olá! Sou a Specula. Como posso te ajudar com análise de imagens?"
            tipo = "geral"
        
        processing_time = time.time() - start_time
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'pergunta': pergunta,
            'resposta': resposta,
            'tipo_pergunta': tipo,
            'correlacao_com_imagem': tipo == "sobre_imagem",
            'dados_utilizados': 'modo local'
        }

    def _gerar_resposta_sobre_imagem_local(self, pergunta, contador_objetos, total_pessoas, faces_nomes):
        """Resposta local sobre imagem"""
        pergunta_lower = pergunta.lower()
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        objetos_pt = {self._traduzir_objeto(obj): qtd for obj, qtd in outros_objetos.items()}
        
        # "Quantas pessoas?"
        if 'quantas pessoas' in pergunta_lower:
            if total_detectado > 0:
                if total_detectado == 1:
                    return "Tem uma pessoa."
                elif total_detectado == 2:
                    return "Tem duas pessoas."
                else:
                    return f"Tem {total_detectado} pessoas."
            else:
                return "Não tem pessoas visíveis."
        
        # "O que tem?" ou "Descreva"
        elif any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva']):
            return self._gerar_descricao_local(contador_objetos, total_pessoas, faces_nomes)
        
        # "Tem [objeto]?"
        elif 'tem ' in pergunta_lower:
            for objeto in ['cadeira', 'mesa', 'computador', 'tv', 'planta', 'carro']:
                if objeto in pergunta_lower:
                    tem_objeto = any(objeto in obj_pt.lower() for obj_pt in objetos_pt.keys())
                    return f"{'Sim' if tem_objeto else 'Não'}, {'tem' if tem_objeto else 'não tem'} {objeto}."
        
        # Resposta genérica
        if total_detectado > 0:
            return f"Vejo {total_detectado} pessoa{'s' if total_detectado > 1 else ''} na imagem."
        elif objetos_pt:
            primeiro = list(objetos_pt.items())[0]
            obj_nome = primeiro[0]
            qtd = primeiro[1]
            
            if qtd == 1:
                artigo = "uma" if obj_nome in ['bola', 'cadeira', 'mesa', 'planta'] else "um"
                return f"Vejo {artigo} {obj_nome}."
            else:
                return f"Vejo {qtd} {obj_nome}s."
        else:
            return "Não estou identificando elementos específicos nesta imagem."

    def _verificar_pergunta_tempo(self, pergunta):
        """Verifica pergunta sobre tempo"""
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

    def _gerar_cumprimento(self):
        """Gera cumprimento"""
        hora = datetime.now().hour
        if 5 <= hora < 12:
            return "☀️ Bom dia! Eu sou a Specula, sua assistente visual. Como posso te ajudar hoje?"
        elif 12 <= hora < 18:
            return "🌤️ Boa tarde! Eu sou a Specula, pronta para te ajudar com análise de imagens. O que precisa?"
        else:
            return "🌙 Boa noite! Sou a Specula, sua assistente visual. Como posso te ajudar nesta noite?"

    # ========== MÉTODOS AUXILIARES ==========

    def _contar_objetos(self, objetos_detectados):
        """Conta objetos"""
        contador = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', '')
            count = obj.get('count', 1)
            contador[nome] = contador.get(nome, 0) + count
        return contador

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objeto"""
        return self.objetos_traduzidos.get(objeto_ingles, objeto_ingles)

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados"""
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        return f"{objetos_count} objetos, {faces_count} pessoas"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """Estatísticas"""
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
            'plantas': ['potted plant', 'flower', 'tree']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"