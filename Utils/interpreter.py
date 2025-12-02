"""
Interpreter - VERSÃO COMPLETA COM OPENAI 0.28.1
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
        self.openai_available = False
        
        # LOG DE INICIALIZAÇÃO DETALHADO
        logger.info("=" * 60)
        logger.info("🔧 INICIANDO INTERPRETER - VERSÃO COMPLETA")
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
            
            # TENTAR IMPORTAR E CONFIGURAR OPENAI 0.28.1
            try:
                logger.info("🔄 Importando biblioteca OpenAI...")
                import openai
                
                # Verificar versão
                logger.info(f"📦 Versão OpenAI: {openai.__version__}")
                
                # Configurar API key (forma antiga da 0.28.1)
                openai.api_key = api_key
                
                # TESTAR A CONEXÃO
                logger.info("🧪 Testando conexão com OpenAI...")
                try:
                    # Teste simples - fazer uma chamada pequena
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Teste de conexão"}],
                        max_tokens=5
                    )
                    
                    logger.info(f"✅ OpenAI CONECTADO COM SUCESSO!")
                    logger.info(f"✅ Modelo padrão: {self.model_name}")
                    self.openai_available = True
                    
                except Exception as test_e:
                    logger.error(f"❌ Teste de conexão falhou: {test_e}")
                    logger.warning("⚠️ Usando modo local devido a erro de conexão")
                    self.openai_available = False
                    
            except ImportError as e:
                logger.error(f"❌ Biblioteca 'openai' não instalada: {e}")
                logger.error("❌ Execute: pip install openai==0.28.1")
                self.openai_available = False
            except Exception as e:
                logger.error(f"❌ Erro ao configurar OpenAI: {str(e)[:200]}")
                self.openai_available = False
        else:
            logger.warning("⚠️ Nenhuma API key válida recebida")
            logger.warning("⚠️ Usando MODO LOCAL")
            self.openai_available = False
        
        logger.info(f"🎯 Estado final do Interpreter: {'OPENAI ATIVO' if self.openai_available else 'MODO LOCAL'}")
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
        """Gera descrição natural - ADAPTADA PARA DEFICIENTES VISUAIS"""
        logger.info("🌄 Gerando descrição natural")
        
        # Preparar dados
        contador_objetos = self._contar_objetos(objetos_detectados)
        total_pessoas = len(faces_nomes or [])
        
        # Se temos OpenAI disponível, usar IA
        if self.openai_available:
            try:
                # Preparar dados para o prompt - VERSÃO ADAPTADA
                dados_texto = self._formatar_dados_para_prompt(
                    contador_objetos, 
                    total_pessoas, 
                    faces_nomes,
                    objetos_detalhados=objetos_detectados
                )
                
                import openai
                
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Você é a Specula, uma assistente especializada em descrever ambientes para pessoas com deficiência visual.

DADOS DETECTADOS NA IMAGEM:
{dados_texto}

REGRAS IMPORTANTES PARA DEFICIENTES VISUAIS:
1. NUNCA peça para o usuário descrever algo
2. NUNCA diga "se você puder me dar mais detalhes"
3. Baseie-se APENAS nos dados detectados
4. Use informações de posição quando disponível
5. Seja clara e direta nas descrições
6. Use pontos de referência espaciais

EXEMPLOS ADEQUADOS:
- "Estou detectando uma pessoa no centro da imagem."
- "Vejo uma cadeira à direita."
- "Não estou identificando objetos específicos no momento."

EXEMPLOS INADEQUADOS (NUNCA USE):
- "Se você puder descrever melhor..." ❌
- "Me dê mais detalhes sobre..." ❌
- "O que você está vendo?" ❌

Descreva o ambiente baseado nos dados acima:"""
                        },
                        {"role": "user", "content": "Descreva o ambiente que estou vendo:"}
                    ],
                    max_tokens=150,
                    temperature=0.5,
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"❌ Erro OpenAI na descrição: {e}")
                # Fallback para modo local adaptado
                return self._gerar_descricao_local_adaptada(contador_objetos, total_pessoas, faces_nomes, objetos_detectados)
        else:
            # Modo local adaptado
            return self._gerar_descricao_local_adaptada(contador_objetos, total_pessoas, faces_nomes, objetos_detectados)

    def _gerar_descricao_local_adaptada(self, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados=None):
        """Descrição local adaptada para deficientes visuais"""
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

        # Objetos com detalhes se disponível
        if objetos_detalhados:
            for obj in objetos_detalhados:
                nome = obj.get('name', '')
                quantidade = obj.get('count', 1)

                obj_pt = self._traduzir_objeto(nome)

                if quantidade == 1:
                    artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'planta', 'flor', 'árvore'] else "um"
                    partes.append(f"{artigo} {obj_pt}")
                else:
                    partes.append(f"{quantidade} {obj_pt}s")

        # Fallback para contador simples
        elif contador_objetos:
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
            return "Olá, sou a Specula. Não estou identificando elementos específicos nesta imagem no momento."

        inicios = ["Estou detectando ", "Na imagem, vejo ", "Identifico ", "Estou vendo "]
        inicio = random.choice(inicios)

        if len(partes) == 1:
            return f"{inicio}{partes[0]}."
        else:
            return f"{inicio}{', '.join(partes[:-1])} e {partes[-1]}."

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
        """Responde perguntas - VERSÃO ADAPTADA PARA DEFICIENTES VISUAIS"""
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
        
        # DEBUG: Log detalhado dos dados recebidos
        logger.info(f"📥 Dados recebidos para a pergunta '{pergunta}':")
        logger.info(f"   Total objetos: {len(objetos_detectados or [])}")
        logger.info(f"   Contador: {contador_objetos}")
        if objetos_detectados:
            for i, obj in enumerate(objetos_detectados[:5]):
                logger.info(f"   Objeto {i+1}: {obj.get('name')} x{obj.get('count', 1)} conf:{obj.get('confidence', 0):.2f}")
        
        # Se temos OpenAI disponível, usar IA inteligente
        if self.openai_available:
            try:
                # Preparar dados DETALHADOS para o prompt
                dados_texto = self._formatar_dados_para_prompt(
                    contador_objetos, 
                    total_pessoas, 
                    faces_nomes,
                    objetos_detalhados=objetos_detectados
                )
                
                # DEBUG: Log dos dados enviados
                logger.info(f"📤 Dados formatados para OpenAI:\n{dados_texto}")
                
                # Analisar se a pergunta é sobre a imagem
                pergunta_lower = pergunta.lower()
                parece_sobre_imagem = any(palavra in pergunta_lower for palavra in [
                    'imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
                    'o que você vê', 'o que tem', 'descreva', 'analise',
                    'quantas pessoas', 'tem ', 'há ', 'vejo', 'identifica',
                    'onde está', 'posição', 'localização', 'lado', 'direita', 'esquerda',
                    'quantos ', 'tem alguma', 'há alguma', 'ambiente', 'cena', 'cenário'
                ])
                
                # **PROMOT ADAPTADO PARA DEFICIENTES VISUAIS**
                prompt = f"""# ESPECIFICAÇÕES DA SPECULA - ASSISTENTE PARA DEFICIENTES VISUAIS
    
    ## QUEM VOCÊ É:
    Você é a Specula, uma assistente especializada em descrever ambientes para pessoas com deficiência visual. Você NUNCA pede para a   pessoa descrever o que está vendo.
    
    ## DADOS DETECTADOS NA IMAGEM ATUAL (COM POSIÇÕES E CONFIANÇAS):
    {dados_texto}
    
    ## CONTEXTO DA PERGUNTA:
    - Pergunta: "{pergunta}"
    - Tipo: {"SOBRE A IMAGEM" if parece_sobre_imagem else "GERAL"}
    - Usuário: Pessoa com deficiência visual
    
    ## REGRAS ABSOLUTAS - NUNCA FAÇA ISSO:
    ❌ NUNCA peça para o usuário descrever o que está vendo
    ❌ NUNCA diga "se você puder me dar mais detalhes"
    ❌ NUNCA diga "se você puder descrever"
    ❌ NUNCA diga "se você tiver mais informações"
    ❌ NUNCA sugira que o usuário precisa ver algo
    
    ## COMO RESPONDER PARA DEFICIENTES VISUAIS:
    
    ### SE A PERGUNTA É SOBRE A IMAGEM (use os dados acima):
    - Baseie-se APENAS nos dados detectados
    - Use informações de posição (esquerda, direita, centro) quando disponível
    - Mencione confiança da detecção quando relevante
    - Se a pergunta for sobre algo não detectado: "Não estou detectando [objeto] nesta imagem"
    - Se não tem dados suficientes: "Não estou conseguindo identificar [elemento] com clareza"
    - Use descrições táteis/spaciais quando possível
    
    ### SE A PERGUNTA É GERAL:
    - Responda naturalmente como assistente
    - Seja útil e acolhedora
    - Se não souber: "Não tenho essa informação no momento"
    
    ### TOM E ESTILO:
    - Fale de forma clara e direta
    - Use descrições objetivas
    - Seja empática mas não condescendente
    - Use pontos de referência espaciais
    - Adicione emojis ocasionalmente para tom amigável
    
    ### EXEMPLOS DE RESPOSTAS ADEQUADAS:
    - "Vejo uma pessoa no centro da imagem."
    - "Não estou detectando cadeiras neste ambiente."
    - "Há uma mesa à direita, a cerca de 2 metros de distância."
    - "A imagem parece mostrar um ambiente interno bem iluminado."
    
    ### EXEMPLOS DE RESPOSTAS INADEQUADAS (EVITAR):
    - "Se você puder descrever melhor..." ❌
    - "Me dê mais detalhes..." ❌
    - "O que você está vendo?" ❌
    - "Descreva o ambiente para mim..." ❌
    
    IMPORTANTE: Você é a Specula, os olhos do usuário. Descreva o que está detectando sem nunca pedir ajuda visual ao usuário.
    
    Agora responda à pergunta:"""
                
                import openai
                
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": pergunta}
                    ],
                    max_tokens=300,
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
                # Fallback para modo local adaptado
                return self._responder_local_adaptado(pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detectados,   start_time)
        else:
            # Modo local adaptado
            return self._responder_local_adaptado(pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detectados, start_time)
    
    def _responder_local_adaptado(self, pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados, start_time):
        """Resposta local adaptada para deficientes visuais"""
        pergunta_lower = pergunta.lower()
        
        # Respostas pré-definidas ADAPTADAS
        if 'oque é batata' in pergunta_lower or 'o que é batata' in pergunta_lower:
            resposta = "🍠 A batata é um tubérculo comestível muito versátil! Pode ser frita, cozida, assada... É um alimento básico em     muitas culturas!"
            tipo = "geral"
        
        elif 'quem é você' in pergunta_lower:
            resposta = "👋 Eu sou a Specula! Sou sua assistente visual, criada para ajudar a entender ambientes através da análise de   imagens."
            tipo = "geral"
        
        elif any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']):
            resposta = self._gerar_cumprimento_adaptado()
            tipo = "geral"
        
        elif 'temperatura' in pergunta_lower:
            resposta = "🌡️ Não tenho acesso a informações meteorológicas no momento. Mas posso te ajudar analisando imagens do ambiente!"
            tipo = "geral"
        
        # Perguntas sobre imagem - RESPOSTAS ADAPTADAS
        elif any(palavra in pergunta_lower for palavra in ['imagem', 'foto', 'essa ', 'esta ', 'o que tem', 'descreva', 'quantas pessoas',  'onde está', 'tem ']):
            # Usar dados da imagem
            resposta = self._gerar_resposta_sobre_imagem_adaptada(pergunta, contador_objetos, total_pessoas, faces_nomes,   objetos_detalhados)
            tipo = "sobre_imagem"
        
        else:
            resposta = "Olá! Sou a Specula, sua assistente visual. Como posso te ajudar com análise de imagens?"
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
    
    def _gerar_resposta_sobre_imagem_adaptada(self, pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados):
        """Resposta adaptada sobre imagem para deficientes visuais"""
        pergunta_lower = pergunta.lower()
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Objetos com detalhes
        objetos_pt = {}
        if objetos_detalhados:
            for obj in objetos_detalhados:
                nome = obj.get('name', '')
                quantidade = obj.get('count', 1)
                obj_pt = self._traduzir_objeto(nome)
                objetos_pt[obj_pt] = objetos_pt.get(obj_pt, 0) + quantidade
        
        # "O que tem nessa imagem?" ou "Descreva o ambiente"
        if any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva', 'ambiente']):
            if total_detectado > 0:
                if total_detectado == 1:
                    base = "Na imagem, estou detectando uma pessoa."
                elif total_detectado == 2:
                    base = "Estou vendo duas pessoas."
                else:
                    base = f"Vejo {total_detectado} pessoas."
                
                if objetos_pt:
                    objetos_lista = []
                    for obj_pt, qtd in objetos_pt.items():
                        if qtd == 1:
                            artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'planta', 'flor', 'árvore'] else "um"
                            objetos_lista.append(f"{artigo} {obj_pt}")
                        else:
                            objetos_lista.append(f"{qtd} {obj_pt}s")
                    
                    if objetos_lista:
                        return f"{base} Também identifico {', '.join(objetos_lista)}."
                
                return base + " Não estou identificando outros objetos específicos."
            elif objetos_pt:
                primeiro = list(objetos_pt.items())[0]
                obj_nome = primeiro[0]
                qtd = primeiro[1]
                
                if qtd == 1:
                    artigo = "uma" if obj_nome in ['bola', 'cadeira', 'mesa', 'planta'] else "um"
                    return f"Estou detectando {artigo} {obj_nome}."
                else:
                    return f"Vejo {qtd} {obj_nome}s."
            else:
                return "Não estou identificando elementos específicos nesta imagem no momento."
        
        # "Quantas pessoas?"
        elif 'quantas pessoas' in pergunta_lower:
            if total_detectado > 0:
                if total_detectado == 1:
                    return "Estou detectando uma pessoa na imagem."
                elif total_detectado == 2:
                    return "Vejo duas pessoas."
                else:
                    return f"Identifico {total_detectado} pessoas."
            else:
                return "Não estou detectando pessoas nesta imagem."
        
        # "Tem [objeto]?"
        elif 'tem ' in pergunta_lower:
            for obj_pt in self.objetos_traduzidos.values():
                if obj_pt in pergunta_lower:
                    tem_objeto = obj_pt in objetos_pt
                    if tem_objeto:
                        quantidade = objetos_pt[obj_pt]
                        if quantidade == 1:
                            artigo = "uma" if obj_pt in ['bola', 'cadeira', 'mesa', 'planta', 'flor', 'árvore'] else "um"
                            return f"Sim, estou detectando {artigo} {obj_pt}."
                        else:
                            return f"Sim, identifico {quantidade} {obj_pt}s."
                    else:
                        return f"Não, não estou detectando {obj_pt}."
        
        # "Esta foto parece ser interna ou externa?"
        elif any(palavra in pergunta_lower for palavra in ['interna', 'externa', 'dentro', 'fora']):
            # Baseado nos objetos detectados, tentar inferir
            if 'person' in contador_objetos and len(contador_objetos) == 1:
                return "Com base na detecção, parece ser um ambiente interno, mas não tenho certeza absoluta."
            else:
                return "Não tenho informações suficientes para determinar se é interno ou externo."
        
        # Resposta genérica adaptada
        if total_detectado > 0:
            return f"Estou detectando {total_detectado} pessoa{'s' if total_detectado > 1 else ''} na imagem."
        elif objetos_pt:
            primeiro = list(objetos_pt.items())[0]
            obj_nome = primeiro[0]
            qtd = primeiro[1]
            
            if qtd == 1:
                artigo = "uma" if obj_nome in ['bola', 'cadeira', 'mesa', 'planta'] else "um"
                return f"Identifico {artigo} {obj_nome}."
            else:
                return f"Vejo {qtd} {obj_nome}s."
        else:
            return "Não estou conseguindo identificar elementos específicos nesta imagem."
    
    def _gerar_cumprimento_adaptado(self):
        """Gera cumprimento adaptado"""
        hora = datetime.now().hour
        if 5 <= hora < 12:
            return "☀️ Bom dia! Eu sou a Specula, sua assistente visual. Estou pronta para descrever o ambiente para você!"
        elif 12 <= hora < 18:
            return "🌤️ Boa tarde! Sou a Specula, sua assistente para análise de ambientes. O que gostaria de saber?"
        else:
            return "🌙 Boa noite! Eu sou a Specula, pronta para ajudar você a entender o ambiente ao seu redor."
    
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