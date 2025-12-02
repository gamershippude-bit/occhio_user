"""
Interpreter - VERSÃO HIPER-RESTRITIVA - SEM INVENÇÕES
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
        logger.info("🔧 INICIANDO INTERPRETER - VERSÃO HIPER-RESTRITIVA")
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
            'vase': 'vaso',
            'tie': 'gravata',
            'truck': 'caminhão',
            'backpack': 'mochila',
            'handbag': 'bolsa',
            'remote': 'controle remoto'
        }

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """Gera descrição natural - HIPER RESTRITIVA"""
        logger.info("🌄 Gerando descrição natural")
        
        # Preparar dados
        contador_objetos = self._contar_objetos(objetos_detectados)
        total_pessoas = len(faces_nomes or [])
        
        # Se temos OpenAI disponível, usar IA
        if self.openai_available:
            try:
                # Preparar dados para o prompt - VERSÃO HIPER RESTRITIVA
                dados_texto = self._formatar_dados_para_prompt_restritivo(
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
                            "content": f"""# ESPECIFICAÇÕES DA SPECULA - VERSÃO HIPER RESTRITIVA

## SUA FUNÇÃO:
Você é a Specula, uma assistente que APENAS repete os dados detectados. Você NÃO inventa, NÃO adiciona, NÃO interpreta.

## DADOS DETECTADOS (APENAS ESTES):
{dados_texto}

## REGRAS ABSOLUTAS - NUNCA FAÇA:
❌ NUNCA invente posições (esquerda, direita, centro)
❌ NUNCA invente quantidades diferentes das fornecidas
❌ NUNCA invente objetos não listados
❌ NUNCA use "parece", "aparenta", "talvez", "provavelmente"
❌ NUNCA faça suposições sobre ambiente (interno/externo)
❌ NUNCA mencione confianças ou porcentagens
❌ NUNCA peça para o usuário descrever algo

## COMO RESPONDER:
1. Se tem dados: Diga APENAS o que está nos dados
2. Se não tem dados: "Não estou detectando elementos específicos"
3. Use frases curtas e diretas
4. Não adicione interpretações

## EXEMPLOS CORRETOS:
- Se dados são "1 pessoa": "Estou detectando uma pessoa."
- Se dados são "2 pessoas, 1 cadeira": "Estou detectando duas pessoas e uma cadeira."
- Se dados são "Nenhum objeto ou pessoa detectado": "Não estou detectando elementos específicos."

## EXEMPLOS INCORRETOS:
- "Uma pessoa à esquerda" ❌ (inventou posição)
- "Parece ter duas pessoas" ❌ (usou "parece")
- "Talvez seja um ambiente interno" ❌ (fez suposição)
- "Provavelmente há uma mesa" ❌ (inventou objeto)

Responda APENAS com base nos dados acima:"""
                        },
                        {"role": "user", "content": "Descreva o ambiente que estou vendo:"}
                    ],
                    max_tokens=100,  # MENOS tokens para evitar invenções
                    temperature=0.3,  # TEMPERATURA BAIXA para menos criatividade
                )
                
                resposta = response.choices[0].message.content.strip()
                
                # VALIDAÇÃO EXTRA: Verificar se resposta não inventou nada
                resposta_validada = self._validar_resposta_contra_dados(resposta, contador_objetos, total_pessoas)
                
                return resposta_validada
                
            except Exception as e:
                logger.error(f"❌ Erro OpenAI na descrição: {e}")
                # Fallback para modo local
                return self._gerar_descricao_local_restritiva(contador_objetos, total_pessoas, faces_nomes, objetos_detectados)
        else:
            # Modo local
            return self._gerar_descricao_local_restritiva(contador_objetos, total_pessoas, faces_nomes, objetos_detectados)

    def _gerar_descricao_local_restritiva(self, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados=None):
        """Descrição local restritiva"""
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Objetos
        objetos_lista = []
        if objetos_detalhados:
            objetos_agrupados = {}
            for obj in objetos_detalhados:
                nome = obj.get('name', '')
                quantidade = obj.get('count', 1)
                if nome not in objetos_agrupados:
                    objetos_agrupados[nome] = 0
                objetos_agrupados[nome] += quantidade
            
            for nome_ingles, quantidade in objetos_agrupados.items():
                if nome_ingles != 'person':  # Pessoas já tratamos separadamente
                    nome_pt = self._traduzir_objeto(nome_ingles)
                    if quantidade == 1:
                        objetos_lista.append(f"1 {nome_pt}")
                    else:
                        objetos_lista.append(f"{quantidade} {nome_pt}s")
        
        # Montar resposta
        partes = []
        
        if total_detectado > 0:
            if total_detectado == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{total_detectado} pessoas")
        
        if objetos_lista:
            partes.extend(objetos_lista)
        
        if not partes:
            return "Não estou detectando elementos específicos."
        
        if len(partes) == 1:
            return f"Estou detectando {partes[0]}."
        else:
            return f"Estou detectando {', '.join(partes[:-1])} e {partes[-1]}."

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas - VERSÃO HIPER RESTRITIVA"""
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
                dados_texto = self._formatar_dados_para_prompt_restritivo(
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
                
                # **PROMOT HIPER RESTRITIVO**
                prompt = f"""# ESPECIFICAÇÕES DA SPECULA - VERSÃO HIPER RESTRITIVA

## SUA FUNÇÃO:
Você é a Specula, uma assistente que APENAS usa os dados detectados. NÃO invente, NÃO adicione, NÃO interprete.

## DADOS DETECTADOS (APENAS ESTES - NADA MAIS):
{dados_texto}

## PERGUNTA DO USUÁRIO:
"{pergunta}"

## REGRAS ABSOLUTAS - NUNCA FAÇA:
❌ NUNCA invente posições (esquerda, direita, centro, etc.)
❌ NUNCA invente quantidades diferentes das fornecidas
❌ NUNCA invente objetos não listados
❌ NUNCA use palavras como "parece", "aparenta", "talvez", "provavelmente"
❌ NUNCA faça suposições sobre o ambiente
❌ NUNCA mencione confianças ou porcentagens
❌ NUNCA peça para o usuário descrever algo

## COMO RESPONDER PARA PERGUNTAS SOBRE A IMAGEM:
1. Se a pergunta é sobre algo NÃO detectado: "Não estou detectando [objeto/pergunta]"
2. Se a pergunta é sobre algo DETECTADO: Responda APENAS com os dados
3. Se não tem dados: "Não estou detectando elementos específicos"

## EXEMPLOS CORRETOS:
- Dados: "1 pessoa" | Pergunta: "Quantas pessoas?" → "1 pessoa"
- Dados: "1 pessoa" | Pergunta: "Tem cadeira?" → "Não estou detectando cadeira"
- Dados: "Nenhum objeto" | Pergunta: "O que tem?" → "Não estou detectando elementos específicos"

## EXEMPLOS INCORRETOS:
- "Uma pessoa à esquerda" ❌ (inventou posição)
- "Parece ter duas pessoas" ❌ (usou "parece")
- "Talvez seja interno" ❌ (fez suposição)
- "Provavelmente 90% de certeza" ❌ (mencionou confiança)

## IMPORTANTE:
Você recebeu APENAS os dados acima. Se não está nos dados, NÃO EXISTE.

Agora responda à pergunta APENAS com base nos dados fornecidos:"""
                
                import openai
                
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": pergunta}
                    ],
                    max_tokens=150,  # MENOS tokens para evitar invenções
                    temperature=0.3,  # TEMPERATURA BAIXA para menos criatividade
                )
                
                resposta = response.choices[0].message.content.strip()
                processing_time = time.time() - start_time
                
                # VALIDAÇÃO: Verificar se resposta não inventou nada
                resposta = self._validar_resposta_contra_dados(resposta, contador_objetos, total_pessoas)
                
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
                # Fallback para modo local restritivo
                return self._responder_local_restritivo(pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detectados, start_time)
        else:
            # Modo local restritivo
            return self._responder_local_restritivo(pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detectados, start_time)

    def _formatar_dados_para_prompt_restritivo(self, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados=None):
        """Formata dados para o prompt - HIPER RESTRITIVO - SEM POSIÇÕES"""
        partes = []
        
        # Pessoas - contagem SIMPLES
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if total_detectado == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{total_detectado} pessoas")
        
        # Objetos - contagem SIMPLES, SEM posições
        if objetos_detalhados:
            # Agrupar objetos por tipo
            objetos_agrupados = {}
            for obj in objetos_detalhados:
                nome = obj.get('name', '')
                if nome != 'person':  # Pessoas já tratamos
                    quantidade = obj.get('count', 1)
                    if nome not in objetos_agrupados:
                        objetos_agrupados[nome] = 0
                    objetos_agrupados[nome] += quantidade
            
            # Formatar objetos agrupados
            for nome_ingles, quantidade in objetos_agrupados.items():
                nome_pt = self._traduzir_objeto(nome_ingles)
                
                if quantidade == 1:
                    partes.append(f"1 {nome_pt}")
                else:
                    partes.append(f"{quantidade} {nome_pt}s")
        
        # Fallback para contador simples
        elif contador_objetos:
            outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
            if outros_objetos:
                for obj_ingles, quantidade in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj_ingles)
                    if quantidade == 1:
                        partes.append(f"1 {obj_pt}")
                    else:
                        partes.append(f"{quantidade} {obj_pt}s")
        
        if not partes:
            return "Nenhum objeto ou pessoa detectado"
        
        return " | ".join(partes)

    def _validar_resposta_contra_dados(self, resposta, contador_objetos, total_pessoas):
        """Valida se a resposta não inventou dados"""
        resposta_lower = resposta.lower()
        
        # Verificar se inventou posições
        palavras_proibidas_posicao = ['esquerda', 'direita', 'centro', 'lado', 'posicionada', 'localizada']
        for palavra in palavras_proibidas_posicao:
            if palavra in resposta_lower:
                logger.warning(f"⚠️ Resposta inventou posição: '{palavra}'")
                # Corrigir resposta
                return self._corrigir_resposta_inventada(resposta, contador_objetos, total_pessoas)
        
        # Verificar se inventou quantidades de pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Contar menções a "pessoa" na resposta
        if total_detectado == 1:
            # Não deve mencionar mais de 1 pessoa
            if resposta_lower.count('pessoa') > 1 or resposta_lower.count('pessoas') > 0:
                logger.warning(f"⚠️ Resposta inventou mais pessoas: {resposta}")
                return "Estou detectando uma pessoa."
        elif total_detectado == 0:
            # Não deve mencionar pessoas
            if 'pessoa' in resposta_lower or 'pessoas' in resposta_lower:
                logger.warning(f"⚠️ Resposta inventou pessoas: {resposta}")
                return "Não estou detectando pessoas."
        
        # Verificar palavras de suposição
        palavras_suposicao = ['parece', 'aparenta', 'talvez', 'provavelmente', 'acho que', 'acredito que']
        for palavra in palavras_suposicao:
            if palavra in resposta_lower:
                logger.warning(f"⚠️ Resposta usou suposição: '{palavra}'")
                # Remover suposições
                for sup in palavras_suposicao:
                    resposta = resposta.replace(sup, "").replace(sup.capitalize(), "")
                return resposta.strip()
        
        return resposta

    def _corrigir_resposta_inventada(self, resposta, contador_objetos, total_pessoas):
        """Corrige resposta que inventou dados"""
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado == 1:
            return "Estou detectando uma pessoa."
        elif total_detectado > 1:
            return f"Estou detectando {total_detectado} pessoas."
        else:
            return "Não estou detectando elementos específicos."

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

    def _responder_local_restritivo(self, pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados, start_time):
        """Resposta local restritiva"""
        pergunta_lower = pergunta.lower()
        
        # Respostas pré-definidas
        if 'oque é batata' in pergunta_lower or 'o que é batata' in pergunta_lower:
            resposta = "🍠 A batata é um tubérculo comestível muito versátil! Pode ser frita, cozida, assada... É um alimento básico em muitas culturas!"
            tipo = "geral"
        
        elif 'quem é você' in pergunta_lower:
            resposta = "👋 Eu sou a Specula! Sou sua assistente visual, criada para ajudar a entender ambientes através da análise de imagens."
            tipo = "geral"
        
        elif any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite']):
            resposta = self._gerar_cumprimento()
            tipo = "geral"
        
        elif 'temperatura' in pergunta_lower:
            resposta = "🌡️ Não tenho acesso a informações meteorológicas no momento."
            tipo = "geral"
        
        # Perguntas sobre imagem - RESTRITIVAS
        elif any(palavra in pergunta_lower for palavra in ['imagem', 'foto', 'essa ', 'esta ', 'o que tem', 'descreva', 'quantas pessoas', 'onde está', 'tem ']):
            # Usar dados da imagem SEM inventar
            resposta = self._gerar_resposta_sobre_imagem_restritiva(pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados)
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

    def _gerar_cumprimento(self):
        """Gera cumprimento"""
        hora = datetime.now().hour
        if 5 <= hora < 12:
            return "☀️ Bom dia! Eu sou a Specula, sua assistente visual."
        elif 12 <= hora < 18:
            return "🌤️ Boa tarde! Sou a Specula, sua assistente para análise de ambientes."
        else:
            return "🌙 Boa noite! Eu sou a Specula, sua assistente visual."

    def _gerar_resposta_sobre_imagem_restritiva(self, pergunta, contador_objetos, total_pessoas, faces_nomes, objetos_detalhados):
        """Resposta restritiva sobre imagem - NÃO INVENTA"""
        pergunta_lower = pergunta.lower()
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        # Objetos
        objetos_pt = {}
        if objetos_detalhados:
            for obj in objetos_detalhados:
                nome = obj.get('name', '')
                if nome != 'person':  # Pessoas separadas
                    quantidade = obj.get('count', 1)
                    obj_pt = self._traduzir_objeto(nome)
                    objetos_pt[obj_pt] = objetos_pt.get(obj_pt, 0) + quantidade
        
        # "O que tem nessa imagem?" ou "Descreva"
        if any(palavra in pergunta_lower for palavra in ['o que tem', 'descreva', 'ambiente']):
            return self._gerar_descricao_local_restritiva(contador_objetos, total_pessoas, faces_nomes, objetos_detalhados)
        
        # "Quantas pessoas?"
        elif 'quantas pessoas' in pergunta_lower:
            if total_detectado > 0:
                if total_detectado == 1:
                    return "1 pessoa"
                else:
                    return f"{total_detectado} pessoas"
            else:
                return "Não estou detectando pessoas"
        
        # "Tem [objeto]?"
        elif 'tem ' in pergunta_lower:
            for obj_pt in self.objetos_traduzidos.values():
                if obj_pt in pergunta_lower:
                    if obj_pt in objetos_pt:
                        quantidade = objetos_pt[obj_pt]
                        if quantidade == 1:
                            return f"Sim, 1 {obj_pt}"
                        else:
                            return f"Sim, {quantidade} {obj_pt}s"
                    else:
                        return f"Não, não estou detectando {obj_pt}"
        
        # Resposta genérica
        if total_detectado > 0:
            if total_detectado == 1:
                return "Estou detectando uma pessoa"
            else:
                return f"Estou detectando {total_detectado} pessoas"
        elif objetos_pt:
            primeiro = list(objetos_pt.items())[0]
            obj_nome = primeiro[0]
            qtd = primeiro[1]
            
            if qtd == 1:
                return f"Estou detectando 1 {obj_nome}"
            else:
                return f"Estou detectando {qtd} {obj_nome}s"
        else:
            return "Não estou detectando elementos específicos"

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
            'plantas': ['potted plant', 'flower', 'tree'],
            'vestuário': ['tie', 'handbag', 'backpack']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"