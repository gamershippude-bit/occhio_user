"""
Interpreter - Versão OTMIIZADA com respostas naturais e humanizadas
"""

import logging
import os
import time
import math
import re
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
            'copo': ['wine glass', 'cup'],
            'garrafa': ['bottle'],
            'cadeira': ['chair'],
            'vaso sanitário': ['toilet'],
            'pia': ['sink'],
            'porta': ['door'],
            'janela': ['window'],
            'planta': ['potted plant'],
            'flor': ['flower']
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

    # ========== NOVA FUNÇÃO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /processar - Gera descrição natural do ambiente
        Usa IA generativa para criar uma descrição fluida e natural
        """
        logger.info("🌄 Gerando descrição natural do ambiente")
        
        if not self.client:
            return "Estou analisando o ambiente para você. Entre em contato para mais detalhes."
        
        # Preparar dados das detecções
        objetos_filtrados = self._filtrar_objetos_relevantes(
            [obj.get('name', '') for obj in (objetos_detectados or [])]
        )
        
        contador_objetos = Counter(objetos_filtrados)
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Construir contexto para a IA
        contexto = self._construir_contexto_descricao(contador_objetos, total_pessoas, faces_conhecidas)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Você é um assistente visual que ajuda pessoas com deficiência visual a entender seu ambiente.
                    
                    REGRAS IMPORTANTES:
                    1. Você ESTÁ analisando uma imagem real, não está apenas imaginando
                    2. Suas respostas devem ser NATURAIS, como se estivesse conversando
                    3. Se não houver muitos dados, seja honesto mas útil
                    4. Use linguagem acessível e descritiva
                    5. Evite dizer "não tenho dados" - em vez disso, descreva o que consegue perceber
                    
                    EXEMPLOS DE COMO FALAR:
                    ✅ "Pelo que consigo analisar na imagem..."
                    ✅ "Na cena que estou vendo..."
                    ✅ "Analisando o ambiente, percebo que..."
                    ❌ "Não tenho dados sobre..."
                    ❌ "Não consigo visualizar..."
                    
                    Seja útil e acolhedor!"""
                },
                {
                    "role": "user", 
                    "content": f"Analise esta cena para mim. Aqui está o que detectei: {contexto}. Por favor, me descreva o ambiente de forma natural:"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.4,  # Um pouco mais criativo
            )
            
            descricao = response.choices[0].message.content.strip()
            logger.info(f"✅ Descrição natural gerada")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição natural: {e}")
            return self._gerar_descricao_fallback(contador_objetos, total_pessoas, faces_conhecidas)

    # ========== MÉTODOS PRINCIPAIS PARA AS ROTAS ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat com IA sobre a imagem
        Responde tanto perguntas sobre a imagem quanto perguntas gerais
        """
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Classificar o tipo de pergunta
        tipo_pergunta = self._classificar_tipo_pergunta(pergunta)
        logger.info(f"🔍 Pergunta classificada como: {tipo_pergunta}")
        
        if tipo_pergunta == "sobre_imagem":
            # Pergunta sobre a imagem - usar dados detectados
            resposta = self._responder_sobre_imagem_melhorada(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
        else:
            # Pergunta geral - usar OpenAI com respostas mais curtas
            resposta = self._responder_pergunta_geral_curta(pergunta)
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
            'dados_utilizados': self._formatar_dados_utilizados(objetos_detectados, faces_nomes) if correlacao else "Pergunta geral sobre conhecimento"
        }

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """
        PARA ROTA /estatistica - Dados técnicos detalhados
        Retorna métricas, precisões e logs para análise
        """
        logger.info("📊 Gerando estatísticas técnicas")
        
        start_time = time.time()
        
        # Processar objetos detectados
        objetos_processados = self._processar_objetos_estatisticas(objetos_detectados)
        faces_processadas = self._processar_faces_estatisticas(faces_detectadas or [])
        
        # Calcular métricas de precisão (SEM NUMPY)
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

    # ========== MÉTODOS DE RESPOSTA MELHORADOS ==========

    def _classificar_tipo_pergunta(self, pergunta):
        """Classifica se a pergunta é sobre a imagem ou geral - MELHORADO"""
        pergunta_lower = pergunta.lower().strip()
        
        # Palavras-chave FORTES que indicam pergunta sobre a imagem
        palavras_fortes_imagem = [
            'essa imagem', 'esta foto', 'nesta imagem', 'nesta foto',
            'na foto', 'na imagem', 'nesta cena', 'na cena',
            'foto', 'imagem', 'fotografia', 'cena', 'cenário',
            'vejo', 'vê', 'está vendo', 'consegue ver',
            'o que tem', 'quem está', 'onde está', 'tem ', 'há ',
            'mostre', 'mostrar', 'identifique', 'reconhece',
            'descreva', 'descrever', 'analise', 'analisar'
        ]
        
        # Palavras-chave MODERADAS
        palavras_moderadas_imagem = [
            'quantos', 'quantas', 'existe', 'existem',
            'pessoa', 'pessoas', 'gente', 'humano',
            'objeto', 'objetos', 'coisa', 'coisas',
            'ambiente', 'lugar', 'local', 'sala', 'quarto'
        ]
        
        # Palavras-chave GERAIS (sobre conhecimento)
        palavras_gerais = [
            'o que é', 'como funciona', 'qual é', 'quem foi',
            'história de', 'significado de', 'definição de',
            'explique', 'explicar', 'conceito de'
        ]
        
        # Verificar perguntas gerais primeiro (mais específicas)
        for palavra in palavras_gerais:
            if palavra in pergunta_lower:
                return "geral"
        
        # Verificar perguntas FORTES sobre imagem
        for palavra in palavras_fortes_imagem:
            if palavra in pergunta_lower:
                return "sobre_imagem"
        
        # Se a pergunta tem palavras moderadas + contexto visual
        palavras_moderadas_presentes = any(palavra in pergunta_lower for palavra in palavras_moderadas_imagem)
        
        # Perguntas como "Quantas pessoas?" sem contexto são ambíguas
        # Mas no nosso sistema, se o usuário mandou imagem, provavelmente é sobre ela
        if palavras_moderadas_presentes:
            # Se parece pergunta sobre quantidades/identificação
            if any(palavra in pergunta_lower for palavra in ['quantos', 'quantas', 'tem ', 'há ', 'existe']):
                return "sobre_imagem"
        
        # Fallback: usar classificação inteligente se disponível
        return self._classificar_com_ia(pergunta) if self.client else "sobre_imagem"

    def _responder_sobre_imagem_melhorada(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas sobre a imagem - VERSÃO CORRIGIDA E MELHORADA"""
        # DEBUG: Ver o que está chegando
        print(f"\n🔍 DEBUG interpreter - objetos recebidos: {[o.get('name') for o in (objetos_detectados or [])]}")
        print(f"🔍 DEBUG interpreter - faces recebidas: {faces_nomes}")
        
        # Filtrar objetos relevantes
        objetos_filtrados = []
        if objetos_detectados:
            for obj in objetos_detectados:
                nome = obj.get('name', '')
                # Usar contagem se disponível
                count = obj.get('count', 1)
                # Adicionar múltiplas vezes se count > 1
                for _ in range(count):
                    objetos_filtrados.append(nome)
        
        print(f"🔍 DEBUG interpreter - objetos filtrados: {objetos_filtrados}")
        
        # Contar ocorrências CORRETAMENTE
        contador_objetos = Counter(objetos_filtrados)
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        print(f"🔍 DEBUG interpreter - contador: {dict(contador_objetos)}")
        print(f"🔍 DEBUG interpreter - total pessoas: {total_pessoas}")
        
        # Se temos OpenAI, usar para resposta melhorada
        if self.client:
            return self._responder_com_ia_melhorada_nova(pergunta, contador_objetos, total_pessoas, faces_conhecidas)
        else:
            return self._responder_base_simples_nova(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _responder_com_ia_melhorada_nova(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta com IA - VERSÃO COMPLETAMENTE NOVA E MELHORADA"""
        try:
            # Construir contexto SIMPLES e CLARO
            contexto_partes = []
            
            # Pessoas
            pessoas_yolo = contador_objetos.get('person', 0)
            total_pessoas_detectadas = max(total_pessoas, pessoas_yolo)
            
            if total_pessoas_detectadas > 0:
                if faces_conhecidas:
                    contexto_partes.append(f"Pessoas presentes: {', '.join(faces_conhecidas)}")
                else:
                    contexto_partes.append(f"{total_pessoas_detectadas} pessoa{'s' if total_pessoas_detectadas > 1 else ''}")
            
            # Objetos (excluir 'person' que já contamos como pessoas)
            outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
            if outros_objetos:
                objetos_desc = []
                for obj_ingles, quantidade in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj_ingles)
                    objetos_desc.append(f"{quantidade} {obj_pt}{'s' if quantidade > 1 else ''}")
                contexto_partes.append(f"Objetos detectados: {', '.join(objetos_desc)}")
            
            contexto_str = ". ".join(contexto_partes) if contexto_partes else "A imagem parece ter poucos elementos detectáveis"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""VOCÊ É UM ASSISTENTE VISUAL QUE ESTÁ ANALISANDO UMA IMAGEM REAL AGORA MESMO.

                    INFORMAÇÕES QUE VOCÊ DETECTOU NA IMAGEM (baseado em análise computacional):
                    {contexto_str}

                    REGRAS ABSOLUTAS PARA SUAS RESPOSTAS:
                    1. VOCÊ ESTÁ VENDO ESTA IMAGEM AGORA - nunca diga que não pode ver imagens
                    2. Use linguagem natural: "Na imagem que estou analisando...", "Pelo que consigo ver..."
                    3. Seja útil para pessoas com deficiência visual
                    4. Baseie-se APENAS nas informações detectadas acima
                    5. Se algo não foi detectado, diga de forma natural: "Não estou vendo..." ou "Não detectei..."

                    EXEMPLOS CORRETOS:
                    ❌ ERRADO: "Não consigo visualizar imagens"
                    ✅ CORRETO: "Na imagem que estou analisando, vejo 2 pessoas"
                    
                    ❌ ERRADO: "Não tenho dados sobre objetos"
                    ✅ CORRETO: "Não estou detectando objetos específicos na imagem"

                    Lembre: você está ajudando alguém a "ver" através da sua descrição!"""
                },
                {
                    "role": "user", 
                    "content": pergunta
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=180,
                temperature=0.2,  # Mais objetivo
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # CORREÇÃO AUTOMÁTICA: Se a resposta contiver frases problemáticas, corrigir
            frases_problematicas = [
                "não consigo visualizar",
                "não posso ver imagens", 
                "não tenho acesso a imagens",
                "como um modelo de texto",
                "não tenho dados sobre",
                "não posso analisar imagens"
            ]
            
            resposta_lower = resposta.lower()
            for frase in frases_problematicas:
                if frase in resposta_lower:
                    # Substituir por resposta manual melhor
                    return self._criar_resposta_manual_melhorada(pergunta, contexto_str, contador_objetos, total_pessoas_detectadas)
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._criar_resposta_manual_melhorada(pergunta, "erro técnico", contador_objetos, max(total_pessoas, contador_objetos.get('person', 0)))

    def _criar_resposta_manual_melhorada(self, pergunta, contexto, contador_objetos, total_pessoas):
        """Cria resposta manual quando a IA falha ou precisa corrigir"""
        pergunta_lower = pergunta.lower()
        
        # Calcular totais
        pessoas_yolo = contador_objetos.get('person', 0)
        total_final = max(total_pessoas, pessoas_yolo)
        
        # Outros objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        if "quantas pessoas" in pergunta_lower:
            if total_final > 0:
                return f"Na imagem que estou analisando, vejo {total_final} pessoa{'s' if total_final > 1 else ''}."
            return "Estou analisando a imagem, mas não estou detectando pessoas no momento."
        
        elif "o que tem" in pergunta_lower or "descreva" in pergunta_lower:
            partes = []
            if total_final > 0:
                partes.append(f"{total_final} pessoa{'s' if total_final > 1 else ''}")
            
            if outros_objetos:
                for obj, qtd in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj)
                    partes.append(f"{qtd} {obj_pt}{'s' if qtd > 1 else ''}")
            
            if partes:
                return f"Analisando a imagem: vejo {', '.join(partes)}."
            return "Estou examinando a imagem, mas parece um ambiente com poucos elementos visíveis."
        
        elif "objetos" in pergunta_lower or "identifica" in pergunta_lower:
            if outros_objetos:
                lista = []
                for obj, qtd in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj)
                    lista.append(f"{qtd} {obj_pt}{'s' if qtd > 1 else ''}")
                return f"Na imagem, identifico: {', '.join(lista)}."
            return "No momento, não estou detectando objetos específicos na imagem."
        
        elif "ambiente" in pergunta_lower or "lugar" in pergunta_lower or "intern" in pergunta_lower or "extern" in pergunta_lower:
            # Tentar inferir baseado nos objetos
            if outros_objetos:
                objetos_chave = [obj for obj in outros_objetos.keys() if obj in ['chair', 'table', 'bed', 'couch', 'tv', 'computer']]
                if objetos_chave:
                    return "Pela disposição dos objetos, parece um ambiente interno, como uma sala ou quarto."
            return "É difícil determinar sem mais detalhes, mas pela análise geral, parece um espaço comum."
        
        elif "cor" in pergunta_lower:
            return "Como estou analisando através de detecção de objetos, não consigo identificar cores específicas. Foco em identificar pessoas e objetos."
        
        else:
            # Resposta genérica melhorada
            if total_final > 0:
                return f"Estou analisando uma imagem que você enviou. Vejo {total_final} pessoa{'s' if total_final > 1 else ''}."
            elif outros_objetos:
                obj_list = list(outros_objetos.keys())[:2]
                obj_pt = [self._traduzir_objeto(obj) for obj in obj_list]
                return f"Analisando sua imagem: detecto alguns objetos como {', '.join(obj_pt)}."
            else:
                return "Estou processando a imagem que você enviou. No momento, não estou detectando muitos elementos específicos."

    def _responder_base_simples_nova(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta base simples sem IA - VERSÃO NOVA"""
        pergunta_lower = pergunta.lower()
        
        # Calcular total de pessoas (YOLO + face recognition)
        pessoas_yolo = contador_objetos.get('person', 0)
        total = max(total_pessoas, pessoas_yolo)
        
        # Outros objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        if "quantas pessoas" in pergunta_lower:
            if total > 0:
                if faces_conhecidas:
                    return f"Na imagem que estou vendo, há {total} pessoa{'s' if total > 1 else ''} (incluindo {', '.join(faces_conhecidas)})."
                return f"Analisando a imagem, vejo {total} pessoa{'s' if total > 1 else ''}."
            return "Estou analisando a imagem, mas não estou vendo pessoas."
        
        elif "o que tem" in pergunta_lower or "descreva" in pergunta_lower or "o que você vê" in pergunta_lower:
            partes = []
            if total > 0:
                if faces_conhecidas:
                    partes.append(f"reconheço {', '.join(faces_conhecidas)}")
                else:
                    partes.append(f"{total} pessoa{'s' if total > 1 else ''}")
            
            if outros_objetos:
                for obj, qtd in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj)
                    partes.append(f"{qtd} {obj_pt}{'s' if qtd > 1 else ''}")
            
            if partes:
                return f"Na imagem que estou analisando: {', '.join(partes)}."
            return "Estou examinando a imagem, mas não estou detectando muitos elementos no momento."
        
        elif "objetos" in pergunta_lower or "identifica" in pergunta_lower:
            if outros_objetos:
                lista = []
                for obj, qtd in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj)
                    lista.append(f"{qtd} {obj_pt}{'s' if qtd > 1 else ''}")
                return f"Identifico estes objetos: {', '.join(lista)}."
            return "No momento, não estou vendo objetos específicos na imagem."
        
        else:
            # Resposta genérica
            if total > 0:
                return f"Estou analisando uma imagem com {total} pessoa{'s' if total > 1 else ''}."
            elif outros_objetos:
                primeiro_obj = list(outros_objetos.keys())[0]
                obj_pt = self._traduzir_objeto(primeiro_obj)
                qtd = outros_objetos[primeiro_obj]
                return f"Analisando sua imagem: vejo {qtd} {obj_pt}{'s' if qtd > 1 else ''}, entre outros elementos."
            else:
                return "Analisando a imagem que você enviou. Parece um ambiente com poucos elementos detectáveis."

    def _responder_pergunta_geral_curta(self, pergunta):
        """Responde perguntas gerais sobre o mundo - VERSÃO MAIS CURTA"""
        if not self.client:
            return "No momento, estou focado em ajudar com a análise de imagens. Podemos conversar sobre o que estou vendo?"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Você é um assistente útil que responde perguntas gerais, mas de forma CONCISA.
                    
                    REGRAS:
                    1. Seja breve e direto ao ponto (máximo 2-3 frases)
                    2. Foque no essencial da pergunta
                    3. Se for muito complexo, sugira simplificar
                    4. Lembre que o usuário pode preferir voltar à análise da imagem
                    
                    Exemplo:
                    ❌ Muito longo: explicação de 10 frases
                    ✅ Ideal: 1-2 frases claras e objetivas
                    
                    Se a pergunta for sobre algo muito complexo, diga apenas o básico."""
                },
                {
                    "role": "user", 
                    "content": f"Responda de forma breve e objetiva (máximo 2 frases): {pergunta}"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=80,  # Limitar bastante
                temperature=0.1,  # Mais objetivo
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # Se a resposta for muito longa, encurtar
            sentences = resposta.split('. ')
            if len(sentences) > 2:
                resposta = '. '.join(sentences[:2]) + '.'
            
            logger.info(f"💬 Resposta geral (curta): {resposta[:50]}...")
            return resposta
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return "Podemos focar na análise da imagem que você enviou?"

    # ========== MÉTODOS AUXILIARES MELHORADOS ==========

    def _construir_contexto_descricao(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto para a descrição natural - MELHORADO"""
        partes = []
        
        # Informações sobre pessoas (mais natural)
        if total_pessoas > 0:
            if faces_conhecidas:
                nomes = ", ".join(faces_conhecidas)
                if len(faces_conhecidas) == total_pessoas:
                    partes.append(f"reconheço {nomes}")
                else:
                    partes.append(f"vejo {total_pessoas} pessoas, incluindo {nomes}")
            else:
                if total_pessoas == 1:
                    partes.append("uma pessoa")
                else:
                    partes.append(f"{total_pessoas} pessoas")
        
        # Informações sobre objetos (agrupados por tipo)
        if contador_objetos:
            objetos_agrupados = {}
            for obj_ingles, quantidade in contador_objetos.items():
                categoria = self._classificar_categoria(obj_ingles)
                if categoria not in objetos_agrupados:
                    objetos_agrupados[categoria] = 0
                objetos_agrupados[categoria] += quantidade
            
            for categoria, total in objetos_agrupados.items():
                if total == 1:
                    partes.append(f"um {categoria}")
                else:
                    partes.append(f"{total} {categoria}s")
        
        if not partes:
            return "poucos elementos visíveis, talvez um ambiente simples"
        
        # Formatar de forma natural
        if len(partes) == 1:
            return partes[0]
        elif len(partes) == 2:
            return f"{partes[0]} e {partes[1]}"
        else:
            return ", ".join(partes[:-1]) + f", e {partes[-1]}"

    def _construir_contexto_para_resposta(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto específico para respostas a perguntas"""
        partes = []
        
        if total_pessoas > 0:
            if faces_conhecidas:
                partes.append(f"Pessoas presentes: {', '.join(faces_conhecidas)}")
            else:
                partes.append(f"{total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}")
        
        objetos_detectados = []
        for obj_ingles, quantidade in contador_objetos.items():
            obj_pt = self._traduzir_objeto(obj_ingles)
            if quantidade == 1:
                objetos_detectados.append(f"1 {obj_pt}")
            else:
                objetos_detectados.append(f"{quantidade} {obj_pt}s")
        
        if objetos_detectados:
            partes.append(f"Objetos: {', '.join(objetos_detectados)}")
        
        if not partes:
            return "A imagem parece ter poucos elementos detectáveis."
        
        return "Na imagem analisada: " + ". ".join(partes)

    def _construir_descricao_natural_dados(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói descrição natural dos dados para respostas"""
        partes = []
        
        if total_pessoas > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"vejo {faces_conhecidas[0]}")
                else:
                    partes.append(f"vejo {', '.join(faces_conhecidas[:-1])} e {faces_conhecidas[-1]}")
            else:
                partes.append(f"vejo {total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}")
        
        objetos_principais = list(contador_objetos.items())[:3]
        if objetos_principais:
            objetos_desc = []
            for obj_ingles, quantidade in objetos_principais:
                obj_pt = self._traduzir_objeto(obj_ingles)
                if quantidade == 1:
                    objetos_desc.append(f"um {obj_pt}")
                else:
                    objetos_desc.append(f"{quantidade} {obj_pt}s")
            
            partes.append("também estou vendo " + ", ".join(objetos_desc))
        
        if not partes:
            return "não estou detectando muitos elementos específicos"
        
        return ", ".join(partes) + "."

    # ========== MÉTODOS AUXILIARES EXISTENTES (mantenha estes) ==========

    def _filtrar_objetos_relevantes(self, objetos_detectados):
        """Filtra apenas objetos que conhecemos"""
        objetos_filtrados = []
        todos_objetos = [obj for lista in self.objetos_conhecidos.values() for obj in lista]
        
        for obj in objetos_detectados:
            if obj.lower() in todos_objetos:
                objetos_filtrados.append(obj)
        
        return objetos_filtrados

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objeto do inglês para português"""
        for pt, en_list in self.objetos_conhecidos.items():
            if objeto_ingles.lower() in en_list:
                return pt
        return objeto_ingles

    def _classificar_categoria(self, objeto_ingles):
        """Classifica objeto em categoria"""
        categorias = {
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'dining table', 'desk'],
            'pessoas': ['person', 'people', 'human'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'mobile phone', 'monitor'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'plate', 'bowl', 'fork', 'knife', 'spoon'],
            'animais': ['dog', 'cat', 'bird'],
            'veículos': ['car', 'bicycle', 'motorcycle'],
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe'],
            'banheiro': ['toilet', 'sink'],
            'portas/janelas': ['door', 'window'],
            'plantas': ['potted plant', 'flower']
        }
        
        for categoria, objetos in categorias.items():
            if objeto_ingles.lower() in objetos:
                return categoria
        return "outros"

    def _classificar_com_ia(self, pergunta):
        """Usa IA para classificar perguntas ambíguas"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Classifique se a pergunta do usuário é 'sobre_imagem' (sobre a imagem que ele enviou) ou 'geral' (sobre conhecimento do mundo). O usuário SEMPRE enviou uma imagem antes de perguntar. Responda APENAS com 'sobre_imagem' ou 'geral'."
                },
                {
                    "role": "user", 
                    "content": f"O usuário enviou uma imagem e perguntou: '{pergunta}'. Esta pergunta é sobre a imagem ou é geral?"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            
            classificacao = response.choices[0].message.content.strip().lower()
            return "sobre_imagem" if "imagem" in classificacao else "geral"
                
        except Exception as e:
            logger.error(f"❌ Erro ao classificar pergunta com IA: {e}")
            return "sobre_imagem"  # Por padrão, assume que é sobre a imagem

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados para resposta"""
        if not objetos_detectados and not faces_nomes:
            return "Imagem analisada, mas poucos elementos detectados"
        
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        faces_conhecidas = len([n for n in (faces_nomes or []) if n != 'Desconhecido'])
        
        return f"{objetos_count} objetos e {faces_count} faces analisadas ({faces_conhecidas} conhecidas)"

    # ========== MÉTODOS DE ESTATÍSTICAS (mantenha do seu código anterior) ==========

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
        """Calcula métricas de precisão detalhadas (SEM NUMPY)"""
        confiancas_objetos = [obj.get('confidence', 0) for obj in objetos_detectados]
        confiancas_faces = [face.get('confidence', 0) for face in (faces_detectadas or [])]
        
        # Cálculos manuais sem numpy
        def calcular_media(lista):
            return sum(lista) / len(lista) if lista else 0
        
        def calcular_desvio_padrao(lista):
            if not lista:
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
        
        # Análise de objetos
        if total_objetos > 0:
            categorias = Counter([obj['categoria'] for obj in objetos_processados])
            categoria_principal = categorias.most_common(1)[0][0] if categorias else "diversos"
            
            # Calcular quantidade total de itens (considerando count)
            total_itens = sum(obj.get('quantidade', 1) for obj in objetos_processados)
            
            analise.append(f"Ambiente com {total_itens} itens detectados ({total_objetos} tipos), predominância de {categoria_principal}")
        
        # Análise de faces
        if total_faces > 0:
            faces_conhecidas = len([f for f in faces_processadas if f['tipo'] == 'conhecida'])
            if faces_conhecidas > 0:
                analise.append(f"{faces_conhecidas} face(s) conhecida(s) identificada(s)")
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

    def _gerar_logs_diagnostico(self, objetos_detectados, faces_detectadas):
        """Gera logs para diagnóstico técnico"""
        logs = []
        
        # Log de distribuição de confiança
        confiancas_obj = [obj.get('confidence', 0) for obj in objetos_detectados]
        if confiancas_obj:
            logs.append(f"Confiança objetos: min={min(confiancas_obj):.2f}, max={max(confiancas_obj):.2f}, avg={sum(confiancas_obj)/len(confiancas_obj):.2f}")
        
        # Log de tipos de objetos
        tipos_objetos = Counter([obj.get('name', 'desconhecido') for obj in objetos_detectados])
        if tipos_objetos:
            logs.append(f"Tipos objetos detectados: {dict(tipos_objetos)}")
        
        # Log de faces
        if faces_detectadas:
            faces_conhecidas = len([f for f in faces_detectadas if f.get('name', 'Desconhecido') != 'Desconhecido'])
            logs.append(f"Faces: {faces_conhecidas} conhecidas, {len(faces_detectadas) - faces_conhecidas} desconhecidas")
        
        return logs

    def _gerar_descricao_fallback(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição fallback"""
        partes = []
        
        if total_pessoas > 0:
            if faces_conhecidas:
                partes.append(f"Reconheço {', '.join(faces_conhecidas)}")
            else:
                partes.append(f"Vejo {total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}")
        
        objetos_principais = list(contador_objetos.items())[:3]
        if objetos_principais:
            obj_desc = []
            for obj_ingles, quantidade in objetos_principais:
                obj_pt = self._traduzir_objeto(obj_ingles)
                obj_desc.append(f"{quantidade} {obj_pt}{'s' if quantidade > 1 else ''}")
            
            partes.append("também vejo " + ", ".join(obj_desc))
        
        if not partes:
            return "Estou analisando o ambiente, mas parece um espaço com poucos elementos visíveis no momento."
        
        return " ".join(partes) + "."

    def _classificar_nivel_confianca(self, confianca):
        """Classifica nível de confiança"""
        if confianca >= 0.8:
            return "alta"
        elif confianca >= 0.5:
            return "media"
        else:
            return "baixa"