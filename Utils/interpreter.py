"""
Interpreter - Versão OTMIIZADA para as rotas específicas
"""

import logging
import os
import time
import math
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
            'bolsa': ['handbag']
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
            return "Sistema de visão ativo. Entre em contato para mais detalhes."
        
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
                    "content": """Você é um assistente visual que ajuda pessoas com deficiência visual.
                    Crie uma descrição NATURAL e FLUIDA do ambiente baseada nos dados detectados.
                    Seja descritivo, mas objetivo. Use linguagem natural como se estivesse descrevendo para alguém.
                    Inclua pessoas, objetos principais e o contexto geral.
                    Mantenha a descrição concisa mas informativa."""
                },
                {
                    "role": "user", 
                    "content": f"Com base nestas detecções: {contexto}. Por favor, descreva o ambiente de forma natural:"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.3,
            )
            
            descricao = response.choices[0].message.content.strip()
            logger.info(f"✅ Descrição natural gerada: {descricao[:50]}...")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição natural: {e}")
            return self._gerar_descricao_fallback(contador_objetos, total_pessoas, faces_conhecidas)

    def _construir_contexto_descricao(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto para a descrição natural"""
        partes = []
        
        # Informações sobre pessoas
        if total_pessoas > 0:
            if faces_conhecidas:
                nomes = ", ".join(faces_conhecidas)
                partes.append(f"Pessoas presentes: {nomes}")
            else:
                partes.append(f"{total_pessoas} pessoa{'s' if total_pessoas > 1 else ''} no ambiente")
        
        # Informações sobre objetos
        objetos_desc = []
        for obj_ingles, quantidade in contador_objetos.items():
            obj_pt = self._traduzir_objeto(obj_ingles)
            if quantidade == 1:
                objetos_desc.append(f"1 {obj_pt}")
            else:
                objetos_desc.append(f"{quantidade} {obj_pt}s")
        
        if objetos_desc:
            partes.append("Objetos detectados: " + ", ".join(objetos_desc))
        
        if not partes:
            return "Ambiente aparentemente vazio ou poucos objetos detectados"
        
        return ". ".join(partes)

    def _gerar_descricao_fallback(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição fallback caso a IA falhe"""
        partes = []
        
        if total_pessoas > 0:
            if faces_conhecidas:
                partes.append(f"Identifiquei {', '.join(faces_conhecidas)} no ambiente.")
            else:
                partes.append(f"Vejo {total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}.")
        
        objetos_principais = list(contador_objetos.items())[:3]  # Limitar a 3 objetos principais
        if objetos_principais:
            obj_desc = []
            for obj_ingles, quantidade in objetos_principais:
                obj_pt = self._traduzir_objeto(obj_ingles)
                obj_desc.append(f"{quantidade} {obj_pt}{'s' if quantidade > 1 else ''}")
            
            partes.append("Também vejo " + ", ".join(obj_desc) + ".")
        
        if not partes:
            return "Ambiente tranquilo, com poucos elementos visíveis no momento."
        
        return " ".join(partes)

    # ========== MÉTODOS PRINCIPAIS PARA AS ROTAS ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat com IA sobre a imagem
        Responde tanto perguntas sobre a imagem quanto perguntas gerais
        """
        logger.info(f"💬 Processando pergunta para chat: '{pergunta}'")
        
        start_time = time.time()
        
        # Classificar o tipo de pergunta
        tipo_pergunta = self._classificar_tipo_pergunta(pergunta)
        logger.info(f"🔍 Pergunta classificada como: {tipo_pergunta}")
        
        if tipo_pergunta == "sobre_imagem":
            # Pergunta sobre a imagem - usar dados detectados
            resposta = self._responder_sobre_imagem(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
        else:
            # Pergunta geral - usar apenas OpenAI
            resposta = self._responder_pergunta_geral(pergunta)
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

    # ========== MÉTODOS DE PROCESSAMENTO DE ESTATÍSTICAS (CORRIGIDOS) ==========

    def _processar_objetos_estatisticas(self, objetos_detectados):
        """Processa objetos para estatísticas detalhadas"""
        objetos_processados = []
        
        for i, obj in enumerate(objetos_detectados, 1):
            nome_ingles = obj.get('name', 'desconhecido')
            confianca = obj.get('confidence', 0)
            bbox = obj.get('bbox', {})
            
            objeto_info = {
                'id': i,
                'nome_pt': self._traduzir_objeto(nome_ingles),
                'nome_en': nome_ingles,
                'confianca': confianca,
                'confianca_percentual': f"{confianca:.1%}",
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
            
            analise.append(f"Ambiente com {total_objetos} objetos, predominância de {categoria_principal}")
        
        # Análise de faces
        if total_faces > 0:
            faces_conhecidas = len([f for f in faces_processadas if f['tipo'] == 'conhecida'])
            if faces_conhecidas > 0:
                analise.append(f"{faces_conhecidas} face(s) conhecida(s) identificada(s)")
            if total_faces - faces_conhecidas > 0:
                analise.append(f"{total_faces - faces_conhecidas} face(s) desconhecida(s)")
        
        return ". ".join(analise)

    # ========== MÉTODOS DE RESPOSTA INTELIGENTE ==========

    def _classificar_tipo_pergunta(self, pergunta):
        """Classifica se a pergunta é sobre a imagem ou geral"""
        pergunta_lower = pergunta.lower()
        
        # Palavras-chave que indicam pergunta sobre a imagem
        palavras_imagem = [
            'essa imagem', 'esta foto', 'nesta imagem', 'nesta foto',
            'o que tem', 'quem está', 'quantos', 'quantas', 'onde está',
            'vejo', 'vê', 'identifique', 'reconhece', 'descreva',
            'tem ', 'há ', 'existe', 'existem', 'mostre', 'mostrar'
        ]
        
        # Palavras-chave que indicam pergunta geral
        palavras_gerais = [
            'o que é', 'como funciona', 'qual é', 'quem foi',
            'história', 'explicar', 'definir', 'significado'
        ]
        
        # Verificar primeiro perguntas gerais (mais específicas)
        for palavra in palavras_gerais:
            if palavra in pergunta_lower:
                return "geral"
        
        # Verificar perguntas sobre imagem
        for palavra in palavras_imagem:
            if palavra in pergunta_lower:
                return "sobre_imagem"
        
        # Se não identificar claramente, usar classificação inteligente
        return self._classificar_com_ia(pergunta)

    def _classificar_com_ia(self, pergunta):
        """Usa IA para classificar perguntas ambíguas"""
        if not self.client:
            return "sobre_imagem"  # Fallback seguro
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Classifique se a pergunta é 'sobre_imagem' (sobre o conteúdo visual atual) ou 'geral' (sobre conhecimento do mundo). Responda APENAS com 'sobre_imagem' ou 'geral'."
                },
                {
                    "role": "user", 
                    "content": f"Classifique: '{pergunta}'"
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
            return "sobre_imagem"  # Fallback seguro

    def _responder_sobre_imagem(self, pergunta, objetos_detectados, faces_nomes):
        """Responde perguntas sobre a imagem usando dados detectados"""
        # Filtrar objetos relevantes
        objetos_relevantes = self._filtrar_objetos_relevantes(
            [obj.get('name', '') for obj in (objetos_detectados or [])]
        )
        
        # Contar ocorrências
        contador_objetos = Counter(objetos_relevantes)
        total_pessoas = len(faces_nomes or [])
        
        # Se temos OpenAI, usar para resposta contextual
        if self.client:
            return self._responder_com_ia_contextual(pergunta, contador_objetos, total_pessoas)
        else:
            # Fallback: resposta baseada em regras
            return self._responder_base_regras(pergunta, contador_objetos, total_pessoas)

    def _responder_com_ia_contextual(self, pergunta, contador_objetos, total_pessoas):
        """Resposta contextual usando OpenAI"""
        try:
            # Preparar contexto dos dados detectados
            contexto_deteccoes = self._formatar_contexto_deteccoes(contador_objetos, total_pessoas)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""
                    Você é um assistente visual que ajuda deficientes visuais.
                    Baseie sua resposta NOS DADOS DISPONÍVEIS sobre a imagem:
                    
                    {contexto_deteccoes}
                    
                    Seja direto, útil e baseado apenas nos dados fornecidos.
                    Se não houver dados relevantes, diga isso claramente.
                    Responda em português.
                    """
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
                temperature=0.1,
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._responder_base_regras(pergunta, contador_objetos, total_pessoas)

    def _responder_pergunta_geral(self, pergunta):
        """Responde perguntas gerais sobre o mundo"""
        if not self.client:
            return "Desculpe, não posso responder perguntas gerais no momento."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Você é um assistente útil. Responda perguntas gerais de forma clara e em português. Seja conciso e objetivo."
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
                temperature=0.1,
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return "Desculpe, não consegui processar sua pergunta no momento."

    # ========== MÉTODOS AUXILIARES ==========

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
        for categoria, objetos in self.objetos_conhecidos.items():
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

    def _agrupar_objetos_por_categoria(self, objetos_detectados):
        """Agrupa objetos por categoria"""
        categorias = {}
        for obj in objetos_detectados:
            nome = obj.get('name', 'desconhecido')
            categoria = self._classificar_categoria(nome)
            if categoria not in categorias:
                categorias[categoria] = 0
            categorias[categoria] += 1
        return categorias

    def _formatar_contexto_deteccoes(self, contador_objetos, total_pessoas):
        """Formata dados das detecções para contexto da IA"""
        partes = []
        
        if total_pessoas > 0:
            partes.append(f"{total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}")
        
        for obj_ingles, quantidade in contador_objetos.items():
            obj_pt = self._traduzir_objeto(obj_ingles)
            partes.append(f"{quantidade} {obj_pt}{'s' if quantidade > 1 else ''}")
        
        if not partes:
            return "Nenhum objeto ou pessoa detectado na imagem."
        
        return "Na imagem foram detectados: " + ", ".join(partes) + "."

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados para resposta"""
        if not objetos_detectados and not faces_nomes:
            return "Nenhum dado da imagem utilizado"
        
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        
        return f"{objetos_count} objetos e {faces_count} faces utilizados para resposta"

    def _responder_base_regras(self, pergunta, contador_objetos, total_pessoas):
        """Resposta baseada em regras (fallback)"""
        pergunta_lower = pergunta.lower()
        
        # Perguntas sobre quantidade de pessoas
        if any(palavra in pergunta_lower for palavra in ['quantas pessoas', 'tem gente', 'tem pessoas']):
            if total_pessoas > 0:
                return f"Detectei {total_pessoas} pessoa{'s' if total_pessoas > 1 else ''} na imagem."
            else:
                return "Não detectei pessoas na imagem."
        
        # Perguntas sobre objetos específicos
        for obj_pt, obj_en_list in self.objetos_conhecidos.items():
            if obj_pt in pergunta_lower:
                total = sum(contador_objetos.get(obj_en, 0) for obj_en in obj_en_list)
                if total > 0:
                    return f"Sim, detectei {total} {obj_pt}{'s' if total > 1 else ''}."
                else:
                    return f"Não detectei {obj_pt}s na imagem."
        
        # Resposta genérica
        if contador_objetos or total_pessoas > 0:
            return self._formatar_contexto_deteccoes(contador_objetos, total_pessoas)
        else:
            return "Não detectei objetos ou pessoas específicas nesta imagem."

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
            logs.append(f"Tipos objetos: {dict(tipos_objetos)}")
        
        # Log de faces
        if faces_detectadas:
            faces_conhecidas = len([f for f in faces_detectadas if f.get('name', 'Desconhecido') != 'Desconhecido'])
            logs.append(f"Faces: {faces_conhecidas} conhecidas, {len(faces_detectadas) - faces_conhecidas} desconhecidas")
        
        return logs