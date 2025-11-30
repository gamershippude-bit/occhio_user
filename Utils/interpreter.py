"""
Interpreter - Versão CORRIGIDA: Todos os endpoints necessários
"""

import logging
import os
import time
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

        # Lista de objetos relevantes (apenas para filtragem)
        self.objetos_relevantes = {
            'pessoas': ['person', 'people', 'human'],
            'móveis': ['chair', 'couch', 'sofa', 'bed', 'table', 'dining table', 'desk'],
            'eletrônicos': ['laptop', 'computer', 'tv', 'television', 'cell phone', 'mobile phone', 'monitor', 'keyboard', 'mouse'],
            'utensílios': ['cup', 'bottle', 'book', 'vase', 'clock', 'umbrella', 'plate', 'bowl'],
            'animais': ['dog', 'cat', 'bird'],
            'veículos': ['car', 'bicycle', 'motorcycle'],
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe']
        }

        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY não encontrada")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)  # CORREÇÃO: removido proxies
                logger.info(f"✅ Cliente OpenAI configurado para {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Falha ao inicializar cliente OpenAI: {e}")
                self.client = None

    # ========== MÉTODOS DE PROCESSAMENTO DE DETECÇÕES ==========

    def processar_deteccoes(self, objetos_detectados, faces_detectadas=None):
        """
        Endpoint /processar
        Processa detecções e retorna objetos com coordenadas, nomes e confiabilidade
        """
        logger.info("🔄 Processando detecções para endpoint /processar")
        
        start_time = time.time()
        
        # Processar objetos detectados
        objetos_processados = []
        if objetos_detectados:
            for i, obj in enumerate(objetos_detectados, 1):
                nome_objeto = obj.get('name', 'desconhecido')
                confianca = obj.get('confidence', 0)
                bbox = obj.get('bbox', {})
                
                objeto_info = {
                    'id': i,
                    'nome': self._traduzir_objeto(nome_objeto),
                    'nome_ingles': nome_objeto,
                    'confiabilidade': f"{confianca:.1%}",
                    'confianca_decimal': confianca,
                    'coordenadas': {
                        'x': bbox.get('x', 0),
                        'y': bbox.get('y', 0),
                        'width': bbox.get('width', 0),
                        'height': bbox.get('height', 0)
                    }
                }
                objetos_processados.append(objeto_info)
        
        # Processar faces detectadas
        faces_processadas = []
        if faces_detectadas:
            for i, face in enumerate(faces_detectadas, 1):
                nome_face = face.get('name', 'Desconhecido')
                confianca_face = face.get('confidence', 0)
                bbox_face = face.get('bbox', {})
                
                face_info = {
                    'id': i,
                    'nome': nome_face,
                    'tipo': 'conhecida' if nome_face != 'Desconhecido' else 'desconhecida',
                    'confiabilidade': f"{confianca_face:.1%}",
                    'confianca_decimal': confianca_face,
                    'coordenadas': {
                        'x': bbox_face.get('x', 0),
                        'y': bbox_face.get('y', 0),
                        'width': bbox_face.get('width', 0),
                        'height': bbox_face.get('height', 0)
                    }
                }
                faces_processadas.append(face_info)
        
        processing_time = time.time() - start_time
        
        resposta = {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'resumo': {
                'total_objetos': len(objetos_processados),
                'total_faces': len(faces_processadas),
                'faces_conhecidas': len([f for f in faces_processadas if f['tipo'] == 'conhecida']),
                'faces_desconhecidas': len([f for f in faces_processadas if f['tipo'] == 'desconhecida'])
            },
            'deteccoes_detalhadas': {
                'objetos': objetos_processados,
                'faces': faces_processadas
            }
        }
        
        logger.info(f"✅ Processamento concluído: {len(objetos_processados)} objetos, {len(faces_processadas)} faces")
        return resposta

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        Endpoint /perguntar
        Recebe pergunta e retorna resposta correlacionada com a imagem ou geral
        """
        logger.info(f"❓ Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Classificar se a pergunta é sobre a imagem - CORREÇÃO: melhor prompt
        tipo_pergunta = self._classificar_pergunta(pergunta)
        
        # Filtrar objetos relevantes
        objetos_filtrados = self._filtrar_objetos_relevantes(
            [obj.get('name', '') for obj in (objetos_detectados or [])]
        ) if objetos_detectados else []
        
        if tipo_pergunta == "imagem":
            # Resposta baseada na imagem
            resposta_imagem = self._gerar_resposta_sobre_imagem(pergunta, objetos_filtrados, faces_nomes or [])
            correlacao = True
            dados_utilizados = self._formatar_dados_para_resposta(len(faces_nomes or []), Counter(objetos_filtrados))
        else:
            # Resposta geral
            resposta_imagem = self._responder_pergunta_geral(pergunta)
            correlacao = False
            dados_utilizados = "Pergunta geral - sem dados da imagem"
        
        processing_time = time.time() - start_time
        
        resposta = {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': f"{processing_time:.2f}s",
            'pergunta': pergunta,
            'correlacao_com_imagem': correlacao,
            'resposta': resposta_imagem,
            'dados_utilizados': dados_utilizados
        }
        
        logger.info(f"✅ Pergunta processada - Correlação: {correlacao}")
        return resposta

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """
        Endpoint /estatistica
        Retorna dados específicos: tudo identificado, precisão, tempo, quantidades
        """
        logger.info("📊 Gerando estatísticas detalhadas")
        
        start_time = time.time()
        
        # Contagens detalhadas
        objetos_nomes = [obj.get('name', 'desconhecido') for obj in objetos_detectados]
        contador_objetos = Counter(objetos_nomes)
        
        # Estatísticas de confiança
        confiancas_objetos = [obj.get('confidence', 0) for obj in objetos_detectados]
        confiancas_faces = [face.get('confidence', 0) for face in (faces_detectadas or [])]
        
        # Processar métricas
        estatisticas = {
            'contagens': {
                'total_objetos': len(objetos_detectados),
                'total_faces': len(faces_detectadas or []),
                'objetos_por_tipo': {
                    self._traduzir_objeto(nome): quantidade 
                    for nome, quantidade in contador_objetos.items()
                },
                'faces_conhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') != 'Desconhecido']),
                'faces_desconhecidas': len([f for f in (faces_detectadas or []) if f.get('name', 'Desconhecido') == 'Desconhecido'])
            },
            'precisao': {
                'confianca_media_objetos': f"{sum(confiancas_objetos) / len(confiancas_objetos):.1%}" if confiancas_objetos else "0%",
                'confianca_media_faces': f"{sum(confiancas_faces) / len(confiancas_faces):.1%}" if confiancas_faces else "0%",
                'confianca_maxima_objetos': f"{max(confiancas_objetos):.1%}" if confiancas_objetos else "0%",
                'confianca_minima_objetos': f"{min(confiancas_objetos):.1%}" if confiancas_objetos else "0%"
            },
            'deteccoes_detalhadas': {
                'objetos': [
                    {
                        'nome': self._traduzir_objeto(obj.get('name', 'desconhecido')),
                        'nome_ingles': obj.get('name', 'desconhecido'),
                        'confianca': f"{obj.get('confidence', 0):.1%}",
                        'coordenadas': obj.get('bbox', {})
                    }
                    for obj in objetos_detectados
                ],
                'faces': [
                    {
                        'nome': face.get('name', 'Desconhecido'),
                        'tipo': 'conhecida' if face.get('name', 'Desconhecido') != 'Desconhecido' else 'desconhecida',
                        'confianca': f"{face.get('confidence', 0):.1%}",
                        'coordenadas': face.get('bbox', {})
                    }
                    for face in (faces_detectadas or [])
                ]
            }
        }
        
        processing_time = time.time() - start_time
        estatisticas['tempo_processamento'] = f"{processing_time:.2f}s"
        estatisticas['timestamp'] = time.time()
        
        logger.info(f"✅ Estatísticas geradas: {estatisticas['contagens']['total_objetos']} objetos, {estatisticas['contagens']['total_faces']} faces")
        return estatisticas

    def processamento_completo(self, objetos_detectados, faces_detectadas=None, pergunta=None):
        """
        Endpoint /completo
        Junta todos os processamentos em sequência
        """
        logger.info("🎯 Iniciando processamento completo")
        
        start_time = time.time()
        
        # Executar todos os processamentos
        resultado_processar = self.processar_deteccoes(objetos_detectados, faces_detectadas)
        resultado_estatisticas = self.obter_estatisticas(objetos_detectados, faces_detectadas)
        
        # Se há pergunta, incluir também
        resultado_pergunta = None
        if pergunta:
            faces_nomes = [face.get('name', 'Desconhecido') for face in (faces_detectadas or [])]
            resultado_pergunta = self.perguntar_sobre_imagem(pergunta, objetos_detectados, faces_nomes)
        
        processing_time = time.time() - start_time
        
        # Consolidar resultados
        resposta_completa = {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_total_processamento': f"{processing_time:.2f}s",
            'processar': resultado_processar,
            'estatisticas': resultado_estatisticas,
        }
        
        if resultado_pergunta:
            resposta_completa['perguntar'] = resultado_pergunta
        
        # Adicionar resumo inteligente
        resposta_completa['resumo_inteligente'] = self._gerar_resumo_inteligente(
            resultado_estatisticas['contagens'],
            resultado_pergunta['resposta'] if resultado_pergunta else None
        )
        
        logger.info("✅ Processamento completo finalizado")
        return resposta_completa

    # ========== MÉTODOS AUXILIARES ==========

    def _filtrar_objetos_relevantes(self, objetos_detectados):
        """Filtra objetos - APENAS os que sabemos identificar"""
        if not objetos_detectados:
            return []
        
        objetos_filtrados = []
        
        for obj in objetos_detectados:
            obj_lower = obj.lower().strip()
            
            for categoria, objetos in self.objetos_relevantes.items():
                if obj_lower in objetos:
                    objetos_filtrados.append(obj)
                    break
        
        return objetos_filtrados

    def _traduzir_objeto(self, objeto_ingles):
        """Traduz objetos do inglês para português"""
        traducoes = {
            'person': 'pessoa', 'people': 'pessoas', 'human': 'pessoa',
            'chair': 'cadeira', 'couch': 'sofá', 'sofa': 'sofá', 'bed': 'cama',
            'table': 'mesa', 'dining table': 'mesa de jantar', 'desk': 'escritório',
            'laptop': 'laptop', 'computer': 'computador', 'tv': 'televisão',
            'television': 'televisão', 'cell phone': 'celular', 'mobile phone': 'celular',
            'monitor': 'monitor', 'keyboard': 'teclado', 'mouse': 'mouse',
            'cup': 'copo', 'bottle': 'garrafa', 'book': 'livro', 'vase': 'vaso',
            'clock': 'relógio', 'umbrella': 'guarda-chuva', 'plate': 'prato',
            'bowl': 'tigela', 'dog': 'cachorro', 'cat': 'gato', 'bird': 'pássaro',
            'car': 'carro', 'bicycle': 'bicicleta', 'motorcycle': 'motocicleta',
            'backpack': 'mochila', 'handbag': 'bolsa', 'suitcase': 'mala',
            'tie': 'gravata', 'hat': 'chapéu', 'shoe': 'sapato'
        }
        
        return traducoes.get(objeto_ingles.lower(), objeto_ingles)

    def _classificar_pergunta(self, pergunta):
        """Usa ChatGPT para classificar se a pergunta é sobre a imagem ou geral"""
        if not self.client:
            return "imagem"  # CORREÇÃO: fallback para imagem (mais comum)
            
        messages = [
            {
                "role": "system",
                "content": (
                    "Classifique se a pergunta é sobre o conteúdo visual de uma IMAGEM ou é uma pergunta GERAL.\n"
                    "IMAGEM = pergunta sobre objetos, pessoas, cena, ambiente, descrição visual, conteúdo da foto\n" 
                    "GERAL = pergunta sobre conhecimento, fatos, conceitos abstratos não relacionados a imagens\n"
                    "RESPONDA APENAS COM 'imagem' OU 'geral'"
                )
            },
            {
                "role": "user", 
                "content": f"Classifique esta pergunta: '{pergunta}'"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0,  # CORREÇÃO: temperatura zero para consistência
            )
            
            classificacao = response.choices[0].message.content.strip().lower()
            logger.info(f"🔍 ChatGPT classificou '{pergunta}' como: {classificacao}")
            
            return "imagem" if "imagem" in classificacao else "geral"
            
        except Exception as e:
            logger.error(f"❌ Erro ao classificar pergunta: {e}")
            return "imagem"  # CORREÇÃO: fallback para imagem

    def _gerar_resposta_sobre_imagem(self, pergunta, objetos_filtrados, faces_nomes):
        """Gera resposta sobre a imagem baseada nos dados detectados"""
        objetos_contador = Counter(objetos_filtrados)
        total_pessoas_faces = len(faces_nomes or [])
        objetos_sem_pessoas = {obj: count for obj, count in objetos_contador.items() if obj != 'person'}
        
        logger.info(f"🔍 Dados para resposta: {total_pessoas_faces} pessoas, {dict(objetos_sem_pessoas)} objetos")
        
        if total_pessoas_faces == 0 and not objetos_sem_pessoas:
            return "Não detectei pessoas ou objetos específicos nesta imagem."
        
        if self.client:
            dados_detectados = self._formatar_dados_para_chatgpt(total_pessoas_faces, objetos_sem_pessoas)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente para pessoas com deficiência visual. "
                        "Responda a pergunta baseado APENAS nos dados fornecidos sobre a imagem. "
                        "Seja 100% honesto e direto. Use apenas português. "
                        "NUNCA invente informações que não estão nos dados. "
                        "Seja útil e claro para quem não pode ver a imagem."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Pergunta: '{pergunta}'\n\nDados detectados na imagem:\n{dados_detectados}\n\nResposta baseada APENAS nos dados acima:"
                }
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.1,
                )
                resposta = response.choices[0].message.content.strip()
                logger.info(f"💬 Resposta gerada: {resposta}")
                return resposta
            except Exception as e:
                logger.error(f"❌ Erro ao gerar resposta: {e}")
        
        # Fallback
        return self._gerar_resposta_direta_honesta(total_pessoas_faces, objetos_sem_pessoas)

    def _formatar_dados_para_chatgpt(self, total_pessoas, objetos_contador):
        """Formata os dados detectados para o ChatGPT"""
        partes = []
        
        if total_pessoas > 0:
            if total_pessoas == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{total_pessoas} pessoas")
        
        for obj, count in objetos_contador.items():
            obj_traduzido = self._traduzir_objeto(obj)
            if count == 1:
                partes.append(f"1 {obj_traduzido}")
            else:
                partes.append(f"{count} {obj_traduzido}s")
        
        return ", ".join(partes) if partes else "nada detectado"

    def _formatar_dados_para_resposta(self, total_pessoas, objetos_contador):
        """Formata dados para resposta do endpoint /perguntar"""
        objetos_sem_pessoas = {obj: count for obj, count in objetos_contador.items() if obj != 'person'}
        
        return {
            'pessoas_detectadas': total_pessoas,
            'objetos_detectados': {
                self._traduzir_objeto(obj): count 
                for obj, count in objetos_sem_pessoas.items()
            }
        }

    def _gerar_resposta_direta_honesta(self, total_pessoas, objetos_contador):
        """Fallback honesto baseado nos dados reais"""
        partes = []
        
        if total_pessoas > 0:
            if total_pessoas == 1:
                partes.append("1 pessoa")
            else:
                partes.append(f"{total_pessoas} pessoas")
        
        for obj, count in objetos_contador.items():
            obj_traduzido = self._traduzir_objeto(obj)
            if count == 1:
                partes.append(f"1 {obj_traduzido}")
            else:
                partes.append(f"{count} {obj_traduzido}s")
        
        if not partes:
            return "Não detectei pessoas ou objetos específicos."
        
        if len(partes) == 1:
            return f"Detectei {partes[0]}."
        elif len(partes) == 2:
            return f"Detectei {partes[0]} e {partes[1]}."
        else:
            return "Detectei " + ", ".join(partes[:-1]) + " e " + partes[-1] + "."

    def _responder_pergunta_geral(self, pergunta):
        """Responde perguntas gerais usando OpenAI"""
        if not self.client:
            return "Desculpe, não posso responder perguntas gerais no momento."
        
        messages = [
            {
                "role": "system",
                "content": "Você é um assistente útil para pessoas com deficiência visual. Responda de forma clara e direta em português."
            },
            {
                "role": "user", 
                "content": pergunta
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Erro OpenAI: {e}")
            return "Desculpe, não consegui processar sua pergunta."

    def _gerar_resumo_inteligente(self, contagens, resposta_pergunta=None):
        """Gera resumo inteligente para o endpoint completo"""
        total_objetos = contagens['total_objetos']
        total_faces = contagens['total_faces']
        
        resumo = f"Análise completa: {total_objetos} objeto{'s' if total_objetos != 1 else ''} e {total_faces} pessoa{'s' if total_faces != 1 else ''} detectados."
        
        if resposta_pergunta:
            resumo += f" Resposta: {resposta_pergunta}"
            
        return resumo

    def descrever_ambiente(self, objetos_detectados=None, faces_nomes=None):
        """Descrição do ambiente (método legado)"""
        return self.perguntar_sobre_imagem("Descreva o que você vê nesta imagem", objetos_detectados, faces_nomes)