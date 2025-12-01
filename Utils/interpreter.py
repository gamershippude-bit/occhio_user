"""
Interpreter - VERSÃO NATURAL como um humano descrevendo
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
            'quadro': ['picture', 'painting']
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
        PARA ROTA /processar - Gera descrição natural como um humano
        """
        logger.info("🌄 Gerando descrição natural do ambiente")
        
        if not self.client:
            return "Vou descrever o ambiente para você. Parece um espaço bem comum."
        
        # Preparar dados das detecções
        objetos_filtrados = self._filtrar_objetos_relevantes(
            [obj.get('name', '') for obj in (objetos_detectados or [])]
        )
        
        contador_objetos = Counter(objetos_filtrados)
        total_pessoas = len(faces_nomes or [])
        faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
        
        # Se temos poucos dados, usar resposta simples
        if not objetos_filtrados and total_pessoas == 0:
            return "Olha, parece um ambiente bem simples ou vazio. Não estou identificando muitos objetos ou pessoas."
        
        try:
            # Construir contexto simples
            contexto = self._construir_contexto_simples(contador_objetos, total_pessoas, faces_conhecidas)
            
            messages = [
                {
                    "role": "system",
                    "content": """Você é uma pessoa descrevendo um ambiente para um amigo com deficiência visual.
                    
                    ESTILO DE FALA:
                    - Natural, como em uma conversa
                    - Use expressões do dia a dia
                    - Seja acolhedor e útil
                    - Não seja muito técnico
                    - Fale como se estivesse realmente observando
                    
                    COMO COMEÇAR:
                    ✅ "Olha, pelo que dá pra perceber..."
                    ✅ "Então, vou te contar o que tem por aqui..."
                    ✅ "Parece que estamos em um..."
                    ✅ "Vou descrever o que estou vendo..."
                    ✅ "Aqui tem..."
                    
                    DESCREVA NATURALMENTE:
                    "Tem uma mesa com duas cadeiras na sala"
                    "Vejo o João e a Maria conversando"
                    "Aqui no canto tem um sofá bem confortável"
                    "Tem uma TV na parede e algumas plantas"
                    
                    EVITE:
                    ❌ "Na análise foi detectado..."
                    ❌ "O sistema identificou..."
                    ❌ "Os dados mostram..."
                    ❌ Linguagem técnica ou robótica"""
                },
                {
                    "role": "user", 
                    "content": f"Descreva este ambiente naturalmente: {contexto}"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=220,
                temperature=0.7,  # Mais criativo e natural
            )
            
            descricao = response.choices[0].message.content.strip()
            
            # Remover qualquer linguagem técnica que possa ter escapado
            descricao = self._tornar_descricao_mais_humana(descricao)
            
            logger.info(f"✅ Descrição natural gerada")
            return descricao
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição natural: {e}")
            return self._gerar_descricao_simples(contador_objetos, total_pessoas, faces_conhecidas)

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """
        PARA ROTA /perguntar - Chat natural sobre a imagem
        """
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Classificar o tipo de pergunta
        tipo_pergunta = self._classificar_tipo_pergunta(pergunta)
        logger.info(f"🔍 Pergunta classificada como: {tipo_pergunta}")
        
        if tipo_pergunta == "sobre_imagem":
            # Pergunta sobre a imagem - usar dados detectados
            resposta = self._responder_sobre_imagem_natural(pergunta, objetos_detectados, faces_nomes)
            correlacao = True
        else:
            # Pergunta geral
            resposta = self._responder_pergunta_geral_natural(pergunta)
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

    # ========== MÉTODO NATURAL PARA RESPONDER SOBRE IMAGEM ==========

    def _responder_sobre_imagem_natural(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde de forma natural, como um humano"""
        # Preparar dados
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
        
        # Se temos OpenAI, usar para resposta natural
        if self.client:
            return self._responder_com_ia_natural(pergunta, contador_objetos, total_pessoas, faces_conhecidas)
        else:
            return self._responder_base_natural(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _responder_com_ia_natural(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta natural usando IA"""
        try:
            # Construir contexto em linguagem natural
            contexto = self._construir_contexto_natural(contador_objetos, total_pessoas, faces_conhecidas)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""Você é uma pessoa observando um ambiente e respondendo perguntas sobre ele.

                    O QUE VOCÊ OBSERVA NO AMBIENTE:
                    {contexto}

                    SEU ESTILO DE FALA:
                    - Fale como um amigo descrevendo algo
                    - Use linguagem casual e natural
                    - Não seja robótico ou técnico
                    - Admita quando não souber algo
                    - Seja útil e acolhedor

                    COMO RESPONDER:
                    ✅ "Olha, tem algumas pessoas aqui..."
                    ✅ "Pelo que dá pra ver, tem uma mesa e cadeiras..."
                    ✅ "Acho que é um ambiente de..."
                    ✅ "Não estou vendo muito bem, mas parece que..."
                    ✅ "Sim, tem sim! Tem..."

                    PERGUNTAS COMUNS E COMO RESPONDER:
                    "Quantas pessoas tem?" → "Tem umas 3 pessoas aqui"
                    "O que tem na imagem?" → "Olha, tem um sofá, uma TV..."
                    "Tem animais?" → "Não, não estou vendo nenhum animal"
                    "É um lugar interno ou externo?" → "Parece ser uma sala, um ambiente interno"

                    NUNCA USE linguagem técnica como "análise", "detecção", "sistema", "dados"."""
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
                temperature=0.6,  # Mais natural
            )
            
            resposta = response.choices[0].message.content.strip()
            
            # Tornar ainda mais natural se necessário
            resposta = self._tornar_resposta_mais_natural(resposta)
            
            return resposta
            
        except Exception as e:
            logger.error(f"❌ Erro ao responder com IA: {e}")
            return self._responder_base_natural(pergunta, contador_objetos, total_pessoas, faces_conhecidas)

    def _construir_contexto_natural(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto em linguagem natural"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"o {faces_conhecidas[0]} está aqui")
                else:
                    partes.append(f"tem o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
            else:
                if total_detectado == 1:
                    partes.append("tem uma pessoa")
                elif total_detectado == 2:
                    partes.append("tem duas pessoas")
                else:
                    partes.append(f"tem umas {total_detectado} pessoas")
        
        # Objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_naturais = []
            for obj_ingles, quantidade in list(outros_objetos.items())[:4]:  # Limitar
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                    objetos_naturais.append(f"{artigo} {obj_pt}")
                elif quantidade == 2:
                    objetos_naturais.append(f"duas {obj_pt}s")
                else:
                    objetos_naturais.append(f"alguns {obj_pt}s")
            
            if objetos_naturais:
                if len(objetos_naturais) == 1:
                    partes.append(f"tem {objetos_naturais[0]}")
                else:
                    partes.append(f"tem {', '.join(objetos_naturais[:-1])} e {objetos_naturais[-1]}")
        
        if not partes:
            return "o ambiente parece bem vazio ou simples"
        
        return "tem " + " e também ".join(partes)

    def _responder_base_natural(self, pergunta, contador_objetos, total_pessoas, faces_conhecidas):
        """Resposta base natural sem IA"""
        pergunta_lower = pergunta.lower()
        
        # Calcular totais
        pessoas_yolo = contador_objetos.get('person', 0)
        total = max(total_pessoas, pessoas_yolo)
        
        # Outros objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        
        # Respostas naturais para perguntas comuns
        if "quantas pessoas" in pergunta_lower:
            if total > 0:
                if faces_conhecidas:
                    return f"Olha, tem o {', '.join(faces_conhecidas)} aqui. No total, umas {total} pessoas."
                if total == 1:
                    return "Tem uma pessoa."
                elif total == 2:
                    return "Tem duas pessoas."
                else:
                    return f"Tem umas {total} pessoas."
            return "Não tem ninguém que eu consiga ver."
        
        elif "o que tem" in pergunta_lower or "descreva" in pergunta_lower or "o que você vê" in pergunta_lower:
            partes = []
            
            # Pessoas
            if total > 0:
                if faces_conhecidas:
                    if len(faces_conhecidas) == 1:
                        partes.append(f"o {faces_conhecidas[0]}")
                    else:
                        partes.append(f"o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
                else:
                    if total == 1:
                        partes.append("uma pessoa")
                    else:
                        partes.append(f"umas {total} pessoas")
            
            # Objetos
            if outros_objetos:
                objetos_desc = []
                for obj, qtd in list(outros_objetos.items())[:3]:  # Limitar a 3 objetos
                    obj_pt = self._traduzir_objeto(obj)
                    
                    if qtd == 1:
                        artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                        objetos_desc.append(f"{artigo} {obj_pt}")
                    elif qtd == 2:
                        objetos_desc.append(f"duas {obj_pt}s")
                    else:
                        objetos_desc.append(f"alguns {obj_pt}s")
                
                if objetos_desc:
                    partes.append(f"{', '.join(objetos_desc)}")
            
            if partes:
                if len(partes) == 1:
                    return f"Olha, tem {partes[0]}."
                else:
                    return f"Então, tem {', '.join(partes[:-1])} e também tem {partes[-1]}."
            return "Parece um ambiente bem vazio, não tem muita coisa."
        
        elif "objetos" in pergunta_lower or "identifica" in pergunta_lower or "tem algum objeto" in pergunta_lower:
            if outros_objetos:
                lista = []
                for obj, qtd in outros_objetos.items():
                    obj_pt = self._traduzir_objeto(obj)
                    
                    if qtd == 1:
                        artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                        lista.append(f"{artigo} {obj_pt}")
                    elif qtd == 2:
                        lista.append(f"duas {obj_pt}s")
                    else:
                        lista.append(f"vários {obj_pt}s")
                
                return f"Sim! Tem {', '.join(lista)}."
            return "Não, não estou vendo muitos objetos específicos."
        
        elif "quem está" in pergunta_lower or "tem alguém" in pergunta_lower:
            if total > 0:
                if faces_conhecidas:
                    if len(faces_conhecidas) == 1:
                        return f"Tem o {faces_conhecidas[0]} aqui."
                    else:
                        return f"Tem o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}."
                return f"Tem {total} pessoa{'s' if total > 1 else ''}, mas não reconheço quem são."
            return "Não tem ninguém que eu consiga ver."
        
        elif "ambiente" in pergunta_lower or "lugar" in pergunta_lower:
            # Inferir ambiente
            objetos_chave = list(outros_objetos.keys())
            
            if 'bed' in objetos_chave:
                return "Parece um quarto, tem uma cama."
            elif 'chair' in objetos_chave and 'table' in objetos_chave:
                return "Parece uma sala de estar ou escritório."
            elif 'toilet' in objetos_chave or 'sink' in objetos_chave:
                return "Parece um banheiro."
            elif 'car' in objetos_chave:
                return "Parece uma garagem ou rua."
            elif 'couch' in objetos_chave and 'tv' in objetos_chave:
                return "Parece uma sala de TV ou estar."
            else:
                return "É difícil dizer, mas parece um ambiente interno comum."
        
        elif "cor" in pergunta_lower:
            return "Ah, isso eu não consigo te dizer. Só consigo falar sobre o que tem no ambiente."
        
        elif "intern" in pergunta_lower or "extern" in pergunta_lower:
            objetos_chave = list(outros_objetos.keys())
            if any(obj in ['chair', 'table', 'bed', 'couch', 'tv', 'computer'] for obj in objetos_chave):
                return "Parece um ambiente interno, tipo uma sala ou quarto."
            elif 'car' in objetos_chave or 'bicycle' in objetos_chave:
                return "Parece ser externo, tipo uma rua ou garagem."
            else:
                return "É difícil saber só pelos objetos."
        
        else:
            # Resposta genérica natural
            if total > 0:
                return f"Olha, tem umas {total} pessoa{'s' if total > 1 else ''} no ambiente."
            elif outros_objetos:
                primeiro = list(outros_objetos.items())[0]
                obj_pt = self._traduzir_objeto(primeiro[0])
                
                if primeiro[1] == 1:
                    artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                    return f"Tem {artigo} {obj_pt}, entre outras coisas."
                else:
                    return f"Tem alguns {obj_pt}s no ambiente."
            else:
                return "Parece um ambiente bem simples, não tem muita coisa pra descrever."

    # ========== MÉTODO PARA PERGUNTAS GERAIS NATURAIS ==========

    def _responder_pergunta_geral_natural(self, pergunta):
        """Responde perguntas gerais de forma natural"""
        if not self.client:
            return "Podemos conversar sobre o ambiente que você está querendo entender?"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Você é uma pessoa conversando naturalmente. Responda perguntas de forma casual, como se estivesse numa conversa normal. Seja útil mas não muito formal."
                },
                {
                    "role": "user", 
                    "content": pergunta
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.5,
            )
            
            resposta = response.choices[0].message.content.strip()
            
            return resposta
                
        except Exception as e:
            logger.error(f"❌ Erro ao responder pergunta geral: {e}")
            return "Vamos focar no ambiente que você quer entender?"

    # ========== MÉTODOS AUXILIARES NATURAIS ==========

    def _construir_contexto_simples(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Constrói contexto simples para descrição"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"o {faces_conhecidas[0]}")
                else:
                    partes.append(f"o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
            else:
                partes.append(f"{total_detectado} pessoa{'s' if total_detectado > 1 else ''}")
        
        # Objetos (limitado)
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_lista = []
            for obj_ingles, quantidade in list(outros_objetos.items())[:3]:
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                    objetos_lista.append(f"{artigo} {obj_pt}")
                elif quantidade == 2:
                    objetos_lista.append(f"duas {obj_pt}s")
                else:
                    objetos_lista.append(f"alguns {obj_pt}s")
            
            if objetos_lista:
                partes.append(f"{', '.join(objetos_lista)}")
        
        if not partes:
            return "ambiente vazio"
        
        return ", ".join(partes)

    def _tornar_descricao_mais_humana(self, descricao):
        """Torna a descrição mais humana e natural"""
        # Substituir frases robóticas por naturais
        substituicoes = {
            "com base nos dados": "pelo que dá pra ver",
            "foram detectados": "tem",
            "foram identificados": "tem",
            "a análise mostra": "parece que",
            "o sistema identificou": "tem",
            "na imagem": "aqui",
            "o ambiente contém": "tem",
            "é possível observar": "dá pra ver",
            "verifica-se a presença": "tem",
            "constata-se": "tem",
            "observa-se": "tem",
            "percebe-se": "tem"
        }
        
        resultado = descricao
        for tecnico, natural in substituicoes.items():
            if tecnico in resultado.lower():
                padrao = re.compile(re.escape(tecnico), re.IGNORECASE)
                resultado = padrao.sub(natural, resultado)
        
        # Adicionar início mais natural se necessário
        inicio_natural = ["Olha,", "Então,", "Vou te contar,", "Pelo que vejo,", "Aqui"]
        if not any(resultado.startswith(inicio) for inicio in inicio_natural):
            resultado = f"Olha, {resultado.lower()}"
        
        return resultado

    def _tornar_resposta_mais_natural(self, resposta):
        """Torna a resposta mais natural"""
        substituicoes = {
            "de acordo com": "pelo que",
            "com base em": "pelo",
            "foi possível identificar": "tem",
            "a detecção revelou": "tem",
            "os resultados indicam": "parece que",
            "a imagem apresenta": "tem",
            "o processamento mostrou": "tem",
            "verificou-se": "tem",
            "constatou-se": "tem"
        }
        
        resultado = resposta
        for tecnico, natural in substituicoes.items():
            if tecnico in resultado.lower():
                padrao = re.compile(re.escape(tecnico), re.IGNORECASE)
                resultado = padrao.sub(natural, resultado)
        
        return resultado

    def _gerar_descricao_simples(self, contador_objetos, total_pessoas, faces_conhecidas):
        """Gera descrição simples e natural"""
        partes = []
        
        # Pessoas
        pessoas_yolo = contador_objetos.get('person', 0)
        total_detectado = max(total_pessoas, pessoas_yolo)
        
        if total_detectado > 0:
            if faces_conhecidas:
                if len(faces_conhecidas) == 1:
                    partes.append(f"o {faces_conhecidas[0]} está aqui")
                else:
                    partes.append(f"tem o {', '.join(faces_conhecidas[:-1])} e o {faces_conhecidas[-1]}")
            else:
                if total_detectado == 1:
                    partes.append("tem uma pessoa")
                else:
                    partes.append(f"tem umas {total_detectado} pessoas")
        
        # Objetos
        outros_objetos = {k: v for k, v in contador_objetos.items() if k != 'person'}
        if outros_objetos:
            objetos_desc = []
            for obj_ingles, quantidade in list(outros_objetos.items())[:2]:
                obj_pt = self._traduzir_objeto(obj_ingles)
                
                if quantidade == 1:
                    artigo = "um" if obj_pt[0] not in 'aeiou' else "uma"
                    objetos_desc.append(f"{artigo} {obj_pt}")
                else:
                    objetos_desc.append(f"{quantidade} {obj_pt}s")
            
            if objetos_desc:
                partes.append(f"tem {', '.join(objetos_desc)}")
        
        if not partes:
            return "Olha, parece um ambiente bem vazio ou simples. Não tem muita coisa pra descrever."
        
        return f"Então, {', '.join(partes)}."

    # ========== MÉTODOS AUXILIARES BÁSICOS ==========

    def _classificar_tipo_pergunta(self, pergunta):
        """Classifica se a pergunta é sobre a imagem ou geral"""
        pergunta_lower = pergunta.lower().strip()
        
        palavras_imagem = [
            'essa imagem', 'esta foto', 'na foto', 'na imagem',
            'o que tem', 'quem está', 'onde está', 'tem ', 'há ',
            'quantos', 'quantas', 'pessoa', 'pessoas',
            'descreva', 'analise', 'identifique', 'reconhece',
            'mostra', 'mostrar', 'o que você vê', 'que tem aí',
            'ambiente', 'lugar', 'sala', 'quarto', 'cozinha'
        ]
        
        palavras_gerais = [
            'o que é', 'como funciona', 'qual é', 'quem foi',
            'história de', 'significado de', 'definição de',
            'explique', 'explicar', 'conceito de',
            'capital de', 'população de', 'onde fica',
            'conta uma piada', 'qual o sentido da vida'
        ]
        
        # Verificar perguntas gerais
        for palavra in palavras_gerais:
            if palavra in pergunta_lower:
                return "geral"
        
        # Verificar perguntas sobre imagem
        for palavra in palavras_imagem:
            if palavra in pergunta_lower:
                return "sobre_imagem"
        
        # Se parece pergunta sobre o que está na imagem
        if '?' in pergunta and len(pergunta.split()) < 8:
            return "sobre_imagem"
        
        return "geral"

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

    # ========== MÉTODO PARA /estatistica (mantido para compatibilidade) ==========

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

    # ========== MÉTODOS DE ESTATÍSTICAS (mantidos) ==========

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
            'roupas': ['backpack', 'handbag', 'suitcase', 'tie', 'hat', 'shoe'],
            'banheiro': ['toilet', 'sink'],
            'portas/janelas': ['door', 'window'],
            'plantas': ['potted plant', 'flower'],
            'decoração': ['picture', 'painting']
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