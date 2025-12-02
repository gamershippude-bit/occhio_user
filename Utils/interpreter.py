"""
Interpreter - VERSÃO ROBUSTA com fallback automático
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class Interpreter:
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        
        logger.info(f"🔧 Inicializando Interpreter com API key: {'SIM' if api_key else 'NÃO'}")
        
        # Tentar inicializar OpenAI apenas se tiver API key
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✅ Cliente OpenAI configurado para {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Falha ao inicializar cliente OpenAI: {e}")
                self.client = None
        else:
            logger.warning("⚠️ Sem API key - usando modo local")
            self.client = None

    # ========== MÉTODO PARA /processar ==========

    def gerar_descricao_natural(self, objetos_detectados=None, faces_nomes=None):
        """Gera descrição natural"""
        logger.info("🌄 Gerando descrição natural")
        
        # Se não tem cliente OpenAI, usar modo local
        if not self.client:
            return self._gerar_descricao_local(objetos_detectados, faces_nomes)
        
        try:
            # Preparar dados
            objetos_lista = []
            for obj in (objetos_detectados or []):
                nome = obj.get('name', '')
                count = obj.get('count', 1)
                objetos_lista.append(f"{count} {nome}{'s' if count > 1 else ''}")
            
            total_pessoas = len(faces_nomes or [])
            faces_conhecidas = [nome for nome in (faces_nomes or []) if nome != 'Desconhecido']
            
            # Construir prompt simples
            dados = f"Objetos: {', '.join(objetos_lista) if objetos_lista else 'nenhum'}. "
            dados += f"Pessoas: {total_pessoas}. "
            if faces_conhecidas:
                dados += f"Pessoas conhecidas: {', '.join(faces_conhecidas)}."
            
            from openai import OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Você é a Specula, assistente para pessoas com deficiência visual.
                        Dados detectados: {dados}
                        Descreva o ambiente naturalmente baseado apenas nesses dados."""
                    },
                    {"role": "user", "content": "Descreva o ambiente:"}
                ],
                max_tokens=150,
                temperature=0.5,
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar descrição com OpenAI: {e}")
            return self._gerar_descricao_local(objetos_detectados, faces_nomes)

    def _gerar_descricao_local(self, objetos_detectados=None, faces_nomes=None):
        """Descrição local"""
        objetos_contados = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', '')
            count = obj.get('count', 1)
            objetos_contados[nome] = objetos_contados.get(nome, 0) + count
        
        total_pessoas = len(faces_nomes or [])
        
        if total_pessoas > 0 or objetos_contados:
            partes = []
            
            if total_pessoas > 0:
                if total_pessoas == 1:
                    partes.append("uma pessoa")
                else:
                    partes.append(f"{total_pessoas} pessoas")
            
            for obj, qtd in objetos_contados.items():
                if qtd == 1:
                    partes.append(f"um {obj}")
                else:
                    partes.append(f"{qtd} {obj}s")
            
            if len(partes) == 1:
                return f"Tem {partes[0]}."
            else:
                return f"Tem {', '.join(partes[:-1])} e {partes[-1]}."
        else:
            return "Não estou identificando elementos específicos no ambiente. Sou a Specula, sua assistente visual!"

    # ========== MÉTODO PRINCIPAL PARA /perguntar ==========

    def perguntar_sobre_imagem(self, pergunta, objetos_detectados=None, faces_nomes=None):
        """Responde perguntas"""
        logger.info(f"💬 Processando pergunta: '{pergunta}'")
        
        start_time = time.time()
        
        # Primeiro verificar se é pergunta sobre tempo
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
        
        # Se não tem cliente OpenAI, usar modo local
        if not self.client:
            return self._responder_local(pergunta, objetos_detectados, faces_nomes, start_time)
        
        try:
            # Preparar dados da imagem
            dados_imagem = self._preparar_dados_imagem(objetos_detectados, faces_nomes)
            
            # Criar prompt inteligente
            prompt = self._criar_prompt_inteligente(pergunta, dados_imagem)
            
            from openai import OpenAI
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
            
            # Determinar tipo de pergunta
            tipo_pergunta = self._determinar_tipo_pergunta(pergunta, dados_imagem)
            correlacao = tipo_pergunta == "sobre_imagem"
            
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
            logger.error(f"❌ Erro ao responder com OpenAI: {e}")
            return self._responder_local(pergunta, objetos_detectados, faces_nomes, start_time)

    def _criar_prompt_inteligente(self, pergunta, dados_imagem):
        """Cria prompt inteligente"""
        return f"""Você é a Specula, uma assistente amigável e útil.

DADOS DA IMAGEM (se disponíveis):
{dados_imagem}

INSTRUÇÕES:
- Se a pergunta for sobre a imagem, use os dados acima
- Se a pergunta for geral, responda naturalmente
- Seja sempre útil, amigável e acolhedora
- Use emojis ocasionalmente 😊
- Se não souber algo, seja honesta

Responda à pergunta como a Specula:"""

    def _preparar_dados_imagem(self, objetos_detectados, faces_nomes):
        """Prepara dados da imagem"""
        if not objetos_detectados and not faces_nomes:
            return "Nenhum dado de imagem disponível."
        
        partes = []
        
        # Contar objetos
        objetos_contados = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', '')
            count = obj.get('count', 1)
            objetos_contados[nome] = objetos_contados.get(nome, 0) + count
        
        if objetos_contados:
            objetos_texto = []
            for obj, qtd in objetos_contados.items():
                if qtd == 1:
                    objetos_texto.append(f"1 {obj}")
                else:
                    objetos_texto.append(f"{qtd} {obj}s")
            partes.append(f"Objetos: {', '.join(objetos_texto)}")
        
        # Pessoas
        total_pessoas = len(faces_nomes or [])
        if total_pessoas > 0:
            partes.append(f"Pessoas: {total_pessoas}")
        
        return " | ".join(partes)

    def _responder_local(self, pergunta, objetos_detectados, faces_nomes, start_time):
        """Resposta local"""
        pergunta_lower = pergunta.lower()
        
        # Respostas pré-definidas
        if 'que horas' in pergunta_lower or 'hora é' in pergunta_lower:
            resposta = self._obter_hora_local()
        elif 'oque é batata' in pergunta_lower or 'o que é batata' in pergunta_lower:
            resposta = "🍠 A batata é um tubérculo comestível muito versátil!"
        elif 'quem é você' in pergunta_lower:
            resposta = "👋 Eu sou a Specula, sua assistente visual!"
        elif any(palavra in pergunta_lower for palavra in ['oi', 'olá', 'bom dia', 'boa tarde']):
            resposta = self._gerar_cumprimento_local()
        else:
            # Verificar se tem dados da imagem
            objetos_contados = {}
            for obj in (objetos_detectados or []):
                nome = obj.get('name', '')
                count = obj.get('count', 1)
                objetos_contados[nome] = objetos_contados.get(nome, 0) + count
            
            total_pessoas = len(faces_nomes or [])
            
            if objetos_contados or total_pessoas > 0:
                # Pergunta sobre imagem
                if 'quantas pessoas' in pergunta_lower:
                    if total_pessoas > 0:
                        resposta = f"Tem {total_pessoas} pessoa{'s' if total_pessoas > 1 else ''}."
                    else:
                        resposta = "Não tem pessoas."
                elif 'tem ' in pergunta_lower:
                    # Verificar objeto específico
                    for objeto in ['cadeira', 'mesa', 'computador', 'tv', 'planta']:
                        if objeto in pergunta_lower:
                            tem_objeto = any(objeto in obj.lower() for obj in objetos_contados.keys())
                            resposta = f"{'Sim' if tem_objeto else 'Não'}, {'tem' if tem_objeto else 'não tem'} {objeto}."
                            break
                    else:
                        resposta = "Analisando a imagem..."
                else:
                    resposta = "Estou analisando a imagem..."
            else:
                resposta = "Olá! Sou a Specula. No momento estou em modo local."
        
        processing_time = time.time() - start_time
        
        # Determinar tipo
        palavras_imagem = ['imagem', 'foto', 'analisando', 'vejo', 'pessoa', 'pessoas']
        tipo = "sobre_imagem" if any(palavra in pergunta_lower for palavra in palavras_imagem) else "geral"
        
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
        
        elif any(palavra in pergunta_lower for palavra in ['que dia é hoje', 'qual a data']):
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

    def _obter_hora_local(self):
        """Obtém hora local"""
        agora = datetime.now()
        hora_str = agora.strftime("%H:%M")
        return f"São {hora_str}."

    def _gerar_cumprimento_local(self):
        """Gera cumprimento local"""
        hora = datetime.now().hour
        if 5 <= hora < 12:
            return "☀️ Bom dia! Eu sou a Specula."
        elif 12 <= hora < 18:
            return "🌤️ Boa tarde! Eu sou a Specula."
        else:
            return "🌙 Boa noite! Eu sou a Specula."

    def _determinar_tipo_pergunta(self, pergunta, dados_imagem):
        """Determina tipo de pergunta"""
        pergunta_lower = pergunta.lower()
        
        palavras_imagem = ['imagem', 'foto', 'essa ', 'esta ', 'nesta ', 'dessa ',
                          'o que você vê', 'o que tem', 'descreva', 'analise',
                          'quantas pessoas', 'tem ', 'há ', 'vejo']
        
        if any(palavra in pergunta_lower for palavra in palavras_imagem) and "Nenhum dado" not in dados_imagem:
            return "sobre_imagem"
        else:
            return "geral"

    def _formatar_dados_utilizados(self, objetos_detectados, faces_nomes):
        """Formata dados utilizados"""
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_nomes or [])
        return f"{objetos_count} objetos, {faces_count} pessoas"

    # ========== MÉTODO PARA /estatistica ==========

    def obter_estatisticas(self, objetos_detectados, faces_detectadas=None):
        """Estatísticas"""
        logger.info("📊 Gerando estatísticas")
        
        objetos_count = len(objetos_detectados or [])
        faces_count = len(faces_detectadas or [])
        
        # Agrupar objetos por tipo
        categorias = {}
        for obj in (objetos_detectados or []):
            nome = obj.get('name', 'desconhecido')
            count = obj.get('count', 1)
            categorias[nome] = categorias.get(nome, 0) + count
        
        return {
            'sucesso': True,
            'timestamp': time.time(),
            'tempo_processamento': "0.1s",
            'contagens': {
                'total_objetos': objetos_count,
                'total_faces': faces_count,
                'objetos_por_categoria': categorias,
                'faces_conhecidas': len([f for f in (faces_detectadas or []) if f.get('name') != 'Desconhecido']),
                'faces_desconhecidas': len([f for f in (faces_detectadas or []) if f.get('name') == 'Desconhecido'])
            },
            'deteccoes_detalhadas': {
                'objetos': (objetos_detectados or [])[:5],
                'faces': (faces_detectadas or [])[:3]
            }
        }