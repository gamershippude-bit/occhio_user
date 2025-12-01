"""
TESTE_NUVEM.PY - Testador da API Occhio Cloud na nuvem - CORRIGIDO
"""

import requests
import json
import base64
import time
import sys
import os
from pathlib import Path

# Configuração
CLOUD_URL = "https://occhio-cloud-109479952880.us-central1.run.app"
TIMEOUT = 60

class TestadorNuvem:
    def __init__(self, cloud_url):
        self.cloud_url = cloud_url
        self.session = requests.Session()
        
    def carregar_imagem_base64(self, caminho_imagem):
        """Carrega imagem e converte para base64"""
        try:
            with open(caminho_imagem, 'rb') as f:
                imagem_bytes = f.read()
            return base64.b64encode(imagem_bytes).decode('utf-8')
        except Exception as e:
            print(f"❌ Erro ao carregar imagem {caminho_imagem}: {e}")
            return None
    
    def fazer_requisicao(self, endpoint, dados=None, metodo="POST"):
        """Faz requisição com tratamento robusto de erros"""
        url = f"{self.cloud_url}/{endpoint}"
        try:
            if metodo == "POST":
                resposta = self.session.post(url, json=dados, timeout=TIMEOUT)
            else:
                resposta = self.session.get(url, timeout=TIMEOUT)
            
            if resposta.status_code == 200:
                try:
                    return resposta.json(), resposta.status_code, None
                except json.JSONDecodeError:
                    # Se não for JSON, retorna o texto
                    return resposta.text, resposta.status_code, None
            else:
                return None, resposta.status_code, f"HTTP {resposta.status_code}"
                
        except Exception as e:
            return None, None, str(e)
    
    def testar_health(self):
        """Testa endpoint de health"""
        print("🧪 Testando HEALTH...")
        dados, status, erro = self.fazer_requisicao("health", metodo="GET")
        
        if status == 200:
            print(f"✅ Status: {status}")
            print(f"📄 Resposta: {dados}")
            return True
        else:
            print(f"❌ Erro: {erro}")
            return False
    
    def testar_health_completo(self):
        """Testa health completo - CORRIGIDO"""
        print("\n🧪 Testando HEALTH COMPLETO...")
        
        # Tentar ambas as versões (hífen e underline)
        endpoints = ["health-completo", "health_completo", "estatisticas-sistema"]
        
        for endpoint in endpoints:
            dados, status, erro = self.fazer_requisicao(endpoint, metodo="GET")
            
            if status == 200:
                print(f"✅ Status: 200 (endpoint: /{endpoint})")
                
                # Tratamento robusto para diferentes formatos
                estatisticas = {}
                
                if isinstance(dados, dict):
                    if 'estatisticas' in dados:
                        estatisticas = dados.get('estatisticas', {})
                    else:
                        estatisticas = dados  # Pode ser direto
                
                print(f"📊 Estatísticas:")
                print(f"   - Faces cadastradas: {estatisticas.get('faces_cadastradas', 0)}")
                print(f"   - Detector objetos: {estatisticas.get('detector_objetos_ativo', False)}")
                print(f"   - Detector faces: {estatisticas.get('detector_faces_ativo', False)}")
                print(f"   - Interpreter: {estatisticas.get('interpreter_ativo', False)}")
                return True
        
        # Se nenhum endpoint funcionou
        print("❌ Nenhum endpoint de health completo encontrado")
        print("   Endpoints tentados: health-completo, health_completo, estatisticas-sistema")
        return False
    
    def testar_processar(self, imagem_base64):
        """Testa endpoint /processar"""
        print("\n🎯 Testando PROCESSAR...")
        payload = {"image_data": imagem_base64}
        dados, status, erro = self.fazer_requisicao("processar", payload)
        
        if status == 200:
            print(f"✅ Status: {status}")
            
            # Tratamento robusto para diferentes formatos de resposta
            sucesso = False
            resumo = {}
            deteccoes = {}
            
            if isinstance(dados, dict):
                sucesso = dados.get('sucesso', False) or dados.get('success', False)
                resumo = dados.get('resumo', {})
                deteccoes = dados.get('deteccoes_detalhadas', {})
            else:
                print("❌ Resposta não é JSON válido")
                return False
            
            if sucesso:
                print("📦 DETECÇÕES ENCONTRADAS!")
                
                print(f"   - Total objetos: {resumo.get('total_objetos', 0)}")
                print(f"   - Total faces: {resumo.get('total_faces', 0)}")
                print(f"   - Faces conhecidas: {resumo.get('faces_conhecidas', 0)}")
                
                # Mostrar detecções
                objetos = deteccoes.get('objetos', [])
                faces = deteccoes.get('faces', [])
                
                print(f"   📍 Objetos detectados: {len(objetos)}")
                for obj in objetos[:3]:
                    nome = obj.get('nome', obj.get('classe', '?'))
                    conf = obj.get('confiabilidade', obj.get('confianca', '?'))
                    print(f"      - {nome} ({conf})")
                
                print(f"   👤 Faces detectadas: {len(faces)}")
                for face in faces[:3]:
                    nome = face.get('nome', 'Desconhecido')
                    conf = face.get('confiabilidade', face.get('confianca', '?'))
                    conhecida = face.get('conhecida', False)
                    status = "Conhecida" if conhecida else "Desconhecida"
                    print(f"      - {nome} ({conf}) - {status}")
                    
                return True
            else:
                error_msg = dados.get('error', 'Erro desconhecido')
                print(f"❌ Erro do servidor: {error_msg}")
                return False
                
        else:
            print(f"❌ Erro: {erro}")
            return False
    
    def testar_perguntar(self, imagem_base64, pergunta):
        """Testa endpoint /perguntar - CORRIGIDO"""
        print(f"\n❓ Testando PERGUNTAR: '{pergunta}'")
        payload = {
            "image_data": imagem_base64,
            "pergunta": pergunta
        }
        
        dados, status, erro = self.fazer_requisicao("perguntar", payload)
        
        if status == 200:
            print(f"✅ Status: {status}")
            
            # 🔥 CORREÇÃO PRINCIPAL - Tratamento robusto para diferentes formatos
            resposta_texto = ""
            correlacao_imagem = False
            dados_utilizados = {}
            sucesso = False
            
            if isinstance(dados, dict):
                # Formato JSON normal
                sucesso = dados.get('sucesso', False) or dados.get('success', False)
                resposta_texto = dados.get('resposta', 'Resposta não disponível')
                correlacao_imagem = dados.get('correlacao_com_imagem', False)
                dados_utilizados = dados.get('dados_utilizados', {})
            elif isinstance(dados, str):
                # Resposta direta em string
                resposta_texto = dados
                sucesso = True  # Assumir sucesso se recebeu resposta
            else:
                resposta_texto = "Formato de resposta desconhecido"
            
            print(f"💬 RESPOSTA: {resposta_texto}")
            
            # Só mostrar estes campos se estiverem disponíveis
            if isinstance(dados, dict):
                print(f"🔗 Correlação com imagem: {correlacao_imagem}")
                
                if dados_utilizados and dados_utilizados != "Pergunta geral - sem dados da imagem":
                    if isinstance(dados_utilizados, dict):
                        pessoas = dados_utilizados.get('pessoas_detectadas', 0)
                        objetos = dados_utilizados.get('objetos_detectados', {})
                        print(f"📊 Dados utilizados: {pessoas} pessoa(s), {len(objetos)} tipo(s) de objeto(s)")
                    else:
                        print(f"📊 Dados utilizados: {dados_utilizados}")
            
            return True
        else:
            print(f"❌ Erro: {erro}")
            if dados:
                print(f"📄 Resposta parcial: {dados}")
            return False
    
    def testar_estatistica(self, imagem_base64):
        """Testa endpoint /estatistica"""
        print("\n📊 Testando ESTATISTICA...")
        payload = {"image_data": imagem_base64}
        dados, status, erro = self.fazer_requisicao("estatistica", payload)
        
        if status == 200:
            print(f"✅ Status: {status}")
            
            # Tratamento robusto para diferentes formatos
            contagens = {}
            precisao = {}
            objetos_tipo = {}
            
            if isinstance(dados, dict):
                # Pode vir direto ou dentro de 'contagens'
                if 'contagens' in dados:
                    contagens = dados.get('contagens', {})
                    precisao = dados.get('precisao', {})
                else:
                    contagens = dados  # Dados diretos
                
                objetos_tipo = contagens.get('objetos_por_tipo', {})
            
            print("📈 ESTATÍSTICAS ENCONTRADAS!")
            print(f"   - Total objetos: {contagens.get('total_objetos', 0)}")
            print(f"   - Total faces: {contagens.get('total_faces', 0)}")
            print(f"   - Faces conhecidas: {contagens.get('faces_conhecidas', 0)}")
            
            # Objetos por tipo
            if objetos_tipo:
                print("   📦 Objetos por tipo:")
                for obj, qtd in objetos_tipo.items():
                    print(f"      - {obj}: {qtd}")
            elif isinstance(dados, dict):
                # Tentar encontrar objetos de outra forma
                objetos_detectados = contagens.get('objetos_detectados', {})
                if objetos_detectados:
                    print("   📦 Objetos por tipo:")
                    for obj, qtd in objetos_detectados.items():
                        print(f"      - {obj}: {qtd}")
            
            # Precisão
            print("   🎯 Precisão:")
            conf_objetos = precisao.get('confianca_media_objetos', 'N/A')
            conf_faces = precisao.get('confianca_media_faces', 'N/A')
            print(f"      - Confiança média objetos: {conf_objetos}")
            print(f"      - Confiança média faces: {conf_faces}")
            
            return True
        else:
            print(f"❌ Erro: {erro}")
            return False
    
    def testar_completo(self, imagem_base64, pergunta):
        """Testa endpoint /completo"""
        print(f"\n🎪 Testando COMPLETO...")
        payload = {
            "image_data": imagem_base64,
            "pergunta": pergunta
        }
        
        dados, status, erro = self.fazer_requisicao("completo", payload)
        
        if status == 200:
            print(f"✅ Status: {status}")
            
            # Tratamento robusto para diferentes formatos
            if isinstance(dados, dict):
                sucesso = dados.get('sucesso', False) or dados.get('success', False)
                
                if sucesso:
                    print("📋 RESUMO COMPLETO ENCONTRADO!")
                    
                    # Resumo inteligente
                    resumo = dados.get('resumo_inteligente', '')
                    if resumo:
                        print(f"   🧠 {resumo}")
                    
                    # Resposta da pergunta
                    resposta_pergunta = dados.get('perguntar', {})
                    if isinstance(resposta_pergunta, dict):
                        resposta = resposta_pergunta.get('resposta', '')
                        if resposta:
                            print(f"   💬 Resposta: {resposta}")
                    else:
                        print(f"   💬 Resposta: {resposta_pergunta}")
                    
                    # Estatísticas
                    stats = dados.get('estatisticas', {})
                    if isinstance(stats, dict):
                        contagens = stats.get('contagens', {})
                        total_obj = contagens.get('total_objetos', 0)
                        total_faces = contagens.get('total_faces', 0)
                        print(f"   📊 Estatísticas: {total_obj} objetos, {total_faces} faces")
                    
                    # Tempo de processamento
                    tempo_total = dados.get('tempo_total_processamento', 'N/A')
                    print(f"   ⏱️  Tempo total: {tempo_total}")
                    
                    return True
                else:
                    error_msg = dados.get('error', 'Erro desconhecido')
                    print(f"❌ Erro do servidor: {error_msg}")
                    return False
            else:
                print(f"📄 Resposta: {dados}")
                return True
                
        else:
            print(f"❌ Erro: {erro}")
            return False
    
    def executar_testes_completos(self, caminho_imagem):
        """Executa todos os testes com uma imagem"""
        print("=" * 60)
        print("🎪 TESTADOR DA API OCCHIO CLOUD - NUVEM")
        print("=" * 60)
        print(f"🌐 Conectando em: {self.cloud_url}")
        print("=" * 60)
        
        # Carregar imagem
        print(f"📸 Carregando imagem: {caminho_imagem}")
        imagem_base64 = self.carregar_imagem_base64(caminho_imagem)
        if not imagem_base64:
            print("❌ Falha ao carregar imagem. Abortando testes.")
            return False
        
        print(f"✅ Imagem convertida! Tamanho base64: {len(imagem_base64)} caracteres")
        
        resultados = []
        
        # Executar testes
        resultados.append(self.testar_health())
        resultados.append(self.testar_health_completo())
        resultados.append(self.testar_processar(imagem_base64))
        
        # Testar várias perguntas
        perguntas = [
            "Quantas pessoas estão na foto?",
            "O que tem nesta imagem?",
            "o céu é azul?"
        ]
        
        for pergunta in perguntas:
            resultados.append(self.testar_perguntar(imagem_base64, pergunta))
            time.sleep(1)  # Pequena pausa entre requisições
        
        resultados.append(self.testar_estatistica(imagem_base64))
        resultados.append(self.testar_completo(imagem_base64, "Descreva o que você vê"))
        
        # Resumo final
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES")
        print("=" * 60)
        sucessos = sum(resultados)
        total = len(resultados)
        print(f"✅ Testes passaram: {sucessos}/{total}")
        print(f"📈 Taxa de sucesso: {sucessos/total*100:.1f}%" if total > 0 else "0%")
        
        if sucessos == total:
            print("🎉 TODOS OS TESTES PASSARAM! A API está funcionando perfeitamente!")
        elif sucessos > 0:
            print("⚠️  Alguns testes passaram, outros falharam.")
        else:
            print("❌ Todos os testes falharam. Verifique a API.")
        
        return sucessos > 0

def main():
    """Função principal"""
    if len(sys.argv) < 2:
        print("Uso: python teste_nuvem.py <caminho_para_imagem>")
        print("Exemplo: python teste_nuvem.py teste.jpg")
        sys.exit(1)
    
    caminho_imagem = sys.argv[1]
    
    if not os.path.exists(caminho_imagem):
        print(f"❌ Arquivo não encontrado: {caminho_imagem}")
        sys.exit(1)
    
    testador = TestadorNuvem(CLOUD_URL)
    sucesso = testador.executar_testes_completos(caminho_imagem)
    
    sys.exit(0 if sucesso else 1)

if __name__ == "__main__":
    main()