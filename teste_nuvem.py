"""
TESTE_LOCAL.PY - Testador da API Occhio Cloud LOCAL
"""

import requests
import json
import base64
import time
import sys
import os
from pathlib import Path

# Configuração - Mude para LOCAL
LOCAL_URL = "http://localhost:8080"  # URL local
CLOUD_URL = "https://occhio-cloud-109479952880.us-central1.run.app"  # URL cloud
TIMEOUT = 60

# Escolha qual URL usar
USAR_LOCAL = True  # Mude para False para testar na nuvem

class TestadorOcchio:
    def __init__(self, base_url):
        self.base_url = base_url
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
    
    def testar_health(self):
        """Testa endpoint de health"""
        print("🧪 Testando HEALTH...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            print(f"📄 Resposta: {data}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def testar_health_completo(self):
        """Testa health completo"""
        print("\n🧪 Testando HEALTH COMPLETO...")
        try:
            response = self.session.get(f"{self.base_url}/health-completo", timeout=TIMEOUT)
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            
            if data.get('success'):  # Este ainda é 'success'
                stats = data.get('estatisticas', {})
                print(f"📊 Estatísticas:")
                print(f"   - Faces cadastradas: {stats.get('faces_cadastradas', 0)}")
                print(f"   - Detector objetos: {stats.get('detector_objetos_ativo', False)}")
                print(f"   - Detector faces: {stats.get('detector_faces_ativo', False)}")
                print(f"   - Interpreter: {stats.get('interpreter_ativo', False)}")
                return True
            else:
                print(f"❌ Success: False")
                return False
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def testar_processar(self, imagem_base64):
        """Testa endpoint /processar"""
        print("\n🎯 Testando PROCESSAR...")
        try:
            payload = {
                "image_data": imagem_base64
            }
            
            response = self.session.post(
                f"{self.base_url}/processar",
                json=payload,
                timeout=TIMEOUT
            )
            
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            
            if data.get('sucesso'):  # CORREÇÃO: 'sucesso' em português
                print("📦 DETECÇÕES ENCONTRADAS!")
                
                resumo = data.get('resumo', {})
                print(f"   - Total objetos: {resumo.get('total_objetos', 0)}")
                print(f"   - Total faces: {resumo.get('total_faces', 0)}")
                print(f"   - Faces conhecidas: {resumo.get('faces_conhecidas', 0)}")
                
                # Mostrar detecções
                deteccoes = data.get('deteccoes_detalhadas', {})
                objetos = deteccoes.get('objetos', [])
                faces = deteccoes.get('faces', [])
                
                print(f"   📍 Objetos detectados: {len(objetos)}")
                for obj in objetos[:3]:
                    nome = obj.get('nome', '?')
                    conf = obj.get('confiabilidade', '?')
                    print(f"      - {nome} ({conf})")
                
                print(f"   👤 Faces detectadas: {len(faces)}")
                for face in faces[:3]:
                    nome = face.get('nome', '?')
                    conf = face.get('confiabilidade', '?')
                    print(f"      - {nome} ({conf})")
                    
                return True
            else:
                error_msg = data.get('error', 'Erro desconhecido')
                print(f"❌ Erro do servidor: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def testar_perguntar(self, imagem_base64, pergunta):
        """Testa endpoint /perguntar"""
        print(f"\n❓ Testando PERGUNTAR: '{pergunta}'")
        try:
            payload = {
                "image_data": imagem_base64,
                "pergunta": pergunta
            }
            
            response = self.session.post(
                f"{self.base_url}/perguntar",
                json=payload,
                timeout=TIMEOUT
            )
            
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            
            if data.get('sucesso'):  # CORREÇÃO: 'sucesso' em português
                resposta = data.get('resposta', 'N/A')
                print(f"💬 RESPOSTA: {resposta}")
                
                correlacao = data.get('correlacao_com_imagem', False)
                tipo_pergunta = data.get('tipo_pergunta', 'N/A')
                print(f"🔗 Correlação com imagem: {correlacao}")
                print(f"📝 Tipo de pergunta: {tipo_pergunta}")
                
                # Mostrar dados utilizados
                dados_utilizados = data.get('dados_utilizados', {})
                if dados_utilizados and dados_utilizados != "Pergunta geral sobre o mundo - sem dados da imagem":
                    pessoas = dados_utilizados.get('pessoas_detectadas', 0)
                    objetos = dados_utilizados.get('objetos_detectados', {})
                    print(f"📊 Dados utilizados: {pessoas} pessoa(s), {len(objetos)} tipo(s) de objeto(s)")
                else:
                    print(f"📊 Dados utilizados: {dados_utilizados}")
                    
                return True
            else:
                error_msg = data.get('error', 'Erro desconhecido')
                print(f"❌ Erro do servidor: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def testar_estatistica(self, imagem_base64):
        """Testa endpoint /estatistica"""
        print("\n📊 Testando ESTATISTICA...")
        try:
            payload = {
                "image_data": imagem_base64
            }
            
            response = self.session.post(
                f"{self.base_url}/estatistica",
                json=payload,
                timeout=TIMEOUT
            )
            
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            
            # CORREÇÃO: endpoint /estatistica não tem campo 'sucesso', é direto
            if 'contagens' in data:  
                print("📈 ESTATÍSTICAS ENCONTRADAS!")
                
                contagens = data.get('contagens', {})
                print(f"   - Total objetos: {contagens.get('total_objetos', 0)}")
                print(f"   - Total faces: {contagens.get('total_faces', 0)}")
                print(f"   - Faces conhecidas: {contagens.get('faces_conhecidas', 0)}")
                
                # Objetos por tipo
                objetos_tipo = contagens.get('objetos_por_tipo', {})
                if objetos_tipo:
                    print("   📦 Objetos por tipo:")
                    for obj, qtd in objetos_tipo.items():
                        print(f"      - {obj}: {qtd}")
                
                # Precisão
                precisao = data.get('precisao', {})
                print("   🎯 Precisão:")
                print(f"      - Confiança média objetos: {precisao.get('confianca_media_objetos', 'N/A')}")
                print(f"      - Confiança média faces: {precisao.get('confianca_media_faces', 'N/A')}")
                
                return True
            else:
                print("❌ Estrutura de resposta inesperada")
                return False
                
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def testar_completo(self, imagem_base64, pergunta):
        """Testa endpoint /completo"""
        print(f"\n🎪 Testando COMPLETO...")
        try:
            payload = {
                "image_data": imagem_base64,
                "pergunta": pergunta
            }
            
            response = self.session.post(
                f"{self.base_url}/completo",
                json=payload,
                timeout=TIMEOUT
            )
            
            print(f"✅ Status: {response.status_code}")
            data = response.json()
            
            if data.get('sucesso'):  # CORREÇÃO: 'sucesso' em português
                print("📋 RESUMO COMPLETO ENCONTRADO!")
                
                # Resumo inteligente
                resumo = data.get('resumo_inteligente', '')
                if resumo:
                    print(f"   🧠 {resumo}")
                
                # Resposta da pergunta
                resposta_pergunta = data.get('perguntar', {})
                if isinstance(resposta_pergunta, dict) and resposta_pergunta.get('sucesso'):
                    resposta = resposta_pergunta.get('resposta', '')
                    if resposta:
                        print(f"   💬 Resposta: {resposta}")
                
                # Estatísticas
                stats = data.get('estatisticas', {})
                if isinstance(stats, dict) and 'contagens' in stats:
                    contagens = stats['contagens']
                    print(f"   📊 Estatísticas: {contagens.get('total_objetos', 0)} objetos, {contagens.get('total_faces', 0)} faces")
                
                # Tempo de processamento
                tempo_total = data.get('tempo_total_processamento', 'N/A')
                print(f"   ⏱️  Tempo total: {tempo_total}")
                    
                return True
            else:
                error_msg = data.get('error', 'Erro desconhecido')
                print(f"❌ Erro do servidor: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False
    
    def executar_testes_completos(self, caminho_imagem):
        """Executa todos os testes com uma imagem"""
        ambiente = "LOCAL" if "localhost" in self.base_url else "NUVEM"
        print("=" * 60)
        print(f"🎪 TESTADOR DA API OCCHIO CLOUD - {ambiente}")
        print("=" * 60)
        print(f"🌐 Conectando em: {self.base_url}")
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
        
        # Testar várias perguntas - INCLUINDO PERGUNTAS GERAIS
        perguntas = [
            "Quantas pessoas estão na foto?",
            "O que tem nesta imagem?",
            "Tem cadeiras na imagem?",
            "o céu é azul?",
            "Quantos planetas existem no sistema solar?",
            "Descreva o ambiente"
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
        print("Uso: python teste_local.py <caminho_para_imagem>")
        print("Exemplo: python teste_local.py teste.jpg")
        sys.exit(1)
    
    caminho_imagem = sys.argv[1]
    
    if not os.path.exists(caminho_imagem):
        print(f"❌ Arquivo não encontrado: {caminho_imagem}")
        sys.exit(1)
    
    # Escolher URL base
    if USAR_LOCAL:
        url = LOCAL_URL
        print("🔧 Modo: TESTE LOCAL")
    else:
        url = CLOUD_URL  
        print("☁️  Modo: TESTE NUVEM")
    
    testador = TestadorOcchio(url)
    sucesso = testador.executar_testes_completos(caminho_imagem)
    
    sys.exit(0 if sucesso else 1)

if __name__ == "__main__":
    main()