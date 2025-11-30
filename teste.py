#!/usr/bin/env python3
"""
Teste para a API Occhio Cloud LOCAL
Autor: Teste Local
"""

import requests
import base64
import json
import sys
import time
from pathlib import Path

# Configurações para servidor LOCAL
API_URL = "http://localhost:8080"  # ← MUDOU PARA LOCALHOST

def image_to_base64(image_path):
    """Converte imagem para base64"""
    try:
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_data
    except Exception as e:
        print(f"❌ Erro ao converter imagem: {e}")
        return None

def test_health():
    """Testa endpoints de saúde"""
    print("🧪 Testando saúde da API LOCAL...")
    
    try:
        # Health Simple
        response = requests.get(f"{API_URL}/health-simple", timeout=10)
        print(f"✅ Health Simple: {response.text} (Status: {response.status_code})")
        
        # Health Check
        response = requests.get(f"{API_URL}/health", timeout=10)
        health_data = response.json()
        print(f"✅ Health Check: {health_data}")
        
        # Estatísticas
        response = requests.get(f"{API_URL}/estatisticas", timeout=10)
        stats = response.json()
        print("📊 Estatísticas do Sistema:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except requests.exceptions.ConnectionError:
        print("❌ ERRO: Não consegui conectar com o servidor local!")
        print("   Certifique-se de que o servidor está rodando:")
        print("   - Execute: python main.py")
        print("   - Ou: docker run -p 8080:8080 occhio-local")
        return False
    except Exception as e:
        print(f"❌ Erro no health check: {e}")
        return False
    
    return True

def test_analise_rapida(image_base64):
    """Testa análise rápida"""
    print("\n🚀 Testando análise rápida...")
    
    payload = {
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/analise-rapida", json=payload, timeout=30)
        result = response.json()
        
        if result.get("success"):
            print("✅ Análise Rápida - Sucesso!")
            print(f"📝 Resposta: {result.get('resposta', 'N/A')}")
            print(f"👥 Detecções: {result.get('deteccoes_basicas', {})}")
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro na análise rápida: {e}")

def test_perguntar(image_base64, pergunta):
    """Testa fazer pergunta sobre imagem"""
    print(f"\n❓ Testando pergunta: '{pergunta}'")
    
    payload = {
        "pergunta": pergunta,
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/perguntar", json=payload, timeout=30)
        result = response.json()
        
        if result.get("success"):
            print("✅ Pergunta - Sucesso!")
            print(f"📝 Resposta: {result.get('resposta', 'N/A')}")
            print(f"🔍 Detecções: {result.get('deteccoes', {})}")
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro na pergunta: {e}")

def test_deteccoes_detalhadas(image_base64):
    """Testa detecções detalhadas"""
    print("\n🎯 Testando detecções detalhadas...")
    
    payload = {
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/deteccoes-detalhadas", json=payload, timeout=30)
        result = response.json()
        
        if result.get("success"):
            print("✅ Detecções Detalhadas - Sucesso!")
            coordenadas = result.get('coordenadas', {})
            print(f"👥 Faces detectadas: {len(coordenadas.get('faces', []))}")
            print(f"📦 Objetos detectados: {len(coordenadas.get('objetos', []))}")
            
            # Mostrar detalhes das faces
            for i, face in enumerate(coordenadas.get('faces', [])):
                print(f"   👤 Face {i+1}: {face['nome']} (confiança: {face['confianca']:.2f})")
            
            # Mostrar detalhes dos objetos
            for i, obj in enumerate(coordenadas.get('objetos', [])):
                print(f"   📍 Objeto {i+1}: {obj.get('classe', 'N/A')} (confiança: {obj.get('confianca', 0):.2f})")
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro nas detecções detalhadas: {e}")

def test_completo(image_base64, pergunta):
    """Testa endpoint completo"""
    print(f"\n🎯 Testando endpoint COMPLETO...")
    
    payload = {
        "pergunta": pergunta,
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/completo", json=payload, timeout=30)
        result = response.json()
        
        if result.get("success"):
            print("✅ Endpoint Completo - Sucesso!")
            print(f"📝 Resposta Inteligente: {result.get('resposta_inteligente', 'N/A')}")
            
            # Análise rápida
            analise = result.get('analise_rapida', {})
            print(f"📊 Análise Rápida: {analise.get('resposta', 'N/A')}")
            
            # Detecções detalhadas
            deteccoes = result.get('deteccoes_detalhadas', {})
            coordenadas = deteccoes.get('coordenadas', {})
            print(f"👥 Faces detectadas: {len(coordenadas.get('faces', []))}")
            print(f"📦 Objetos detectados: {len(coordenadas.get('objetos', []))}")
            
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro no endpoint completo: {e}")

def test_processar(image_base64):
    """Testa processamento completo"""
    print("\n🔄 Testando processamento completo...")
    
    payload = {
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/processar", json=payload, timeout=30)
        result = response.json()
        
        if result.get("success"):
            print("✅ Processamento - Sucesso!")
            analise = result.get('analise_rapida', {})
            print(f"📝 Resposta: {analise.get('resposta', 'N/A')}")
            
            deteccoes = result.get('deteccoes_detalhadas', {})
            coordenadas = deteccoes.get('coordenadas', {})
            print(f"👥 Faces: {len(coordenadas.get('faces', []))}")
            print(f"📦 Objetos: {len(coordenadas.get('objetos', []))}")
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")

def main():
    """Função principal"""
    print("=" * 60)
    print("🎪 TESTADOR DA API OCCHIO CLOUD - SERVIDOR LOCAL")
    print("=" * 60)
    print(f"🌐 Conectando em: {API_URL}")
    print("=" * 60)
    
    # 1. Testar saúde da API
    if not test_health():
        print("\n💡 DICA: Para iniciar o servidor local:")
        print("   1. Terminal 1: python main.py")
        print("   2. Terminal 2: python teste.py imagem.jpg")
        print("   3. Ou usar Docker: docker run -p 8080:8080 occhio-local")
        return
    
    # 2. Verificar se foi passada uma imagem
    if len(sys.argv) < 2:
        print("\n📝 Uso: python teste.py <caminho_da_imagem>")
        print("   Exemplo: python teste.py minha_foto.jpg")
        print("\n💡 Certifique-se de que o servidor está rodando em outro terminal!")
        return
    
    image_path = sys.argv[1]
    
    # 3. Verificar se arquivo existe
    if not Path(image_path).exists():
        print(f"❌ Arquivo não encontrado: {image_path}")
        return
    
    # 4. Converter imagem para base64
    print(f"\n🖼️ Convertendo imagem: {image_path}")
    image_base64 = image_to_base64(image_path)
    
    if not image_base64:
        print("❌ Falha ao converter imagem")
        return
    
    print(f"✅ Imagem convertida! Tamanho base64: {len(image_base64)} caracteres")
    
    # 5. Testar todos os endpoints
    test_analise_rapida(image_base64)
    test_deteccoes_detalhadas(image_base64)
    test_processar(image_base64)
    
    # Testar diferentes perguntas
    perguntas = [
        "Quantas pessoas estão na foto?",
        "O que tem nesta imagem?",
        "Tem cadeiras na imagem?",
        "Tem livros na imagem?",
        "Tem carros na imagem?",
        "O céu é azul?",
        "Descreva o ambiente",
        "Qual é o conteúdo desta imagem?"
    ]
    
    for pergunta in perguntas:
        test_perguntar(image_base64, pergunta)
        time.sleep(1)  # Pequena pausa entre requisições
    
    # Testar endpoint completo
    test_completo(image_base64, "Descreva detalhadamente o que você vê")
    
    print("\n" + "=" * 60)
    print("🎉 TESTES CONCLUÍDOS!")
    print("=" * 60)

if __name__ == "__main__":
    main()