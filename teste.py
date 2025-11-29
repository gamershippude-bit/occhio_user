#!/usr/bin/env python3
"""
Teste para a API Occhio Cloud
Autor: Seu Amigo
"""

import requests
import base64
import json
import sys
from pathlib import Path

# Configurações
API_URL = "https://occhio-cloud-l6d4xbh4va-uc.a.run.app"

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
    print("🧪 Testando saúde da API...")
    
    try:
        # Health Simple
        response = requests.get(f"{API_URL}/health-simple")
        print(f"✅ Health Simple: {response.text} (Status: {response.status_code})")
        
        # Health Check
        response = requests.get(f"{API_URL}/health")
        health_data = response.json()
        print(f"✅ Health Check: {health_data}")
        
        # Estatísticas
        response = requests.get(f"{API_URL}/estatisticas")
        stats = response.json()
        print("📊 Estatísticas do Sistema:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Erro no health check: {e}")

def test_analise_rapida(image_base64):
    """Testa análise rápida"""
    print("\n🚀 Testando análise rápida...")
    
    payload = {
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/analise-rapida", json=payload)
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
        response = requests.post(f"{API_URL}/perguntar", json=payload)
        result = response.json()
        
        if result.get("success"):
            print("✅ Pergunta - Sucesso!")
            print(f"📝 Resposta: {result.get('resposta', 'N/A')}")
            print(f"🔍 Detecções: {result.get('deteccoes', {})}")
        else:
            print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"❌ Erro na pergunta: {e}")

def test_completo(image_base64, pergunta):
    """Testa endpoint completo"""
    print(f"\n🎯 Testando endpoint COMPLETO...")
    
    payload = {
        "pergunta": pergunta,
        "image_data": image_base64
    }
    
    try:
        response = requests.post(f"{API_URL}/completo", json=payload)
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
        
    def test_debug_deteccoes(image_base64):
     """Testa para ver as detecções detalhadas"""
     print("\n🔍 Debug - Detecções Detalhadas...")

     payload = {
         "image_data": image_base64
     }

     try:
         response = requests.post(f"{API_URL}/deteccoes-detalhadas", json=payload)
         result = response.json()

         if result.get("success"):
             print("✅ Detecções Detalhadas:")
             coordenadas = result.get('coordenadas', {})

             print(f"👥 Faces: {len(coordenadas.get('faces', []))}")
             for face in coordenadas.get('faces', []):
                 print(f"   - {face['nome']} (confiança: {face['confianca']:.2f})")

             print(f"📦 Objetos: {len(coordenadas.get('objetos', []))}")
             for obj in coordenadas.get('objetos', []):
                 print(f"   - {obj['classe']} (confiança: {obj['confianca']:.2f})")
         else:
             print(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")

     except Exception as e:
         print(f"❌ Erro nas detecções detalhadas: {e}")


    test_debug_deteccoes(image_base64)

def main():
    """Função principal"""
    print("=" * 60)
    print("🎪 TESTADOR DA API OCCHIO CLOUD")
    print("=" * 60)
    
    # 1. Testar saúde da API
    test_health()
    
    # 2. Verificar se foi passada uma imagem
    if len(sys.argv) < 2:
        print("\n📝 Uso: python testar_occhio.py <caminho_da_imagem>")
        print("   Exemplo: python testar_occhio.py minha_foto.jpg")
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
    
    # Testar diferentes perguntas
    perguntas = [
        "O ceu é azul?",
        "Quantas pessoas estão na foto?",
        "Descreva o ambiente",
        "Tem algum objeto interessante?"
    ]
    
    for pergunta in perguntas:
        test_perguntar(image_base64, pergunta)
    
    # Testar endpoint completo
    test_completo(image_base64, "Descreva detalhadamente o que você vê")
    
    print("\n" + "=" * 60)
    print("🎉 TESTES CONCLUÍDOS!")
    print("=" * 60)

if __name__ == "__main__":
    main()