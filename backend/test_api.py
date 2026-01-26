#!/usr/bin/env python3
"""
Test API endpoint with real image
"""
import requests
import json

# API endpoint
api_url = "http://localhost:8001/api/chat/images"

# Test image path
image_path = "/app/backend/test_chart.png"

# Custom question for analysis
question = """Faça uma análise técnica COMPLETA e DETALHADA deste gráfico de trading.

INCLUA OBRIGATORIAMENTE:
1. Identificação do ativo e timeframe
2. Tendência atual (alta, baixa ou lateral)
3. Padrões de candlestick visíveis
4. Níveis de suporte e resistência com valores exatos
5. Indicadores técnicos visíveis
6. **RECOMENDAÇÃO CLARA: COMPRA (CALL) ou VENDA (PUT)**
7. **Níveis de entrada, stop loss e take profit COM VALORES NUMÉRICOS**
8. **Percentual de confiança da análise**
9. Estratégia de trading aplicável (Counter-Trend, Breakout, etc.)
10. Análise de risco/retorno

Forneça uma RECOMENDAÇÃO DEFINITIVA ao final."""

# Prepare the request
files = {
    'files': ('chart.png', open(image_path, 'rb'), 'image/png')
}
data = {
    'question': question
}

print("📤 Enviando imagem para análise...")
print(f"🔗 URL: {api_url}")
print(f"📊 Imagem: {image_path}")
print(f"❓ Pergunta personalizada incluída\n")

try:
    response = requests.post(api_url, files=files, data=data, timeout=90)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Resposta recebida com sucesso!\n")
        print("=" * 80)
        print("📊 ANÁLISE DA IA:")
        print("=" * 80)
        print(result['assistant_message']['content'])
        print("\n" + "=" * 80)
        
        # Check for annotated images
        if result.get('annotated_image_paths'):
            print("\n🎨 IMAGENS ANOTADAS GERADAS:")
            for idx, path in enumerate(result['annotated_image_paths']):
                if path:
                    print(f"   {idx + 1}. {path}")
            print("\n✨ As imagens anotadas incluem:")
            print("   - Banner de recomendação (CALL/PUT)")
            print("   - Setas de entrada")
            print("   - Linhas de suporte/resistência")
            print("   - Linhas de tendência")
            print("   - Zonas de trading")
            print("   - Labels de estratégia")
        else:
            print("\n⚠️  Nenhuma imagem anotada foi gerada")
            print("   (Possível que a IA não tenha dado recomendação CALL/PUT clara)")
        
    else:
        print(f"❌ Erro na requisição: {response.status_code}")
        print(f"   Resposta: {response.text}")
        
except requests.exceptions.Timeout:
    print("⏱️  Timeout: A análise está demorando muito")
    print("   Isso é normal para a primeira requisição (carregamento do modelo)")
    print("   Tente novamente em alguns segundos")
except Exception as e:
    print(f"❌ Erro: {e}")
