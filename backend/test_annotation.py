#!/usr/bin/env python3
"""
Test script for chart annotation system
"""
import sys
sys.path.insert(0, '/app/backend')

from image_annotator import ChartAnnotator

# Sample analysis text simulating AI response
analysis_text = """
# Análise Técnica Completa - EUR/USD M5

## 1. Identificação
- **Ativo**: EUR/USD
- **Timeframe**: M5 (5 minutos)
- **Preço Atual**: 1.1867
- **Horário**: 10:33

## 2. Análise Técnica

### Tendência Principal
- **Tendência de BAIXA** confirmada
- Strong Rejection em 1.1867 indicando pressão vendedora
- Formação de topos descendentes

### Padrões Identificados
- Counter-Trend Trade setup visível
- Rejeição forte em zona de resistência
- Possível reversão de curto prazo

### Níveis de Suporte e Resistência
- **Resistência Principal**: 1.1867
- **Suporte Imediato**: 1.1850
- **Suporte Forte**: 1.1837

### Estrutura de Mercado
- Topos e fundos descendentes
- Momentum de baixa aumentando
- Volume confirmando movimento

## 3. Projeções e Estimativas

### Cenário Principal (VENDA/PUT)
- **Probabilidade**: 75%
- **Entrada**: PUT após confirmação em 1.1865
- **Stop Loss**: 1.1875
- **Take Profit 1**: 1.1850
- **Take Profit 2**: 1.1837
- **Risco/Retorno**: 1:2.8

### Condições para Entrada
- Aguardar confirmação com candle de rejeição
- Timeframe recomendado: 1-2 candles (5-10 minutos)
- Gestão de risco: 2% do capital por operação

## 4. Conclusão e Recomendações

**RECOMENDAÇÃO: VENDA (PUT)**

### Pontos Principais:
1. Strong Rejection em zona de resistência confirma pressão vendedora
2. Tendência de baixa bem estabelecida com topos descendentes
3. Counter-Trend Trade com boa relação risco/retorno
4. Volume confirma movimento de venda

**Nível de Confiança: 75%**

### Riscos:
- Possível falso rompimento se preço romper 1.1870
- Eventos econômicos podem inverter movimento
- Volatilidade do M5 requer monitoramento constante

**Exit em 1-2 candles (Exit 1-2 candles no gráfico)**
"""

# Load test image
with open('/app/backend/test_chart.png', 'rb') as f:
    image_bytes = f.read()

print("🔄 Iniciando anotação do gráfico...")
print(f"📊 Tamanho da imagem: {len(image_bytes) / 1024 / 1024:.2f} MB")

# Create annotator and process
annotator = ChartAnnotator()

print("🔍 Extraindo sinais de trading...")
signals = annotator.extract_trading_signals(analysis_text)
print(f"✅ Sinais extraídos:")
print(f"   - Ação: {signals['action']}")
print(f"   - Tendência: {signals['trend']}")
print(f"   - Confiança: {signals['confidence']}%")
print(f"   - Estratégia: {signals['strategy']}")
print(f"   - Suportes: {signals['support_levels']}")
print(f"   - Resistências: {signals['resistance_levels']}")
print(f"   - Stop Loss: {signals['stop_loss']}")
print(f"   - Take Profit: {signals['take_profit']}")

print("\n🎨 Gerando imagem anotada...")
annotated_bytes = annotator.annotate_chart(image_bytes, analysis_text, signals)

print(f"✅ Imagem anotada gerada: {len(annotated_bytes) / 1024 / 1024:.2f} MB")

# Save annotated image
output_path = '/app/backend/test_chart_annotated.png'
with open(output_path, 'wb') as f:
    f.write(annotated_bytes)

print(f"💾 Imagem salva em: {output_path}")
print("✨ Teste concluído com sucesso!")
