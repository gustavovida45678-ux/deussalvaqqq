#!/bin/bash
# Script de Diagnóstico - API Key Customizada

echo "🔍 DIAGNÓSTICO DE API KEY CUSTOMIZADA"
echo "======================================"
echo ""

# Verificar status dos serviços
echo "1️⃣ Verificando status dos serviços..."
sudo supervisorctl status backend frontend | grep -E "RUNNING|FATAL|STOPPED"
echo ""

# Verificar se API está respondendo
echo "2️⃣ Testando endpoint de saúde..."
curl -s http://localhost:8001/api/ | jq .
echo ""

# Testar com chave Emergent (padrão)
echo "3️⃣ Testando com chave Emergent (padrão)..."
response=$(curl -s -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá"}' 2>&1)

if echo "$response" | grep -q "Budget has been exceeded"; then
    echo "❌ Chave Emergent sem créditos! Você PRECISA usar sua própria chave."
    echo "   Siga as instruções para adicionar créditos ou use API key OpenAI."
elif echo "$response" | grep -q "assistant_message"; then
    echo "✅ Chave Emergent funcionando!"
else
    echo "⚠️  Resposta inesperada:"
    echo "$response" | head -3
fi
echo ""

# Instruções para usuário
echo "4️⃣ COMO TESTAR SUA CHAVE OPENAI:"
echo ""
echo "Execute este comando (substitua SUA_CHAVE pela sua chave real):"
echo ""
echo 'curl -X POST http://localhost:8001/api/chat \'
echo '  -H "Content-Type: application/json" \'
echo '  -H "X-Custom-API-Key: SUA_CHAVE_AQUI" \'
echo '  -d '"'"'{"message": "Olá"}'"'"
echo ""
echo "Se sua chave for válida, você verá uma resposta da IA."
echo "Se não for válida, verá erro de autenticação da OpenAI."
echo ""

echo "5️⃣ VERIFICAR CONFIGURAÇÃO NO NAVEGADOR:"
echo ""
echo "Abra o console do navegador (F12) e digite:"
echo ""
echo "console.log('Provider:', localStorage.getItem('api_provider'));"
echo "console.log('Chave configurada:', localStorage.getItem('user_api_key') ? 'SIM' : 'NÃO');"
echo ""
echo "Deve mostrar:"
echo "  Provider: custom"
echo "  Chave configurada: SIM"
echo ""

echo "6️⃣ PROBLEMAS COMUNS:"
echo ""
echo "❌ Erro: 'Incorrect API key provided'"
echo "   → Sua chave está incorreta. Copie novamente de platform.openai.com/api-keys"
echo ""
echo "❌ Erro: 'Insufficient quota'"
echo "   → Sua conta OpenAI não tem créditos. Adicione em platform.openai.com/account/billing"
echo ""
echo "❌ Erro: 'Budget has been exceeded'"
echo "   → Você está usando chave Emergent sem créditos. Use sua chave OpenAI."
echo ""
echo "❌ Erro: Request failed / Network error"
echo "   → Problema de conexão. Verifique internet e tente novamente."
echo ""

echo "✅ PRÓXIMOS PASSOS:"
echo ""
echo "1. Certifique-se de ter uma chave OpenAI válida"
echo "2. Verifique se tem créditos em platform.openai.com/account/billing"
echo "3. Configure a chave clicando no ⚙️ no canto inferior direito"
echo "4. Teste enviando uma mensagem no chat"
echo ""
echo "======================================"
