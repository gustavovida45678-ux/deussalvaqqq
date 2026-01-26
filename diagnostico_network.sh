#!/bin/bash

echo "🔍 DIAGNÓSTICO DE NETWORK ERROR"
echo "================================"
echo ""

# 1. Verificar serviços
echo "1️⃣ Status dos Serviços:"
sudo supervisorctl status | grep -E "backend|frontend"
echo ""

# 2. Testar backend localmente
echo "2️⃣ Testando Backend (localhost:8001):"
response=$(curl -s http://localhost:8001/api/ 2>&1)
if echo "$response" | grep -q "Chat API is running"; then
    echo "✅ Backend respondendo em localhost:8001"
else
    echo "❌ Backend NÃO está respondendo"
    echo "   Resposta: $response"
fi
echo ""

# 3. Verificar .env frontend
echo "3️⃣ Configuração Frontend (.env):"
cat /app/frontend/.env
echo ""

# 4. Testar CORS
echo "4️⃣ Testando CORS:"
curl -s -X OPTIONS http://localhost:8001/api/chat \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v 2>&1 | grep -i "access-control"
echo ""

# 5. Testar conexão frontend -> backend
echo "5️⃣ Teste de Conexão Frontend -> Backend:"
echo "   Frontend deveria estar usando: http://localhost:8001"
echo ""

# 6. Verificar portas
echo "6️⃣ Portas Abertas:"
netstat -tlnp | grep -E "8001|3000" | awk '{print $4, $7}'
echo ""

echo "================================"
echo ""
echo "🔧 SOLUÇÕES POSSÍVEIS:"
echo ""
echo "Se backend não responder:"
echo "  → sudo supervisorctl restart backend"
echo ""
echo "Se frontend não conectar:"
echo "  → Verifique se .env tem: REACT_APP_BACKEND_URL=http://localhost:8001"
echo "  → sudo supervisorctl restart frontend"
echo ""
echo "Se ainda tiver Network Error:"
echo "  → Limpe cache do navegador (Ctrl+Shift+Delete)"
echo "  → Abra em aba anônima"
echo "  → Verifique console do navegador (F12) para mais detalhes"
echo ""
