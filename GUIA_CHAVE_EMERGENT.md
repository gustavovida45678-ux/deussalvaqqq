# 🔑 Guia Completo: Chave API Emergent Universal

## 🎯 O que é a Chave Emergent Universal?

A **Chave Emergent Universal (Universal Key)** é uma chave única que funciona com:
- ✅ OpenAI (GPT-4, GPT-5, DALL-E)
- ✅ Anthropic Claude
- ✅ Google Gemini (incluindo Nano Banana para imagens)

**Vantagem**: Uma única chave para múltiplos provedores!

---

## 📍 Como Encontrar Sua Chave Emergent

### Método 1: Via Dashboard Emergent (Recomendado)

1. **Acesse o Emergent Dashboard**
   - URL: https://emergentagent.com (ou seu domínio Emergent)
   - Ou clique no menu/perfil da aplicação

2. **Vá para Perfil/Profile**
   - Procure por "Profile" ou ícone de usuário
   - Geralmente no canto superior direito

3. **Acesse "Universal Key"**
   - Clique em **"Profile"** → **"Universal Key"**
   - Ou procure menu: **"Settings"** → **"API Keys"**

4. **Copie sua chave**
   - Você verá uma chave tipo: `sk-emergent-xxxxx...`
   - Clique em "Copy" ou "Show Key"

### Método 2: Via Linha de Comando (Se tiver acesso ao servidor)

Se você tem acesso ao servidor/backend:

```bash
# Ver a chave no arquivo .env
cat /app/backend/.env | grep EMERGENT_LLM_KEY
```

Resultado:
```
EMERGENT_LLM_KEY=sk-emergent-xxxxx...
```

### Método 3: Via Ferramenta Emergent (Dentro do Chat)

Se estiver usando o Emergent Agent, pode pedir:

```
@emergent show my universal key
```

ou

```
@emergent account info
```

---

## 💰 Como Adicionar Créditos à Chave Emergent

### Passo a Passo:

1. **Acesse o Dashboard Emergent**
   - Entre na sua conta

2. **Vá para Billing/Créditos**
   - **Profile** → **Universal Key** → **Add Balance**
   - Ou: **Billing** → **Add Credits**

3. **Escolha o valor**
   - Mínimo: geralmente $5
   - Recomendado: $10-$20 para uso normal
   - Pesado: $50+ para uso intensivo

4. **Método de Pagamento**
   - Cartão de crédito
   - PayPal (se disponível)
   - Outros métodos aceitos pelo Emergent

5. **Confirme**
   - Créditos aparecem imediatamente ou em poucos minutos

### Configurar Auto Top-up (Recomendado)

Para não ficar sem créditos:

1. **Profile** → **Universal Key** → **Auto Top-up**
2. Configure:
   - Limite mínimo: Ex: $5
   - Valor de recarga: Ex: $10
   - Ativa quando saldo < $5, adiciona $10 automaticamente

---

## 📊 Como Verificar Saldo/Uso

### Via Dashboard:

1. **Profile** → **Universal Key** → **Usage**
2. Você verá:
   - Saldo atual
   - Uso hoje/semana/mês
   - Histórico de transações
   - Breakdown por modelo (GPT-4, GPT-5, etc.)

### Estimativa de Custos:

| Operação | Custo Médio |
|----------|-------------|
| Mensagem texto (GPT-4) | $0.001 - $0.01 |
| Análise de imagem | $0.05 - $0.10 |
| Geração de imagem | $0.02 - $0.04 |
| Chat longo | $0.05 - $0.20 |

**Com $10 você consegue:**
- ~100-200 análises de gráficos
- ~500 mensagens de texto
- ~200 gerações de imagem

---

## ⚙️ Usar Chave Emergent na Aplicação

### Opção 1: Já está Configurada (Padrão)

A aplicação já vem com a chave Emergent configurada!

**Para usar:**
1. Abra configurações (⚙️)
2. Selecione **"🔑 Chave Emergent (Universal)"**
3. Salve
4. Pronto!

### Opção 2: Ver Qual Chave Está Sendo Usada

Abra console do navegador (F12):

```javascript
console.log('Provider:', localStorage.getItem('api_provider'));
```

**Se mostrar:**
- `emergent` ou `null` → Usando chave Emergent ✅
- `custom` → Usando sua chave própria

**Para voltar a usar Emergent:**

```javascript
localStorage.setItem('api_provider', 'emergent');
localStorage.removeItem('user_api_key');
location.reload();
```

---

## 🔍 Troubleshooting - Chave Emergent

### ❌ Erro: "Budget has been exceeded"

**Causa:** Sem créditos na chave Emergent

**Solução:**
1. Acesse: Profile → Universal Key → Add Balance
2. Adicione pelo menos $5
3. Aguarde 1-2 minutos
4. Tente novamente

### ❌ Erro: "Invalid API key"

**Causa:** Chave Emergent inválida ou expirada

**Solução:**
1. Verifique se está logado no Emergent
2. Gere uma nova chave em Profile → Universal Key
3. Atualize no arquivo `.env` do backend:
   ```bash
   nano /app/backend/.env
   # Edite a linha EMERGENT_LLM_KEY
   ```
4. Reinicie o backend:
   ```bash
   sudo supervisorctl restart backend
   ```

### ❌ Não consigo acessar o Dashboard

**Soluções:**

1. **URL correta?**
   - Verifique o domínio Emergent
   - Geralmente: `emergentagent.com` ou `app.emergent.ai`

2. **Sessão expirada?**
   - Faça logout e login novamente

3. **Esqueceu senha?**
   - Use "Forgot Password" no login

4. **Sem acesso ao dashboard?**
   - Se você só tem acesso ao servidor, use:
   ```bash
   # Ver chave atual
   cat /app/backend/.env | grep EMERGENT_LLM_KEY
   ```

---

## 🆚 Comparação: Emergent vs OpenAI Própria

| Aspecto | Chave Emergent | Chave OpenAI Própria |
|---------|----------------|----------------------|
| **Setup** | ✅ Já configurado | ⚙️ Precisa configurar |
| **Créditos Grátis** | 🎁 Depende do plano | 🎁 $5-$18 (novos usuários) |
| **Múltiplos Provedores** | ✅ OpenAI + Claude + Gemini | ❌ Só OpenAI |
| **Billing** | 💳 Dashboard Emergent | 💳 Dashboard OpenAI |
| **Limite** | 📊 Compartilhado (se multi-user) | 📊 Só seu |
| **Suporte** | 🤝 Emergent Team | 🤝 OpenAI Support |

---

## 💡 Recomendações

### Use Chave Emergent se:
- ✅ Quer simplicidade (já configurado)
- ✅ Usa múltiplos modelos (GPT + Claude + Gemini)
- ✅ Prefere gerenciar tudo no Emergent
- ✅ Tem plano/créditos Emergent

### Use Chave Própria se:
- ✅ Uso muito intensivo (100+ análises/dia)
- ✅ Quer controle total dos custos
- ✅ Já tem conta OpenAI com créditos
- ✅ Precisa de quota maior

---

## 📞 Precisa de Ajuda?

### Suporte Emergent:

- **Email**: support@emergent.ai (exemplo)
- **Chat**: No dashboard Emergent
- **Docs**: docs.emergent.ai

### Links Úteis:

- Dashboard: https://emergentagent.com
- Documentação: https://docs.emergent.ai
- Status: https://status.emergent.ai

---

## ✅ Checklist Rápido

Para usar Chave Emergent na aplicação:

- [ ] Tenho conta no Emergent
- [ ] Consigo acessar o dashboard
- [ ] Tenho créditos na Universal Key (ou acabei de adicionar)
- [ ] Configurei no app para usar "Chave Emergent"
- [ ] Testei enviando uma mensagem
- [ ] Funcionou! ✨

---

## 🚀 Atalhos Rápidos

### Ver chave no servidor:
```bash
cat /app/backend/.env | grep EMERGENT_LLM_KEY
```

### Testar chave Emergent:
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Teste"}'
```

### Configurar no navegador:
```javascript
localStorage.setItem('api_provider', 'emergent');
location.reload();
```

---

**Pronto! Agora você sabe tudo sobre a Chave Emergent Universal!** 🎉
