# 🔑 Guia Completo: Como Obter e Usar Sua Própria API Key

## 📋 Índice
1. [Por que usar sua própria chave?](#por-que)
2. [Como obter API Key OpenAI](#openai)
3. [Como configurar na aplicação](#config)
4. [Custos e Limites](#custos)
5. [Solução de Problemas](#troubleshooting)

---

## 🎯 Por que usar sua própria chave? {#por-que}

### Vantagens:
- ✅ **Sem limites de budget** da chave Emergent
- ✅ **Controle total** sobre seus gastos
- ✅ **Maior quota** de requisições
- ✅ **Modelos mais recentes** disponíveis
- ✅ **Créditos gratuitos** para novos usuários ($5-$18)

### Quando usar:
- Uso intensivo da aplicação
- Muitas análises de gráficos por dia
- Preferência por controle financeiro direto

---

## 🔐 Como obter API Key OpenAI {#openai}

### Passo 1: Criar Conta OpenAI

1. Acesse: **https://platform.openai.com**
2. Clique em **"Sign up"** (ou "Get started")
3. Escolha uma opção:
   - Email + senha
   - Login com Google
   - Login com Microsoft
   - Login com Apple

### Passo 2: Verificar Email (se necessário)

- Verifique sua caixa de entrada
- Clique no link de verificação enviado pela OpenAI
- Complete o perfil se solicitado

### Passo 3: Adicionar Método de Pagamento

⚠️ **IMPORTANTE**: Para usar a API, você precisa adicionar um cartão de crédito.

1. Vá para: **https://platform.openai.com/account/billing**
2. Clique em **"Add payment method"**
3. Insira dados do cartão de crédito
4. Confirme

### Passo 4: Criar API Key

1. Acesse: **https://platform.openai.com/api-keys**
2. Clique em **"+ Create new secret key"**
3. (Opcional) Dê um nome descritivo:
   - Exemplo: "Trading Analysis App"
4. Clique em **"Create secret key"**
5. **COPIE A CHAVE IMEDIATAMENTE** ⚠️
   - Começa com `sk-proj-...` ou `sk-...`
   - Você não poderá ver ela novamente!

### Passo 5: Configurar Limites (Recomendado)

Para evitar gastos inesperados:

1. Vá para: **https://platform.openai.com/account/limits**
2. Configure:
   - **Limite Mensal**: Ex: $10/mês
   - **Limite de Uso Único**: Ex: $5 por requisição
3. Salve as configurações

---

## ⚙️ Como configurar na aplicação {#config}

### Método 1: Interface Web (Recomendado)

1. **Abra a aplicação**
2. **Clique no ícone de engrenagem** ⚙️ (canto inferior direito)
3. **Selecione** "🔐 Minha Própria Chave OpenAI"
4. **Cole sua API key** no campo
5. **Clique em "Salvar Configurações"**
6. **Pronto!** A página recarregará automaticamente

### Método 2: Usar console do navegador

```javascript
// Cole no console (F12)
localStorage.setItem("user_api_key", "sk-proj-SEU_TOKEN_AQUI");
localStorage.setItem("api_provider", "custom");
// Recarregue a página
location.reload();
```

### Verificação

Após configurar, você verá:
- ✅ Status: **"🔐 Chave Própria Configurada"**
- Banner verde no modal de configurações

---

## 💰 Custos e Limites {#custos}

### Preços OpenAI (GPT-4)

**Para análise de gráficos (visão):**
- GPT-4 Vision: ~$0.03 por 1K tokens
- Imagem (1024x1024): ~$0.01 por imagem

**Exemplo de uso:**
- 1 análise de gráfico: ~$0.05 - $0.10
- 10 análises: ~$0.50 - $1.00
- 100 análises: ~$5.00 - $10.00

### Créditos Gratuitos

**Novos usuários recebem:**
- $5 - $18 em créditos gratuitos (varia por região)
- Válido por 3 meses
- Suficiente para ~50-180 análises de gráficos

### Gerenciamento de Gastos

**Monitoramento:**
- Dashboard: https://platform.openai.com/usage
- Veja gastos em tempo real
- Histórico detalhado por dia/mês

**Alertas:**
- Configure em: https://platform.openai.com/account/billing
- Receba email quando atingir % do limite
- Bloqueio automático ao atingir limite máximo

---

## 🔧 Solução de Problemas {#troubleshooting}

### Erro: "Invalid API Key"

**Causas:**
- Chave copiada incorretamente
- Espaços extras no início/fim
- Chave revogada

**Solução:**
1. Verifique se copiou a chave completa
2. Remove espaços em branco
3. Crie uma nova chave se necessário

### Erro: "Insufficient Quota"

**Causa:** Sem créditos na conta OpenAI

**Solução:**
1. Acesse: https://platform.openai.com/account/billing
2. Adicione créditos (mínimo $5)
3. Aguarde alguns minutos para processar

### Erro: "Rate Limit Exceeded"

**Causa:** Muitas requisições em curto período

**Solução:**
1. Aguarde 1 minuto
2. Tente novamente
3. Configure limites mais adequados

### Como voltar para chave Emergent?

**Opção 1: Via Interface**
1. Abra configurações (⚙️)
2. Selecione "🔑 Chave Emergent (Universal)"
3. Salve

**Opção 2: Via Console**
```javascript
localStorage.removeItem("user_api_key");
localStorage.setItem("api_provider", "emergent");
location.reload();
```

---

## 🎯 Recomendações

### Para uso casual (1-10 análises/dia):
✅ **Use chave Emergent** (mais simples)

### Para uso moderado (10-50 análises/dia):
✅ **Use sua própria chave** (melhor custo-benefício)

### Para uso intensivo (50+ análises/dia):
✅ **Definitivamente use sua própria chave**
✅ Configure limites de gasto
✅ Monitore uso diariamente

---

## 📚 Links Úteis

- **OpenAI Platform**: https://platform.openai.com
- **Criar API Keys**: https://platform.openai.com/api-keys
- **Gerenciar Billing**: https://platform.openai.com/account/billing
- **Ver Uso**: https://platform.openai.com/usage
- **Documentação**: https://platform.openai.com/docs
- **Preços**: https://openai.com/pricing

---

## ✅ Checklist Rápido

- [ ] Conta OpenAI criada
- [ ] Email verificado
- [ ] Cartão de crédito adicionado
- [ ] API Key criada e copiada
- [ ] Limites de gasto configurados
- [ ] Chave configurada na aplicação
- [ ] Teste realizado com sucesso

---

## 💡 Dicas Extras

1. **Segurança**: Nunca compartilhe sua API key
2. **Backup**: Salve a chave em local seguro (gerenciador de senhas)
3. **Múltiplas chaves**: Crie diferentes chaves para diferentes apps
4. **Rotação**: Troque sua chave periodicamente
5. **Revogação**: Se comprometida, revogue imediatamente em platform.openai.com

---

**Pronto!** Agora você tem controle total sobre o uso da IA na aplicação! 🎉
