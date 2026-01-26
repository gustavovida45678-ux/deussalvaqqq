import { useState } from "react";
import { Settings, X, Key, Check, AlertCircle } from "lucide-react";

export default function ApiKeySettings({ onClose }) {
  const [apiKey, setApiKey] = useState(localStorage.getItem("user_api_key") || "");
  const [provider, setProvider] = useState(localStorage.getItem("api_provider") || "emergent");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    if (provider === "custom" && !apiKey.trim()) {
      alert("Por favor, insira uma API key válida!");
      return;
    }

    if (provider === "custom") {
      localStorage.setItem("user_api_key", apiKey);
      localStorage.setItem("api_provider", "custom");
    } else {
      localStorage.removeItem("user_api_key");
      localStorage.setItem("api_provider", "emergent");
    }

    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      onClose();
      window.location.reload(); // Reload to apply changes
    }, 1500);
  };

  const handleClear = () => {
    localStorage.removeItem("user_api_key");
    localStorage.removeItem("api_provider");
    setApiKey("");
    setProvider("emergent");
    alert("Configurações resetadas! Usando chave Emergent.");
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="flex items-center gap-2">
            <Settings size={24} />
            <h2>⚙️ Configurações de API</h2>
          </div>
          <button className="modal-close" onClick={onClose}>
            <X size={24} />
          </button>
        </div>

        <div className="modal-body">
          {/* Provider Selection */}
          <div className="setting-section">
            <label className="setting-label">
              <Key size={18} />
              Provedor de API
            </label>
            <select 
              value={provider} 
              onChange={(e) => setProvider(e.target.value)}
              className="setting-select"
            >
              <option value="emergent">🔑 Chave Emergent (Universal)</option>
              <option value="custom">🔐 Minha Própria Chave OpenAI</option>
            </select>
            <p className="setting-description">
              {provider === "emergent" 
                ? "Usando a chave universal Emergent (requer créditos)" 
                : "Use sua própria API key OpenAI (sem limites do Emergent)"}
            </p>
          </div>

          {/* Custom API Key Input */}
          {provider === "custom" && (
            <div className="setting-section">
              <label className="setting-label">
                <Key size={18} />
                Sua API Key OpenAI
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="setting-input"
              />
              <p className="setting-description">
                Sua chave será armazenada localmente no navegador (não no servidor)
              </p>
            </div>
          )}

          {/* Instructions */}
          <div className="instructions-box">
            <div className="instructions-header">
              <AlertCircle size={18} />
              <span>Como obter uma API Key OpenAI</span>
            </div>
            <ol className="instructions-list">
              <li>Acesse: <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="link">platform.openai.com/api-keys</a></li>
              <li>Faça login ou crie uma conta OpenAI</li>
              <li>Clique em "Create new secret key"</li>
              <li>Copie a chave (começa com "sk-...")</li>
              <li>Cole aqui e salve</li>
            </ol>
            <p className="instructions-note">
              💡 <strong>Dica:</strong> Você precisará adicionar créditos na sua conta OpenAI
            </p>
          </div>

          {/* Current Status */}
          <div className="status-box">
            <strong>Status Atual:</strong>
            <span className={`status-badge ${provider === "custom" ? "status-custom" : "status-emergent"}`}>
              {provider === "custom" ? "🔐 Chave Própria Configurada" : "🔑 Usando Chave Emergent"}
            </span>
          </div>

          {/* Action Buttons */}
          <div className="modal-footer">
            <button className="btn-secondary" onClick={handleClear}>
              Limpar
            </button>
            <button 
              className="btn-primary" 
              onClick={handleSave}
              disabled={saved}
            >
              {saved ? (
                <>
                  <Check size={18} />
                  Salvo!
                </>
              ) : (
                <>
                  <Key size={18} />
                  Salvar Configurações
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
