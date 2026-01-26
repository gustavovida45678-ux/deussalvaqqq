import { useState, useEffect, useRef } from "react";
import "@/App.css";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Image as ImageIcon, X, Sparkles, Settings } from "lucide-react";
import ApiKeySettings from "./components/ApiKeySettings";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImages, setSelectedImages] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [showImageGenModal, setShowImageGenModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [imageGenPrompt, setImageGenPrompt] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    // Load existing messages
    const loadMessages = async () => {
      try {
        const response = await axios.get(`${API}/messages`);
        setMessages(response.data);
      } catch (error) {
        console.error("Error loading messages:", error);
      }
    };
    loadMessages();
  }, []);

  // Drag and drop handlers
  useEffect(() => {
    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    };

    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
    };

    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleMultipleFiles(files);
      }
    };

    const handlePaste = (e) => {
      const items = e.clipboardData?.items;
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.startsWith("image/")) {
            const file = items[i].getAsFile();
            if (file) {
              handleImageFile(file);
              e.preventDefault();
            }
          }
        }
      }
    };

    document.addEventListener("dragover", handleDragOver);
    document.addEventListener("dragleave", handleDragLeave);
    document.addEventListener("drop", handleDrop);
    document.addEventListener("paste", handlePaste);

    return () => {
      document.removeEventListener("dragover", handleDragOver);
      document.removeEventListener("dragleave", handleDragLeave);
      document.removeEventListener("drop", handleDrop);
      document.removeEventListener("paste", handlePaste);
    };
  }, []);

  const handleImageFile = (file) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      alert("Por favor, selecione apenas arquivos de imagem (PNG, JPG, WEBP, etc.)");
      return;
    }
    
    // Add to existing images
    setSelectedImages(prev => [...prev, file]);
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreviews(prev => [...prev, { file: file.name, preview: reader.result }]);
    };
    reader.readAsDataURL(file);
  };

  const handleMultipleFiles = (files) => {
    Array.from(files).forEach(file => {
      if (file.type.startsWith("image/")) {
        handleImageFile(file);
      }
    });
  };

  const handleImageSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleMultipleFiles(files);
    }
  };

  const clearImage = (index) => {
    setSelectedImages(prev => prev.filter((_, i) => i !== index));
    setImagePreviews(prev => prev.filter((_, i) => i !== index));
  };

  const clearAllImages = () => {
    setSelectedImages([]);
    setImagePreviews([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && selectedImages.length === 0) || isLoading) return;

    const userMessage = inputMessage.trim() || "Analise estas imagens";
    
    // Check if user wants to generate an image with commands like "/gerar", "/imagem", etc.
    if (userMessage.toLowerCase().startsWith("/gerar ") || 
        userMessage.toLowerCase().startsWith("/imagem ") ||
        userMessage.toLowerCase().startsWith("/criar ")) {
      const prompt = userMessage.split(" ").slice(1).join(" ");
      if (prompt) {
        handleGenerateImage(prompt);
        setInputMessage("");
        return;
      }
    }
    
    setInputMessage("");
    setIsLoading(true);

    // Get custom API key if available
    const customApiKey = localStorage.getItem("user_api_key");
    const apiProvider = localStorage.getItem("api_provider");

    try {
      if (selectedImages.length > 0) {
        // Send images with message
        const formData = new FormData();
        selectedImages.forEach((image, index) => {
          formData.append("files", image);
        });
        formData.append("question", userMessage);

        // Add custom API key to headers if available
        const headers = {
          "Content-Type": "multipart/form-data",
        };
        if (apiProvider === "custom" && customApiKey) {
          headers["X-Custom-API-Key"] = customApiKey;
        }

        const response = await axios.post(`${API}/chat/images`, formData, { headers });

        // Add annotated image URLs to assistant message if available
        const assistantMsg = response.data.assistant_message;
        if (response.data.annotated_image_paths) {
          assistantMsg.annotated_image_urls = response.data.annotated_image_paths;
        }

        setMessages((prev) => [
          ...prev,
          response.data.user_message,
          assistantMsg,
        ]);
        clearAllImages();
      } else {
        // Send text only
        const headers = {};
        if (apiProvider === "custom" && customApiKey) {
          headers["X-Custom-API-Key"] = customApiKey;
        }

        const response = await axios.post(`${API}/chat`, {
          message: userMessage,
        }, { headers });

        setMessages((prev) => [
          ...prev,
          response.data.user_message,
          response.data.assistant_message,
        ]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Extract error message from response
      let errorMessage = "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.";
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "user",
          content: userMessage,
          timestamp: new Date().toISOString(),
        },
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: errorMessage,
          timestamp: new Date().toISOString(),
        },
      ]);
      clearAllImages();
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleGenerateImage = async (prompt) => {
    setIsGeneratingImage(true);
    setShowImageGenModal(false);
    setImageGenPrompt("");

    // Get custom API key if available
    const customApiKey = localStorage.getItem("user_api_key");
    const apiProvider = localStorage.getItem("api_provider");

    try {
      const headers = {};
      if (apiProvider === "custom" && customApiKey) {
        headers["X-Custom-API-Key"] = customApiKey;
      }

      const response = await axios.post(`${API}/generate-image`, {
        prompt: prompt,
        number_of_images: 1,
      }, { headers });

      setMessages((prev) => [
        ...prev,
        response.data.user_message,
        response.data.assistant_message,
      ]);
    } catch (error) {
      console.error("Error generating image:", error);
      
      // Extract error message from response
      let errorMessage = "Desculpe, ocorreu um erro ao gerar a imagem. Por favor, tente novamente.";
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "user",
          content: `🎨 Gerar imagem: ${prompt}`,
          timestamp: new Date().toISOString(),
        },
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: errorMessage,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsGeneratingImage(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="App neural-void-bg">
      <div className="noise-overlay" />

      {/* Settings Button */}
      <button
        className="settings-button"
        onClick={() => setShowSettingsModal(true)}
        title="Configurações de API"
        data-testid="settings-button"
      >
        <Settings size={24} />
      </button>

      {/* Settings Modal */}
      {showSettingsModal && (
        <ApiKeySettings onClose={() => setShowSettingsModal(false)} />
      )}

      {/* Drag and drop overlay */}
      {isDragging && (
        <div className="drag-drop-overlay" data-testid="drag-drop-overlay">
          <div className="drag-drop-content">
            <ImageIcon size={64} />
            <p>Solte a imagem aqui</p>
          </div>
        </div>
      )}

      {/* Image generation modal */}
      {showImageGenModal && (
        <div className="modal-overlay" onClick={() => setShowImageGenModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>🎨 Gerar Imagem com IA</h2>
              <button
                className="modal-close"
                onClick={() => setShowImageGenModal(false)}
                data-testid="close-modal-btn"
              >
                <X size={24} />
              </button>
            </div>
            <div className="modal-body">
              <p className="modal-description">
                Descreva a imagem que deseja criar. Seja o mais específico possível para melhores resultados.
              </p>
              <textarea
                value={imageGenPrompt}
                onChange={(e) => setImageGenPrompt(e.target.value)}
                placeholder="Ex: Uma paisagem futurista com montanhas flutuantes e cachoeiras luminosas ao pôr do sol..."
                className="image-gen-textarea"
                rows={4}
                data-testid="image-gen-prompt"
                autoFocus
              />
              <div className="modal-footer">
                <button
                  className="btn-secondary"
                  onClick={() => setShowImageGenModal(false)}
                >
                  Cancelar
                </button>
                <button
                  className="btn-primary"
                  onClick={() => {
                    if (imageGenPrompt.trim()) {
                      handleGenerateImage(imageGenPrompt.trim());
                    }
                  }}
                  disabled={!imageGenPrompt.trim()}
                  data-testid="generate-image-btn"
                >
                  <Sparkles size={18} />
                  Gerar Imagem
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {messages.length === 0 && !isLoading ? (
        <div className="empty-state">
          <h1 data-testid="welcome-heading">Chat GPT</h1>
          <p data-testid="welcome-message">
            Bem-vindo! Comece uma conversa digitando sua mensagem ou enviando uma imagem.
            <br />
            <span className="tip-text">
              💡 Você também pode arrastar imagens ou usar Ctrl+V para colar
              <br />
              🎨 Use /gerar [descrição] para criar imagens com IA
            </span>
          </p>
        </div>
      ) : (
        <div className="chat-container" data-testid="chat-container">
          {messages.map((message) => (
            <div
              key={message.id}
              data-testid={`message-${message.role}`}
              className={
                message.role === "user"
                  ? "message-bubble-user"
                  : "message-bubble-ai"
              }
            >
              {message.image_urls && message.image_urls.length > 0 && (
                <div className="message-images-grid">
                  {message.image_urls.map((url, idx) => (
                    <img
                      key={idx}
                      src={`${BACKEND_URL}${url}`}
                      alt={`Uploaded ${idx + 1}`}
                      className="message-image"
                    />
                  ))}
                </div>
              )}
              {message.image_url && (
                <img
                  src={`${BACKEND_URL}${message.image_url}`}
                  alt="Uploaded"
                  className="message-image"
                />
              )}
              {/* Show annotated images if available */}
              {message.annotated_image_urls && message.annotated_image_urls.length > 0 && (
                <div className="annotated-images-section">
                  <div className="annotated-label">📊 Análise Visual com Recomendações</div>
                  <div className="message-images-grid">
                    {message.annotated_image_urls.map((url, idx) => (
                      url && (
                        <img
                          key={`annotated-${idx}`}
                          src={`${BACKEND_URL}${url}`}
                          alt={`Gráfico Anotado ${idx + 1}`}
                          className="message-image annotated-image"
                        />
                      )
                    ))}
                  </div>
                </div>
              )}
              {message.role === "user" ? (
                <div>{message.content}</div>
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="typing-indicator" data-testid="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          )}

          {isGeneratingImage && (
            <div className="generating-indicator" data-testid="generating-indicator">
              <Sparkles className="sparkle-icon" size={32} />
              <p>Gerando imagem com IA...</p>
              <span className="generating-subtext">Isso pode levar até 1 minuto</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      )}

      <form
        onSubmit={handleSendMessage}
        className="input-command-bar"
        data-testid="chat-form"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageSelect}
          accept="image/*"
          multiple
          style={{ display: "none" }}
        />

        {imagePreviews.length > 0 && (
          <div className="images-preview-container">
            {imagePreviews.map((img, index) => (
              <div key={index} className="image-preview-wrapper">
                <img src={img.preview} alt={`Preview ${index + 1}`} className="image-preview" />
                <button
                  type="button"
                  onClick={() => clearImage(index)}
                  className="clear-image-btn"
                  data-testid={`clear-image-btn-${index}`}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
            {imagePreviews.length > 1 && (
              <button
                type="button"
                onClick={clearAllImages}
                className="clear-all-btn"
                data-testid="clear-all-images-btn"
              >
                Limpar Todas
              </button>
            )}
          </div>
        )}

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="image-upload-btn"
          disabled={isLoading || isGeneratingImage}
          data-testid="image-upload-btn"
          title="Adicionar imagens (múltiplas)"
        >
          <ImageIcon size={20} />
          {imagePreviews.length > 0 && (
            <span className="image-count">{imagePreviews.length}</span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setShowImageGenModal(true)}
          className="image-gen-btn"
          disabled={isLoading || isGeneratingImage}
          data-testid="image-gen-open-btn"
          title="Gerar imagem com IA"
        >
          <Sparkles size={20} />
        </button>

        <input
          ref={inputRef}
          type="text"
          data-testid="message-input"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={
            imagePreviews.length > 0
              ? `${imagePreviews.length} imagem(ns) selecionada(s)...`
              : "Digite sua mensagem ou envie imagens..."
          }
          disabled={isLoading}
        />
        <button
          type="submit"
          data-testid="send-button"
          disabled={isLoading || isGeneratingImage || (!inputMessage.trim() && selectedImages.length === 0)}
        >
          <Send size={18} />
          Enviar
        </button>
      </form>
    </div>
  );
}

export default App;
