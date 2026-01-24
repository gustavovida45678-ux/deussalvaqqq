import { useState, useEffect, useRef } from "react";
import "@/App.css";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Image as ImageIcon, X } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
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
        const file = files[0];
        if (file.type.startsWith("image/")) {
          handleImageFile(file);
        }
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
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageFile(file);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && !selectedImage) || isLoading) return;

    const userMessage = inputMessage.trim() || "Analise esta imagem";
    setInputMessage("");
    setIsLoading(true);

    try {
      if (selectedImage) {
        // Send image with message
        const formData = new FormData();
        formData.append("file", selectedImage);
        formData.append("question", userMessage);

        const response = await axios.post(`${API}/chat/image`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        setMessages((prev) => [
          ...prev,
          response.data.user_message,
          response.data.assistant_message,
        ]);
        clearImage();
      } else {
        // Send text only
        const response = await axios.post(`${API}/chat`, {
          message: userMessage,
        });

        setMessages((prev) => [
          ...prev,
          response.data.user_message,
          response.data.assistant_message,
        ]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
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
          content:
            "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.",
          timestamp: new Date().toISOString(),
        },
      ]);
      clearImage();
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="App neural-void-bg">
      <div className="noise-overlay" />

      {messages.length === 0 && !isLoading ? (
        <div className="empty-state">
          <h1 data-testid="welcome-heading">Chat GPT</h1>
          <p data-testid="welcome-message">
            Bem-vindo! Comece uma conversa digitando sua mensagem ou enviando uma imagem.
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
              {message.image_url && (
                <img
                  src={`${BACKEND_URL}${message.image_url}`}
                  alt="Uploaded"
                  className="message-image"
                />
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
          style={{ display: "none" }}
        />

        {imagePreview && (
          <div className="image-preview-container">
            <img src={imagePreview} alt="Preview" className="image-preview" />
            <button
              type="button"
              onClick={clearImage}
              className="clear-image-btn"
              data-testid="clear-image-btn"
            >
              <X size={16} />
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="image-upload-btn"
          disabled={isLoading}
          data-testid="image-upload-btn"
        >
          <ImageIcon size={20} />
        </button>

        <input
          ref={inputRef}
          type="text"
          data-testid="message-input"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Digite sua mensagem ou envie uma imagem..."
          disabled={isLoading}
        />
        <button
          type="submit"
          data-testid="send-button"
          disabled={isLoading || (!inputMessage.trim() && !selectedImage)}
        >
          <Send size={18} />
          Enviar
        </button>
      </form>
    </div>
  );
}

export default App;
