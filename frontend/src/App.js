import { useState, useEffect, useRef } from "react";
import "@/App.css";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

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

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${API}/chat`, {
        message: userMessage,
      });

      setMessages((prev) => [
        ...prev,
        response.data.user_message,
        response.data.assistant_message,
      ]);
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
          content: "Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.",
          timestamp: new Date().toISOString(),
        },
      ]);
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
            Bem-vindo! Comece uma conversa digitando sua mensagem abaixo.
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

      <form onSubmit={handleSendMessage} className="input-command-bar" data-testid="chat-form">
        <input
          ref={inputRef}
          type="text"
          data-testid="message-input"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Digite sua mensagem..."
          disabled={isLoading}
        />
        <button
          type="submit"
          data-testid="send-button"
          disabled={isLoading || !inputMessage.trim()}
        >
          <Send size={18} />
          Enviar
        </button>
      </form>
    </div>
  );
}

export default App;
