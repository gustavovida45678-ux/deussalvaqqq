from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # 'user' or 'assistant'
    content: str
    image_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    user_message: Message
    assistant_message: Message

class ImageAnalysisResponse(BaseModel):
    image_id: str
    image_path: str
    user_message: Message
    assistant_message: Message


# Chat endpoint
@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Create user message
        user_message = Message(
            role="user",
            content=request.message
        )
        
        # Save user message to database
        user_doc = user_message.model_dump()
        user_doc['timestamp'] = user_doc['timestamp'].isoformat()
        await db.messages.insert_one(user_doc)
        
        # Initialize LLM chat
        chat_client = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id="chat-session",
            system_message="Você é um assistente útil e amigável. Responda em português de forma clara e concisa."
        )
        chat_client.with_model("openai", "gpt-5.1")
        
        # Send message to AI
        user_msg = UserMessage(text=request.message)
        ai_response = await chat_client.send_message(user_msg)
        
        # Create assistant message
        assistant_message = Message(
            role="assistant",
            content=ai_response
        )
        
        # Save assistant message to database
        assistant_doc = assistant_message.model_dump()
        assistant_doc['timestamp'] = assistant_doc['timestamp'].isoformat()
        await db.messages.insert_one(assistant_doc)
        
        return ChatResponse(
            user_message=user_message,
            assistant_message=assistant_message
        )
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


# Image analysis endpoint
@api_router.post("/chat/image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(default="Faça uma análise técnica completa deste gráfico: identifique o ativo, timeframe, tendência, padrões de candlestick, níveis de suporte/resistência, indicadores visíveis, e forneça projeções com estimativas de próximos movimentos, incluindo probabilidades e recomendações de entrada (COMPRA/VENDA) com níveis de stop loss e take profit.")
):
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida")
        
        # Read image
        image_bytes = await file.read()
        
        # Validate image is not empty
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Imagem está vazia")
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Save image locally
        os.makedirs("uploads", exist_ok=True)
        image_id = str(uuid.uuid4())
        image_filename = f"{image_id}_{file.filename}"
        image_path = f"uploads/{image_filename}"
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        # Create user message with image
        user_message = Message(
            role="user",
            content=question,
            image_url=f"/uploads/{image_filename}"
        )
        
        # Save user message to database
        user_doc = user_message.model_dump()
        user_doc['timestamp'] = user_doc['timestamp'].isoformat()
        await db.messages.insert_one(user_doc)
        
        # Initialize LLM chat with vision model
        chat_client = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id="vision-session",
            system_message="""Você é um analista técnico profissional especializado em análise de gráficos de trading e mercado financeiro.

INSTRUÇÕES PARA ANÁLISE DE GRÁFICOS:

1. IDENTIFICAÇÃO DO ATIVO E TIMEFRAME:
   - Identifique o par/ativo sendo negociado (ex: EUR/USD, BTC/USD, etc.)
   - Determine o timeframe do gráfico (ex: M1, M5, M15, H1, H4, D1)
   - Anote o horário atual e preço atual

2. ANÁLISE TÉCNICA COMPLETA:
   - **Tendência Principal**: Identifique se está em tendência de alta, baixa ou lateral
   - **Padrões de Candlestick**: Identifique padrões (Doji, Hammer, Engulfing, etc.)
   - **Níveis de Suporte e Resistência**: Marque os níveis-chave onde o preço reagiu
   - **Estrutura de Mercado**: Identifique topos e fundos, rompimentos, pullbacks
   - **Volume**: Observe se há indicadores de volume e o que indicam

3. INDICADORES TÉCNICOS (se visíveis):
   - Médias Móveis (posição e cruzamentos)
   - RSI (sobrecompra/sobrevenda)
   - MACD (divergências e cruzamentos)
   - Bandas de Bollinger
   - Fibonacci (retrações e extensões)
   - Outros indicadores visíveis

4. ANÁLISE DO MOMENTUM:
   - Determine se o momentum é forte, fraco ou neutro
   - Identifique divergências entre preço e indicadores
   - Avalie a força da tendência atual

5. PROJEÇÕES E ESTIMATIVAS:
   - **Próxima Resistência/Suporte**: Onde o preço provavelmente reagirá
   - **Cenários Possíveis**: 
     * Cenário Alta: Próximos alvos, condições necessárias
     * Cenário Baixa: Próximos alvos, condições necessárias
     * Cenário Lateral: Faixas de consolidação
   - **Probabilidade**: Estime probabilidades baseadas na análise técnica
   - **Stop Loss e Take Profit**: Sugira níveis prudentes

6. SINAIS DE ENTRADA (se solicitado):
   - Condições para entrada COMPRA (CALL/BUY)
   - Condições para entrada VENDA (PUT/SELL)
   - Timeframe recomendado para a operação
   - Gestão de risco (% do capital)

7. CONTEXTO DE MERCADO:
   - Identifique se estamos perto de aberturas/fechamentos importantes
   - Note qualquer evento econômico relevante (se visível)
   - Avalie a volatilidade atual

8. CONCLUSÃO E RECOMENDAÇÕES:
   - Resuma a análise em 3-4 pontos principais
   - Dê uma recomendação clara (COMPRA, VENDA, ou AGUARDAR)
   - Indique o nível de confiança da análise (%)
   - Destaque os principais riscos

FORMATO DA RESPOSTA:
Use markdown com seções claras, bullet points, e **destaque** para informações importantes.
Seja específico com números (preços, percentuais, timeframes).
Forneça análise profunda como um analista técnico experiente faria.

Responda SEMPRE em português brasileiro de forma profissional e detalhada."""
        )
        chat_client.with_model("openai", "gpt-5.1")
        
        # Create image content
        image_content = ImageContent(image_base64=image_base64)
        
        # Send message with image to AI
        user_msg = UserMessage(
            text=question,
            file_contents=[image_content]
        )
        ai_response = await chat_client.send_message(user_msg)
        
        # Create assistant message
        assistant_message = Message(
            role="assistant",
            content=ai_response
        )
        
        # Save assistant message to database
        assistant_doc = assistant_message.model_dump()
        assistant_doc['timestamp'] = assistant_doc['timestamp'].isoformat()
        await db.messages.insert_one(assistant_doc)
        
        return ImageAnalysisResponse(
            image_id=image_id,
            image_path=image_path,
            user_message=user_message,
            assistant_message=assistant_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in image analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@api_router.get("/messages", response_model=List[Message])
async def get_messages():
    """Get all chat messages"""
    messages = await db.messages.find({}, {"_id": 0}).sort("timestamp", 1).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for msg in messages:
        if isinstance(msg['timestamp'], str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    
    return messages


@api_router.delete("/messages")
async def clear_messages():
    """Clear all chat messages"""
    result = await db.messages.delete_many({})
    return {"deleted_count": result.deleted_count}


@api_router.get("/")
async def root():
    return {"message": "Chat API is running"}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()