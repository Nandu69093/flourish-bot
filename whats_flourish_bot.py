from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request, Header, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader, CSVLoader
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
import requests
from typing import Dict
import shutil
from pydantic import BaseModel
from datetime import datetime, timezone
from bson import ObjectId
import pymongo

load_dotenv("credentials.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://aimsghub_db_user:<db_password>@cluster0.5ufyyk6.mongodb.net/?appName=Cluster0")

# MongoDB Setup
class MongoDB:
    def __init__(self, uri: str, db_name: str = "whatsflour_bot"):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.chats = self.db["chats"]
        self.sessions = self.db["sessions"]
        self.knowledge_bases = self.db["knowledge_bases"]
        self.messages = self.db["messages"]
        
        # Create indexes for better performance
        self.chats.create_index([("phone_number", 1), ("timestamp", -1)])
        self.sessions.create_index([("phone_number", 1)], unique=True)
        self.messages.create_index([("session_id", 1), ("timestamp", 1)])
        
        print("✅ MongoDB connected successfully!")
    
    def ping(self):
        """Test database connection"""
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            return False

# Initialize MongoDB
mongodb = MongoDB(MONGODB_URI)

class E5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/e5-large-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.instruction = "Given a sentence, retrieve semantically similar sentences: "

    def _last_token_pooling(self, hidden_states, attention_mask):
        last_non_padded_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=self.device)
        return hidden_states[batch_indices, last_non_padded_idx]

    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            texts_with_instruction = [self.instruction + t for t in batch_texts]
            inputs = self.tokenizer(
                texts_with_instruction, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = self._last_token_pooling(
                outputs.last_hidden_state, inputs['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def calculate_dynamic_chunk_size(text: str) -> tuple[int, int]:
    """Calculate dynamic chunk size and overlap based on text characteristics"""
    total_chars = len(text)
    dynamic_chunk_size = min(1000, max(200, total_chars // 20))
    dynamic_chunk_overlap = int(dynamic_chunk_size * 0.15)
    return dynamic_chunk_size, dynamic_chunk_overlap

BATCH_SIZE = 16

def analyze_intent(user_input):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "You are an intent classification system. Respond with only the most specific intent in 2-4 words."
            },
            {"role": "user", "content": user_input}],
            model="LLaMA-3.1-8B-Instant",
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in intent analysis -> {e}")
        return "unknown_intent"

def advanced_retrievers(vector_store, embedding_model):
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":5, "lambda_mult":0.6}
    )
    embeddings_filter = EmbeddingsFilter(
        embeddings=embedding_model,
        similarity_threshold=0.75
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=mmr_retriever
    )

def load_and_process_documents(file_path: str, embedding_model):
    """Load and process documents from provided file path, replacing existing knowledge base"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
    
    # Clear existing vector store
    try:
        if os.path.exists("./flourish_chroma_db"):
            shutil.rmtree("./flourish_chroma_db")
            print("Cleared existing knowledge base")
    except Exception as e:
        print(f"Error clearing existing vector store: {e}")

    documents = []
    
    # Load documents based on file type
    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded PDF document: {len(documents)} pages")
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        print(f"Loaded DOCX document: {len(documents)} sections")
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()
        print(f"Loaded TXT document: {len(documents)} sections")
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
        documents = loader.load()
        print(f"Loaded CSV document: {len(documents)} rows")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Calculate dynamic chunking
    full_text = "".join([doc.page_content for doc in documents])
    dynamic_chunk_size, dynamic_chunk_overlap = calculate_dynamic_chunk_size(full_text)
    
    print(f"Dynamic chunking - Size: {dynamic_chunk_size}, Overlap: {dynamic_chunk_overlap}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=dynamic_chunk_size,
        chunk_overlap=dynamic_chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")
    
    # Create new vector store
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="./flourish_chroma_db",
        collection_name="flourish_knowledge_base"
    )
    
    # Store knowledge base info in MongoDB
    kb_info = {
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1],
        "chunk_count": len(split_docs),
        "loaded_at": datetime.now(timezone.utc),
        "status": "active"
    }
    mongodb.knowledge_bases.insert_one(kb_info)
    
    print("Successfully created new vector store")
    return vector_store

def load_existing_vector_store(embedding_model):
    """Load existing vector store if available"""
    try:
        if os.path.exists("./flourish_chroma_db"):
            vector_store = Chroma(
                persist_directory="./flourish_chroma_db",
                embedding_function=embedding_model,
                collection_name="flourish_knowledge_base"
            )
            print("Loaded existing knowledge base from vector store")
            return vector_store
        else:
            print("No existing vector store found")
            return None
    except Exception as e:
        print(f"Error loading existing vector store: {e}")
        return None

def store_chat_message(phone_number: str, message_type: str, content: str, session_id: str = None, intent: str = None):
    """Store chat message in MongoDB"""
    try:
        chat_data = {
            "phone_number": phone_number,
            "message_type": message_type,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.now(timezone.utc),
            "session_id": session_id,
            "intent": intent
        }
        
        result = mongodb.chats.insert_one(chat_data)
        return result.inserted_id
    except Exception as e:
        print(f"Error storing chat message: {e}")
        return None

def get_chat_history(phone_number: str, limit: int = 10):
    """Get chat history for a phone number"""
    try:
        chats = mongodb.chats.find(
            {"phone_number": phone_number}
        ).sort("timestamp", -1).limit(limit)
        
        chat_list = []
        for chat in chats:
            chat_list.append({
                "type": chat["message_type"],
                "content": chat["content"],
                "timestamp": chat["timestamp"],
                "intent": chat.get("intent", "")
            })
        
        return chat_list[::-1]  # Reverse to get chronological order
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def store_user_session(phone_number: str, chat_history: list):
    """Store or update user session in MongoDB"""
    try:
        session_data = {
            "phone_number": phone_number,
            "chat_history": chat_history,
            "last_activity": datetime.now(timezone.utc),
            "message_count": len(chat_history)
        }
        
        mongodb.sessions.update_one(
            {"phone_number": phone_number},
            {"$set": session_data},
            upsert=True
        )
    except Exception as e:
        print(f"Error storing user session: {e}")

def get_user_session(phone_number: str):
    """Get user session from MongoDB"""
    try:
        session = mongodb.sessions.find_one({"phone_number": phone_number})
        return session
    except Exception as e:
        print(f"Error retrieving user session: {e}")
        return None

def chatbot(user_input, retriever, chat_history=None):
    try:
        if chat_history is None:
            chat_history = []
            
        chat_model = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=512)
        
        # Convert chat history to LangChain messages
        lc_chat_history = []
        for msg in chat_history[-6:]:  # Keep last 6 messages
            if msg['type'] == 'user':
                lc_chat_history.append(HumanMessage(content=msg['content']))
            else:
                lc_chat_history.append(AIMessage(content=msg['content']))

        detected_intent = analyze_intent(user_input)
        retrieved_docs = retriever.get_relevant_documents(user_input)
        retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant content found."
        fallback_response = "I'm sorry, I couldn't find relevant information to answer your question at the moment.Please contact at 9568504370 number for further assistance."

        prompt_template = PromptTemplate(
            template="""
You are a professional and friendly AI assistant(your name is Flourish Buddy) for Flourish Digital Pvt Ltd. 
Answer the user query clearly, concisely, and accurately using the knowledge base content provided.

USER QUERY: {query}
DETECTED INTENT: {detected_intent}

RELEVANT KNOWLEDGE BASE CONTENT:
{retrieved_doc}

INSTRUCTIONS:
- Use knowledge base content primarily to answer the query.
- Provide clear, easy-to-understand, and professional responses.
- Avoid vague statements or unnecessary disclaimers.
- If relevant knowledge is not found or the retrieved content is insufficient, respond only with the fallback response: "{fallback_response}".
- Do not include introductions, conclusions, or filler text.
- Respond politely and in a friendly tone, even for short acknowledgments like "ok", "alright", or "good" with emojis.
- Use appropriate emojis according to the whatsapp query context by the user.

RESPONSE:
""",
            input_variables=['query', 'detected_intent', 'retrieved_doc', 'fallback_response']
        )

        formatted_prompt = prompt_template.format(
            query=user_input,
            detected_intent=detected_intent,
            retrieved_doc=retrieved_content,
            fallback_response=fallback_response
        )

        system_message = SystemMessage(content=formatted_prompt)
        user_message = HumanMessage(content=user_input)

        messages = [system_message] + lc_chat_history + [user_message]

        response = chat_model.invoke(messages)

        # Update chat history
        chat_history.append({"type": "user", "content": user_input})
        chat_history.append({"type": "assistant", "content": response.content})

        if len(chat_history) > 10:  # Keep last 10 messages
            chat_history = chat_history[-10:]

        return response.content, chat_history, detected_intent
        
    except Exception as e:
        print(f"Error in chatbot response generation -> {e}")
        return "I'm sorry, I couldn't process your request at the moment.", chat_history, "error"

def send_whatsapp_message(phone_number: str, message: str):
    """Send message via Meta WhatsApp API"""
    try:
        url = f"https://graph.facebook.com/v22.0/{META_PHONE_NUMBER_ID}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "text": {"body": message}
        }
        
        headers = {
            "Authorization": f"Bearer {META_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"Message sent to {phone_number}")
            return True
        else:
            print(f"Failed to send message: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

# Pydantic models for request bodies
class KnowledgeBasePath(BaseModel):
    file_path: str

class MessageRequest(BaseModel):
    phone_number: str
    message: str

class ChatHistoryRequest(BaseModel):
    phone_number: str
    limit: int = 50

# Global variables for chatbot state
embedding_model = E5Embeddings()
vector_store = None
retriever = None
current_kb_path = None

app = FastAPI(
    version="1.0.0", 
    title="WhatsFlour Bot API", 
    description="An intelligent assistant bot powered by Flourish Digital Pvt Ltd"
)  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pathlib import Path

DEFAULT_KB_PATH = Path("Flour.pdf")  

@app.on_event("startup")
async def load_default_knowledge_base():
    global vector_store, retriever
    print(f"Loading default knowledge base from: {DEFAULT_KB_PATH}")
    
    # Test MongoDB connection
    if not mongodb.ping():
        print("MongoDB connection failed!")

    if DEFAULT_KB_PATH.exists():
        vector_store = load_and_process_documents(str(DEFAULT_KB_PATH), embedding_model)
        retriever = advanced_retrievers(vector_store, embedding_model)
        print("Knowledge base loaded successfully!")
    else:
        print(f"Knowledge base file not found: {DEFAULT_KB_PATH}")

@app.get("/")
async def root():
    return {
        "message": "WhatsFlour Bot API is running", 
        "status": "active",
        "knowledge_base_loaded": vector_store is not None,
        "current_kb_path": current_kb_path,
        "mongodb_connected": mongodb.ping()
    }

@app.get("/webhook")
async def verify_webhook(
    request: Request,
    hub_mode: str = None,
    hub_challenge: str = None,
    hub_verify_token: str = None
):
    """Meta webhook verification"""
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return int(hub_challenge)
    else:
        raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
    """Receive messages from WhatsApp"""
    try:
        body = await request.json()
        print(f"Received webhook: {body}")
        
        # Process WhatsApp webhook
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            message = value["messages"][0]
            if message["type"] == "text":
                user_phone = message["from"]
                user_message = message["text"]["body"]
                
                # Store user message in database
                store_chat_message(user_phone, "user", user_message)
                
                # Process message in background
                background_tasks.add_task(
                    process_user_message,
                    user_phone,
                    user_message
                )
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

def process_user_message(user_phone: str, user_message: str):
    """Process user message and send response"""
    try:
        if retriever is None:
            send_whatsapp_message(user_phone, "I'm still learning! Please set up a knowledge base first.")
            return
        
        # Get user session and chat history from MongoDB
        user_session = get_user_session(user_phone)
        chat_history = user_session.get("chat_history", []) if user_session else []
        
        # Get chatbot response
        response, updated_chat_history, detected_intent = chatbot(user_message, retriever, chat_history)
        
        # Store assistant response in database
        store_chat_message(user_phone, "assistant", response, intent=detected_intent)
        
        # Update session in MongoDB
        store_user_session(user_phone, updated_chat_history)
        
        # Send response via WhatsApp
        send_whatsapp_message(user_phone, response)
        
    except Exception as e:
        print(f"Error processing user message: {e}")
        send_whatsapp_message(user_phone, "I'm experiencing technical difficulties. Please try again later.")

@app.post("/send-message")
async def send_message_directly(message_request: MessageRequest):
    """Send a message directly to a WhatsApp number (for testing)"""
    try:
        success = send_whatsapp_message(message_request.phone_number, message_request.message)
        if success:
            # Store the sent message in database
            store_chat_message(message_request.phone_number, "system", message_request.message)
            return {"status": "success", "message": "Message sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_directly(message_request: MessageRequest):
    """Chat directly with the bot (for testing without WhatsApp)"""
    try:
        if retriever is None:
            raise HTTPException(status_code=400, detail="Knowledge base not set. Please set a knowledge base first.")
        
        response, updated_chat_history, detected_intent = chatbot(message_request.message, retriever)
        
        # Store both user message and bot response in database
        store_chat_message(message_request.phone_number, "user", message_request.message, intent=detected_intent)
        store_chat_message(message_request.phone_number, "assistant", response, intent=detected_intent)
        
        return {
            "status": "success",
            "user_message": message_request.message,
            "bot_response": response,
            "detected_intent": detected_intent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-sessions")
async def get_chat_sessions():
    """Get current chat sessions (for monitoring)"""
    try:
        sessions = list(mongodb.sessions.find({}, {"_id": 0, "phone_number": 1, "last_activity": 1, "message_count": 1}))
        return {
            "active_sessions": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{phone_number}")
async def get_chat_history_endpoint(phone_number: str, limit: int = 50):
    """Get chat history for a specific phone number"""
    try:
        chat_history = get_chat_history(phone_number, limit)
        return {
            "phone_number": phone_number,
            "chat_history": chat_history,
            "total_messages": len(chat_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-session/{phone_number}")
async def clear_session(phone_number: str):
    """Clear a specific user session"""
    try:
        # Delete session and chat history
        session_result = mongodb.sessions.delete_one({"phone_number": phone_number})
        chat_result = mongodb.chats.delete_many({"phone_number": phone_number})
        
        return {
            "status": "success", 
            "message": f"Session cleared for {phone_number}",
            "sessions_deleted": session_result.deleted_count,
            "chats_deleted": chat_result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base-status")
async def get_knowledge_base_status():
    """Get current knowledge base status"""
    try:
        # Get latest knowledge base info from MongoDB
        kb_info = mongodb.knowledge_bases.find_one(sort=[("loaded_at", -1)])
        
        return {
            "knowledge_base_loaded": vector_store is not None,
            "current_path": current_kb_path,
            "vector_store_ready": retriever is not None,
            "kb_info": kb_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear the current knowledge base"""
    global vector_store, retriever, current_kb_path
    
    try:
        if os.path.exists("./flourish_chroma_db"):
            shutil.rmtree("./flourish_chroma_db")
        
        vector_store = None
        retriever = None
        current_kb_path = None
        
        # Update knowledge base status in MongoDB
        mongodb.knowledge_bases.update_one(
            {"status": "active"},
            {"$set": {"status": "cleared", "cleared_at": datetime.now(timezone.utc)}},
            upsert=False
        )
        
        return {
            "status": "success",
            "message": "Knowledge base cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get overall statistics"""
    try:
        total_chats = mongodb.chats.count_documents({})
        total_sessions = mongodb.sessions.count_documents({})
        user_count = len(mongodb.chats.distinct("phone_number"))
        
        # Message type distribution
        user_messages = mongodb.chats.count_documents({"message_type": "user"})
        assistant_messages = mongodb.chats.count_documents({"message_type": "assistant"})
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        recent_chats = mongodb.chats.count_documents({"timestamp": {"$gte": yesterday}})
        
        return {
            "total_chats": total_chats,
            "total_sessions": total_sessions,
            "unique_users": user_count,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "recent_chats_24h": recent_chats,
            "knowledge_base_loaded": vector_store is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_loaded": vector_store is not None,
        "retriever_ready": retriever is not None,
        "mongodb_connected": mongodb.ping(),
        "current_kb_path": current_kb_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
