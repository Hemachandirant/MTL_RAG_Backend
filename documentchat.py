from fastapi import FastAPI, UploadFile, File  
from fastapi.middleware.cors import CORSMiddleware  
from dotenv import load_dotenv  
import openai  
from PyPDF2 import PdfReader  
import tempfile  

  
# Load environment variables  
load_dotenv()  
  
app = FastAPI()  
  
origins = [  
    "http://localhost:59507",  # Angular app  
    "http://localhost:8000",   # FastAPI server  
    "http://localhost",  
    "http://localhost:8080",  
    "http://localhost:4200",  
]  
  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  
  
# Initialize OpenAI API key  
openai.api_type = "azure"  
openai.api_base = "https://dwspoc.openai.azure.com/"  
openai.api_version = "2023-07-01-preview"  
openai.api_key = "bd38ee31e244408cacab3e1dd4c32221"  
  
# Function to chat with the document using OpenAI  
@app.post("/chatwithdocument")    
async def chat_with_document(user_message: str):    
    conversation = []  # Define the conversation list here      
    document_text = 'IPC is indian Penal code'
    conversation.append({"role": "system", "content": "You are AI Legal advisor of India, if any question about legal just provide the answer in clear and precise way and dont say that you dont know just make up the things for the question: " + user_message})    
    conversation.append({"role": "assistant", "content": "Document:\n" + document_text})    
    response = openai.ChatCompletion.create(    
        engine="GPT4",    
        messages=conversation,    
        temperature=0.7,    
        max_tokens=800,    
        top_p=0.95,    
        frequency_penalty=0,    
        presence_penalty=0,    
        stop=None    
    )    
    ai_response = response.choices[0].message["content"]    
    
    return {"ai_response": ai_response}    

