import os  
import time  
from dotenv import load_dotenv  
from fastapi.middleware.cors import CORSMiddleware  
from PyPDF2 import PdfReader   
from fastapi import FastAPI, UploadFile,File  
import logging  
from transformers import AutoTokenizer, pipeline  
from langchain_community.document_loaders import PyPDFLoader  
import tempfile  
import logging  
  
import os  
import openai  
import time  
openai.api_type = "azure"  
openai.api_base = "https://dwspoc.openai.azure.com/"  
openai.api_version = "2023-07-01-preview"  
openai.api_key = "bd38ee31e244408cacab3e1dd4c32221"  
  
app = FastAPI()  
  
def translate_chunk(chunk):  
    message_text = [{"role":"system","content":"You are an AI assistant that translates english to the specified language without any loss in context."},{"role":"user","content": chunk+"\n\ntranslate it to french"}]  
    start=time.time()  
    completion = openai.ChatCompletion.create(  
        engine="GPT4-32K",  
        messages = message_text,  
        temperature=0.5,  
        max_tokens=4000,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None  
    )  
    end=time.time()  
    print(end-start)  
    ans = completion.choices[0].message.content  
    return ans  
  
@app.post("/translate")  
async def summary(file: UploadFile = File(...)):    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:    
        contents = await file.read()    
        tmp.write(contents)    
        tmp_path = tmp.name    
    
    loader = PyPDFLoader(tmp_path)     
    docs = loader.load()    
    docs1 = str(docs)   
    logging.debug(docs1)  
  
    # Split document into chunks of 4000 characters  
    chunks = [docs1[i:i + 4000] for i in range(0, len(docs1), 4000)]  
  
    # Translate each chunk and concatenate the results  
    translated_text = ''.join(translate_chunk(chunk) for chunk in chunks)  
  
    return translated_text  
