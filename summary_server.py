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


# Define a function to chunk the text  
def chunk_text(text, chunk_size):  
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/summary")  # Change this to a POST method  
async def summary(file: UploadFile = File(...)):  
    with tempfile.NamedTemporaryFile(delete=False) as tmp:  
        contents = await file.read()  
        tmp.write(contents)  
        tmp_path = tmp.name  
  
    loader = PyPDFLoader(tmp_path)   
    docs = loader.load()  
    docs = str(docs)  
      
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")    
  
    chunk_size = 5000  
    chunks = chunk_text(docs, chunk_size)    
  
    summaries = []    
    for chunk in chunks:    
        summary = summarizer(chunk, max_length=250, min_length=50, do_sample=False)    
        summaries.append(summary[0]['summary_text'])    
  
    final_summary = " ".join(summaries)  
    logging.debug(final_summary)  
    return final_summary  


# python openvino-doc-specific-extractor.py
# uvicorn openvino-rag-server:app --host 0.0.0.0
# uvicorn summary_server:app --host 0.0.0.0 --port 5000 --reload  