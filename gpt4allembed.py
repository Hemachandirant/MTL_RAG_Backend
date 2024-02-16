from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma, FAISS  
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_community.embeddings import GPT4AllEmbeddings  
import os
import time
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader 
from fastapi import FastAPI, UploadFile,File
import logging
from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader


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

  
@app.get("/search")  
def search(query: str):  
    # Load the document and split it into smaller chunks  
    loader = PyPDFLoader("89.pdf")  
    data = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)  
    all_splits = text_splitter.split_documents(data)  
  
    # Initialize the embeddings models  
    embeddings = HuggingFaceEmbeddings()  
    gpt4all_embd = GPT4AllEmbeddings()  
  
    # Create a FAISS index to perform similarity search  
    db = FAISS.from_documents(all_splits, gpt4all_embd)  
  
    # Perform a similarity search for the query string  
    docs = db.similarity_search(query)  
    page_contents = [document.page_content for document in docs]  
  
    # Return the top matching document page content  
    if len(page_contents) > 0:  
        return page_contents[0]  
    else:  
        return "No matching documents found"