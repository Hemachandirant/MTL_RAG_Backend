from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import FAISS

from langchain.vectorstores import Chroma
import os

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.embeddings import HuggingFaceBgeEmbeddings
import shutil


app = FastAPI()


# class CaseId(BaseModel):
#     caseId: str


# class Query(BaseModel):
#     caseId: str
#     query: str


@app.post("/apiCaseId")
async def case_processing(request: Request,caseId:str=None):
    # data = await request.json()
    # caseId = data.get("caseId")

    filename = caseId + ".pdf"
    # folder_name = r"C:\Users\gopik\Downloads\caseDocs-main(1)\caseDocs-main\caseDocs"
    folder_name = r"F:\Documents backup\AI Projects\MTL\Basic_RAG\legalDocs"
    filepath = os.path.join(folder_name, filename)
    loader = PyPDFLoader(filepath)
    data = loader.load()

    # text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    # docs = text_splitter.split_documents(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(data)

    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # embeddings = HuggingFaceEmbeddings()
    vector_store_directory = f"stores/{caseId}"

    if not os.path.exists(vector_store_directory):
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=f"stores/{caseId}",
        )

    # # retriever part separately
    # load_vector_store = Chroma(
    #     persist_directory=f"stores/{caseId}", embedding_function=embeddings
    # )
    # retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

    # query = "urge of Sri Ganapathy lyer"
    # semantic_search = retriever.get_relevant_documents(query)
    # print(semantic_search)

    # db = Chroma.from_documents(docs, embeddings)
    # ques = "give information about second appeal 4th April, 1928"
    # docs = db.similarity_search(ques)
    # print(docs[0])

    return JSONResponse(content={"result": "Vector Datastore Created Successfully"})


@app.post("/apiQuery")
async def query_processing(request: Request,query:str=None,caseId:str=None):
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    data = await request.json()
    # caseId = data.get("caseId")
    # query = data.get("query")

    load_vector_store = Chroma(
        persist_directory=f"stores/{caseId}", embedding_function=embeddings
    )

    retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})
    semantic_search = retriever.get_relevant_documents(query)
    print(semantic_search)

    return JSONResponse(content={"result": semantic_search[0].page_content})


# @app.delete("/apiDeleteVectorStore")
# async def delete_vector_store(caseId: CaseId):
#     caseId = request.args.get("caseId")
#     vector_store_directory = f"stores/{caseId}"
#     shutil.rmtree(vector_store_directory)
#     return JSONResponse(content={"result": "Vector store deleted successfully"})
