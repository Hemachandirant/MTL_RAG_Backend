import requests
from flask import Flask, request, jsonify

# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain

# from langchain.llms import HuggingFaceHub
import os

import boto3
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFaceHub


os.environ["AWS_ACCESS_KEY_ID"] = "AKIAXHRYJD7GZDMNXQFK"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"


app = Flask(__name__)

@app.route("/apiCaseId", methods=["GET"])
def documentDownload():
    caseId = request.args.get("caseId")
    s3 = boto3.client("s3")
    bucket_name = "wilegaldocs"
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    # print(respnse)
    # Iterate through the objects and filter based on case number
    for obj in response["Contents"]:
        # Extract the case number from the object key
        key_parts = obj["Key"].split("_")
        if len(key_parts) >= 2 and key_parts[1] == caseId:
            # Download the object to your server
            s3.download_file(bucket_name, obj["Key"], "caseDoc.pdf")
            current_directory = os.getcwd()
            filename = "caseDoc.pdf"
            file_path = os.path.join(current_directory, filename)
            jsonData = {"path": file_path}
            return jsonify(jsonData)
        else:
            return jsonify({"msg": "No data found"})
        
app.run(debug=True, host='0.0.0.0')





@app.route("/apiQues", methods=["GET"])
def local_llm():
    question = request.args.get("question")
    current_directory = os.getcwd()
    filename = "caseDoc.pdf"
    file_path = os.path.join(current_directory, filename)
    loader = PyPDFLoader(file_path)
    data = loader.load()
    # documents = data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(
        documents=all_splits, embedding=GPT4AllEmbeddings()
    )
    docs = vectorstore.similarity_search(question)
    llmPrompt = "For the question "+ question + " use the following information and provide the answer "+ docs[0]+docs[1]
    url = "http://127.0.0.1.8000/chatbot/1?query="+llmPrompt
    response = requests.get(url)
    if response.status_code == 200:
        # Request was successful
        data = response.json()
        return jsonify(data)
    else:
        # Request failed
        return jsonify({'error': 'Request failed with status code: ' + str(response.status_code)})
 