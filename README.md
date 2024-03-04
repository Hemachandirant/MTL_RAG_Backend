# MTL RAG Backend

This program doesn't rely on any cloud services or webAPIs for inferencing. The program downloads all the data, including reference documents and DL models, and **can perform inference offline**. You don't need any cloud services once you prepare the data locally. 

## How to run

0. Install Python prerequisites

Install steps for Windows.
```sh
python -m venv venv
venv/Scripts/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt

# Install en_core_web_sm, a Spacy pipeline for English
python -m spacy download en_core_web_sm
```


1 . Generate vector store from the legal documents
- Run '`openvino-doc-specific-extractor.py`'.
- The program will store the document object in a pickle file (`doc_obj.pickle`) and use it if it exists the next time.
```sh
python openvino-doc-specific-extractor.py
```
- `.vectorstore_300_0` directory will be created.
	- '_300_0' means the chunk size is 300 and chunk overlap is 0.
	- You can generate the vector store with different chunk configurations by modifying the last few lines of Python code.
	- You can modify the `.env` file to specify which vector store file to use in the client and server programs. 


4. Run the demo
- Run the server
- Note: The '`--host 0.0.0.0`' option is to accept external connection. '`--port xx`' option is also available.
```sh
uvicorn openvino-rag-server:app --host 0.0.0.0 --port 8080
uvicorn summary_server:app --host 0.0.0.0 --port 4000 --reload
```



## Tested environment
- OS: Windows 11
