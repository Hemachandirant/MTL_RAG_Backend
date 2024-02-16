import streamlit as st   
import os  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader  
from dotenv import load_dotenv  
import base64  
from transformers import pipeline  
from optimum.intel.openvino import OVModelForCausalLM  
from transformers import AutoTokenizer  
  
load_dotenv(verbose=True)  
  
#file loader and preprocessing  
def file_preprocessing(file):  
    loader =  PyPDFLoader(file)  
    pages = loader.load_and_split()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)  
    texts = text_splitter.split_documents(pages)  
    final_texts = ""  
    for text in texts:  
        print(text)  
        final_texts = final_texts + text.page_content  
    return final_texts  
  
# LLM pipeline
@st.cache
def llm_pipeline(filepath):
    model_vendor = os.getenv('MODEL_VENDOR')
    model_name = os.getenv('MODEL_NAME')
    model_precision = os.getenv('MODEL_PRECISION')
    inference_device = os.getenv('INFERENCE_DEVICE')
    cache_dir = os.getenv('CACHE_DIR')
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": cache_dir}

    model_id = f'{model_vendor}/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    ov_model_path = f'./{model_name}/{model_precision}'
    model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config,
                                               cache_dir=cache_dir)

    pipe_sum = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)

    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    print("-----------------------------------------------------------------------")
    print(result)
    return result

#streamlit code   
st.set_page_config(layout="wide")  
  
def main():  
    st.title("Document Summarization App using Langauge Model")  
  
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])  
  
    if uploaded_file is not None:  
        if st.button("Summarize"):  
            col1, col2 = st.columns(2)  
            filepath = "data/"+uploaded_file.name  
            with open(filepath, "wb") as temp_file:  
                temp_file.write(uploaded_file.getbuffer())  # Updated from read() to getbuffer()  
            
            # Define base64_pdf before using it
            with open(filepath, "rb") as f:  
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            with col1:  
                st.info("Uploaded File")  
                pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'  
                st.markdown(pdf_display, unsafe_allow_html=True)  
  
            with col2:  
                st.info("Summarization in progress...")
                summary = llm_pipeline(filepath)
                print("---------------------------------------------")
                print(summary)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
