from PyPDF2 import PdfReader  
  
def extract_text_from_pdf(pdf_path):  
    pdf_reader = PdfReader(pdf_path)  
    text = ""  
    for page in pdf_reader.pages:  
        text += page.extract_text()  
    return text  
  
#pdf_text = extract_text_from_pdf(r'C:\Users\hemac\Desktop\RAG\GGUF_RAG\Intel_RAG_Openvino\openvino-llm-chatbot-rag\caseDoc.pdf') 
#pdf_text = extract_text_from_pdf('openvino-llm-chatbot-rag/caseDoc.pdf')  
pdf_text = extract_text_from_pdf('caseDoc.pdf')  

 
print(pdf_text)  



