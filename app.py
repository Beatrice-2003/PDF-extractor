import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def main():
    load_dotenv()

    if os.getenv("AZURE_OPENAI_API_KEY") is None or os.getenv("AZURE_OPENAI_API_KEY") == "":
        print("AZURE_OPENAI_API_KEY is not set")
        exit(1)
    else: 
        print("AZURE_OPENAI_API_KEY is set")

    if os.getenv("AZURE_OPENAI_ENDPOINT") is None or os.getenv("AZURE_OPENAI_ENDPOINT") == "":
        print("AZURE_OPENAI_ENDPOINT is not set")
        exit(1)
    else: 
        print("AZURE_OPENAI_ENDPOINT is set")

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with your documents :books:")

    user_query = st.text_input("Ask questions about your document.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
    
    if st.button("Search"):
            with st.spinner("Thinking.."):
                if pdf_docs is not None:
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=200,
                            length_function=len
                        )

                        chunks = text_splitter.split_text(text=text)

                        embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002", deployment="ada-embedding-ca")
                        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

                         # User input processing and response display outside the sidebar
                        if user_query and pdf_docs is not None:
                            docs = vectorstore.similarity_search(query=user_query)
                            llm = AzureChatOpenAI(
                                azure_deployment="gpt4o",
                                api_version="2024-08-01-preview",
                                temperature=0,
                                max_tokens=None,
                                timeout=None,
                                max_retries=2
                            )
                            chain = load_qa_chain(llm=llm, chain_type="stuff")
                            response = chain.run(input_documents=docs, question=user_query)
                            st.write(response)
                        

    

   

if __name__ == "__main__":
    main()
