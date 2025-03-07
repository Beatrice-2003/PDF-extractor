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
import requests
from bs4 import BeautifulSoup



def fetch_website_content(url):
    """Fetches and extracts text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the website
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        st.error(f"Error fetching the website content: {e}")
        return None

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

    st.set_page_config(page_title="Chat with multiple PDFs and Websites", page_icon=":books:")
    st.header("Chat with your documents and websites :books:")

    user_query = st.text_input("Ask questions about your document or website.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
        website_url = st.text_input("Or enter a website URL here:")
    
    

    if st.button("Search"):
        with st.spinner("Thinking.."):
            vectorstore = None  # Initialize vectorstore

            # Process PDF documents
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

            # Process website URL
            if website_url:
                website_content = fetch_website_content(website_url)
                if website_content:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=200,
                        length_function=len
                    )

                    chunks = text_splitter.split_text(text=website_content)

                    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002", deployment="ada-embedding-ca")
                    website_vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

                    # # Combine vectorstores if both PDFs and website content are processed
                    # if vectorstore is not None:
                    #     vectorstore = FAISS.concatenate(vectorstore, website_vectorstore)
                    # else:
                    #     vectorstore = website_vectorstore

            # User input processing and response display outside the sidebar
            if user_query and vectorstore is not None:
                docs = vectorstore.similarity_search(query=user_query)
                website_docs = website_vectorstore.similarity_search(query=user_query)
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
                response = chain.run(input_documents=website_docs, question=user_query)
                st.write(response)

if __name__ == "__main__":
    main()




