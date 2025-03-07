import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def fetch_website_content(url):
    """Fetches and extracts text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        st.error(f"Error fetching the website content: {e}")
        return None

def process_pdf(pdf):
    """Processes a PDF file and returns the vectorstore."""
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None case

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002", deployment="ada-embedding-ca")
    return FAISS.from_texts(chunks, embedding=embeddings)

def process_website(url):
    """Processes a website URL and returns the vectorstore."""
    website_content = fetch_website_content(url)
    if website_content:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=website_content)
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002", deployment="ada-embedding-ca")
        return FAISS.from_texts(chunks, embedding=embeddings)
    return None

def init():
    load_dotenv()
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        st.error("AZURE_OPENAI_API_KEY is not set")
        st.stop()
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.error("AZURE_OPENAI_ENDPOINT is not set")
        st.stop()

def main():
    init()

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

    st.set_page_config(page_title="Chat with PDFs and Websites", page_icon="🌐")
    st.header("Chat with your documents and websites 🌐")

    user_query = st.text_input("Ask questions about your documents or website.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
        website_url = st.text_input("Or enter a website URL here:")

    if st.button("Search document"):
        with st.spinner("Thinking.."):
            if pdf_docs:
                vectorstore = None
                for pdf in pdf_docs:
                    vectorstore = process_pdf(pdf)

                if vectorstore and user_query:
                    st.session_state.messages.append(HumanMessage(content=user_query))

                    # Perform similarity search
                    docs = vectorstore.similarity_search(query=user_query)
                    if docs:
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

                        # Append AI response to messages
                        st.session_state.messages.append(AIMessage(content=response))
                    else:
                        st.warning("No relevant documents found for your query.")
            else:
                st.warning("Please upload a PDF document.")

    if st.button("Search Website"):
        with st.spinner("Thinking.."):
            if website_url:
                website_vectorstore = process_website(website_url)

                if website_vectorstore and user_query:
                    st.session_state.messages.append(HumanMessage(content=user_query))

                    # Perform similarity search
                    docs = website_vectorstore.similarity_search(query=user_query)
                    if docs:
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

                        # Append AI response to messages
                        st.session_state.messages.append(AIMessage(content=response))
                    else:
                        st.warning("No relevant content found for your query.")
            else:
                st.warning("Please enter a website URL.")

    # Display conversation history
    # for msg in st.session_state.messages:
    #     if isinstance(msg, HumanMessage):
    #         st.write(f"**You:** {msg.content}")
    #     elif isinstance(msg, AIMessage):
    #         st.write(f"**Assistant:** {msg.content}")
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == "__main__":
    main()
