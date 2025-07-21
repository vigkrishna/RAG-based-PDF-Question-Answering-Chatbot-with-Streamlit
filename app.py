import streamlit as st
import asyncio

# Patch for asyncio event loop in Streamlit thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import os
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

# LangChain & Google Generative AI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --------- Utility Functions ----------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def build_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def user_input(question, api_key, pdf_docs, history):
    if not api_key or not pdf_docs:
        st.warning("Please provide a Google API key and upload PDFs.")
        return

    text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(text)
    build_vector_store(chunks, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    rag_chain = get_rag_chain(api_key, retriever)
    result = rag_chain({"query": question})

    answer = result["result"]
    sources = result["source_documents"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf_names = ", ".join([pdf.name for pdf in pdf_docs])

    history.append((question, answer, "Google AI", timestamp, pdf_names))

    st.markdown(display_chat(question, answer), unsafe_allow_html=True)
    for doc in sources:
        st.markdown(f"<div style='color:gray'><b>Source:</b> {doc.page_content[:300]}...</div>", unsafe_allow_html=True)

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def display_chat(user_msg, bot_msg):
    return f"""
    <style>
        .chat-message {{ padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; }}
        .chat-message.user {{ background-color: #2b313e; }}
        .chat-message.bot {{ background-color: #475063; }}
        .avatar img {{ max-width: 78px; max-height: 78px; border-radius: 50%; }}
        .message {{ color: white; padding: 0 1.5rem; width: 80%; }}
    </style>
    <div class="chat-message user">
        <div class="avatar"><img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"></div>
        <div class="message">{user_msg}</div>
    </div>
    <div class="chat-message bot">
        <div class="avatar"><img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp"></div>
        <div class="message">{bot_msg}</div>
    </div>
    """

# --------- Streamlit App UI ----------

def main():
    st.set_page_config(page_title="Chat with PDFs using RAG", page_icon=":books:")
    st.header("Chat with PDFs using RAG :books:")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    with st.sidebar:
        st.title("RAG Chat Settings")
        api_key = st.text_input("Enter your Google API Key:", type="password")
        st.markdown("Get your API key from [Google AI Studio](https://ai.google.dev)")

        col1, col2 = st.columns(2)
        if col1.button("Reset"):
            st.session_state.conversation_history = []
        if col2.button("Rerun"):
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    build_vector_store(chunks, api_key)
                    st.success("Processing complete.")
            else:
                st.warning("Please upload PDF files.")

    # User input
    question = st.text_input("Ask a question from the PDFs:")
    if question:
        user_input(question, api_key, pdf_docs, st.session_state.conversation_history)

if __name__ == "__main__":
    main()
