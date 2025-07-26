import streamlit as st
import asyncio
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
from langchain_core.prompts import ChatPromptTemplate

# --------- Utility Functions ----------

# Step 1: Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Step 2: Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

#Step 3: Build the vector store using FAISS(Facebook AI Similarity Search), basically converting text chunks to vector embeddings
def build_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

#Step 4: Defining the Retrieval-Augmented Generation (RAG) chain
def get_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

#Step 5: Rephrase the question using Gemini Flash
# This function uses a rule-based approach to handle common types of queries and falls back to Gemini
def rephrase_with_gemini(original_question, api_key):
    lowered = original_question.lower().strip()

    if "summarize" in lowered or "summarise" in lowered:
        return "Here is the summary for the PDF you provided."
    elif "extract keywords" in lowered:
        return "Here are the extracted keywords from your document."
    elif "generate questions" in lowered:
        return "Here are some questions generated based on your PDF."
    elif "topics" in lowered:
        return "Here are the main topics covered in your PDF."
    elif "table of contents" in lowered or "outline" in lowered:
        return "Here is the outline of your uploaded PDF."

    prompt = ChatPromptTemplate.from_template(
        "just only give initial starting of answer to user query, for example if user wants to summarise anything , your response should be here is your summary of your provided pdf like that {question}"
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    chain = prompt | model

    try:
        response = chain.invoke({"question": original_question})
        cleaned = response.content.strip().removeprefix("Rephrased:").strip()
        return cleaned
    except Exception:
        return f"You asked: {original_question}"


#Step 6: Handle user input and display chat
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
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf_names = ", ".join([pdf.name for pdf in pdf_docs])

    rephrased_question = rephrase_with_gemini(question, api_key)

    history.append((question, answer, "Google AI", timestamp, pdf_names))

    st.markdown(display_chat(rephrased_question, answer,timestamp), unsafe_allow_html=True)

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

def display_chat(user_msg, bot_msg, timestamp):
    return f"""
    <div style="margin-bottom: 1.5rem;">
        <div>
            <strong style="font-size: 1.05rem;">AI Generated Response</strong>
            <span style="font-size: 0.85rem; color: gray; margin-left: 10px;">{timestamp}</span><br>
            <div style="font-size: 1rem; line-height: 1.6; margin-top: 6px;">
                {user_msg}<br>{bot_msg}
            </div>
        </div>
    </div>
    """





#Step 7: Streamlit UI
def main():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
    <style>
        body, .stMarkdown, .stTextInput, .stButton {
            font-family: 'Roboto', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)


    st.set_page_config(page_title="Ask, Learn, Discover – Directly from Your PDFs!", page_icon="📚")

    st.header("Ask, Learn, Discover – Directly from Your PDFs! :books:")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    with st.sidebar:
        st.title("Smart Document Assistant")
        api_key = st.text_input("Enter your Google API Key",type="password")
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
                
        st.sidebar.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
<div style='text-align: center; font-size: 0.9rem;'>
    Made with ❤️ by <strong>Krishna Vig</strong><br>
    <a href="https://github.com/vigkrishna" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20" style="margin-right: 5px; vertical-align: middle;">GitHub
    </a>
    &nbsp;•&nbsp;
    <a href="https://www.linkedin.com/in/vigkrishna/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" style="margin-right: 5px; vertical-align: middle;">LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)
        

    question = st.text_input(label="", placeholder="Ask anything", key="user_question")

    if question:
        user_input(question, api_key, pdf_docs, st.session_state.conversation_history)

if __name__ == "__main__":
    main()
    
    
    
    
#Project done and dusted, 
#Thank you for using this project,
#If you have any questions or suggestions, feel free to reach out on GitHub or LinkedIn
#Happy coding! 😊    
