### Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw PDFs in **Redis**, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying an **LLM (Gemini model)**.

**Live Site:** 🌐 [smart-doc-assistant.streamlit.app](https://smart-doc-assistant.streamlit.app/)



### Architectural Diagram
![Architectural Diagram](https://github.com/user-attachments/assets/12f49f3e-6aab-42ce-8beb-f910de0075ea)


### Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.  
- **Redis for Raw Storage**: Stores and retrieves raw PDFs efficiently, to implement persistent storage.  
- **PGVector for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.  
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.  
- **LLM Integration**: Uses a **Gemini model** to generate responses based on retrieved context.  

### Tech Stack

#### Programming Language
- Python  

#### Libraries
- `unstructured`
- `pgvector`
- `redis`
- `langchain`
- `gemini-flash`

#### Databases
- **Redis**: For raw PDF storage  
- **PostgreSQL + PGVector**: For embeddings storage  

#### LLM
- **Gemini-Flash** (via Gemini API Key)
