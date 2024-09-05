import os
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
import faiss
from dotenv import load_dotenv

load_dotenv()

st.title("News Research")

st.sidebar.title("News Article URLs")
urls=[]


for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked=st.sidebar.button("Process URLS")

main_placeholder=st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1000)
    main_placeholder.text("Text Splitting started")
    docs=text_splitter.split_documents(data)

    vector_store=FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding vector started building")
    time.sleep(2)
    vector_store.save_local("faiss_index")
    main_placeholder.text("FAISS index and metadata have been saved successfully.")
    time.sleep(2)

query=main_placeholder.text_input("Question: ")

if query:
    if os.path.exists("faiss_index"):

        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
        result=chain({"question":query},return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
        sources=result.get("sources","")
        if sources:
            st.subheader("Sources")
            sources_list=sources.split("\n")
            for source in sources_list:
                st.write(source)
