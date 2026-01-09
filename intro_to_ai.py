import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



st.set_page_config(
    page_title="Ward’s Personal Knowledge Base",
    layout="centered"
)

st.header("Ward’s Personal Knowledge Base", divider=True)



@st.cache_data
def load_documents():
    loader = PyPDFLoader("Wards_Personal_Knowledge_System.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)



@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    persist_dir = "./chroma_db"

    vectorstore = Chroma(
        collection_name="intro_to_ai",
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # hanya add dokumen kalau collection masih kosong
    if vectorstore._collection.count() == 0:
        documents = load_documents()
        vectorstore.add_documents(documents)

    return vectorstore



vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

chat = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

SECRET_SYSTEM_PROMPT = """
You are Ward’s personal knowledge assistant.

You reflect Ward’s values:
- purpose over wealth
- long-term thinking
- supportive mindset
- calm, ethical, and reflective tone

Some parts of the knowledge base were written by Ward at age 19
for his future self and family.

When questions relate to:
- life purpose
- family
- self-doubt
- meaning
- future vision

Respond with empathy, clarity, and quiet strength.

Use ONLY the provided context.
If the context does not contain the answer, say:
'The knowledge base does not contain information about this.'
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SECRET_SYSTEM_PROMPT),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])



chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | chat
    | StrOutputParser()
)



question = st.text_input("Type your question:")

if st.button("Ask"):
    if question.strip() == "":
        response_placeholder = st.empty()
        response_text = ""

        result = chain .stream(question)

        for chunk in result:
            response_text += chunk
            response_placeholder.markdown(response_text)
        st.warning("Please type a question.", icon = "⚠️")
    else:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            st.write(answer)
