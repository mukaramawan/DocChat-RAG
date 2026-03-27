import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")
st.write("Ask any question and get answers based on the available documents")

# Initialize embedding model and vector store
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model="hreyulog/embedinggemma_arkts")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 0.5},
    )
    return retriever

@st.cache_resource
def load_llm():
    llm = ChatGroq(model="groq/compound-mini", temperature=0.7)
    return llm

# Load resources
retriever = load_retriever()
llm = load_llm()

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that provides information based on the provided context. If the context does not contain the answer, say I don't know."""),
    ("human", """Context:\n\n{context}\n\n Question: {question}""")])

# User input
question = st.text_input("Enter your question:")

if question:
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate response
    response = llm.invoke(prompt.format(context=context, question=question))
    
    # Display response
    st.subheader("Answer")
    st.write(response.content)
    
    # Display retrieved documents
    with st.expander("View Retrieved Documents"):
        for i, doc in enumerate(docs, 1):
            st.write(f"**Document {i}:**")
            st.write(doc.page_content)
            st.divider()
