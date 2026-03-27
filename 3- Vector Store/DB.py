from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

Documents =[Document(page_content="Pakistan and India share a border, and have a long history of interactions.", metadata={"source": "dummytext.txt"}),
            Document(page_content="Natural Language processing is part of Deep learning.", metadata={"source": "aibook.txt"}),
            Document(page_content="Pakistan is a country in South Asia.", metadata={"source": "dummytext.txt"})]

embedding_model = HuggingFaceEmbeddings(model="hreyulog/embedinggemma_arkts")

text = "Where is Pakistan located?"

VectorStore = Chroma.from_documents(documents=Documents, embedding=embedding_model, persist_directory="./chroma_db")

result = VectorStore.similarity_search(query=text, k=2)

for i in range(len(result)):
    print(result)
