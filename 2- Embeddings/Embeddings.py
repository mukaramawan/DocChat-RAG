from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = HuggingFaceEmbeddings(model="hreyulog/embedinggemma_arkts")

text = "This is a sample text to be embedded."

embedding = model.embed_documents(text)

print(embedding)

