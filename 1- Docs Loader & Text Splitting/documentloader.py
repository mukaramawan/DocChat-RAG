from langchain_community.document_loaders import TextLoader

docs = TextLoader("data.txt")
data = docs.load()
print(data[0].content)