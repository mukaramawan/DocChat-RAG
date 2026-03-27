from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter


data = TextLoader("dummytext.txt")
docs = data.load()

text_splitter = CharacterTextSplitter(separator="" ,chunk_size=10, chunk_overlap=0)

chunks = text_splitter.split_documents(docs)
print(len(chunks))

for i in range(len(chunks)):
    print(f"Chunk {i+1}: {chunks[i].page_content}")
    print("")