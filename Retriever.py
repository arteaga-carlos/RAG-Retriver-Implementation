from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Sample documents
docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="LangChain helps with LLMs."),
    Document(page_content="Retrievers fetch documents."),
]

# Create embeddings
embeddings = OpenAIEmbeddings()  # Make sure OPENAI_API_KEY is set

# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Turn the vectorstore into a retriever
retriever = vectorstore.as_retriever()

# Use the retriever
query = "What tool helps with large language models?"
results = retriever.get_relevant_documents(query)

for doc in results:
    print(doc.page_content)
