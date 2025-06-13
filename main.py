from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from document_loader import *   
from dotenv import load_dotenv

load_dotenv()  

chat = ChatOpenAI() 
embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=200, 
    chunk_overlap=0
)


try: 

    loader = DocumentLoader(
        file_path="./project_1_publications.json", 
        jq_schema='.[] | {title: .title, body: .publication_description}',
        text_content=False
        )

    docs = loader.load_and_split(
        text_splitter=text_splitter
    )


    db = Chroma.from_documents(
        docs, 
        embedding=embeddings, 
        persist_directory="doc_emb"
    )

    # wrapping into a retriever & QA chain
    retriever = db.as_retriever(search_kwargs={"k": 2})

    qa = RetrievalQA.from_chain_type(
        llm=chat, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=False
    )

    query = "What is ai agent?"
    answer = qa.run(query)

    print("---\nAnswer:\n", answer)
    

except DocumentLoaderException as e: 
    print(f"Error: {e}")