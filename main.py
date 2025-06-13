# from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma 
from langchain_community.vectorstores import Chroma

from document_loader import *   # DocumentLoader, DocumentLoaderException
from dotenv import load_dotenv

load_dotenv()  

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

    results = db.similarity_search(
        "what is ai agent?",
        k=1
    )

    for result in results: 
        print("---\n")
        print(result.page_content)

except DocumentLoaderException as e: 
    print(f"Error: {e}")