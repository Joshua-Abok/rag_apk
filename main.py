from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains.summarize import load_summarize_chain

from document_loader import *   
from dotenv import load_dotenv

load_dotenv()  

chat = ChatOpenAI() 
embeddings = OpenAIEmbeddings()

summarizer = load_summarize_chain(chat, chain_type="map_reduce")

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=100, 
    chunk_overlap=5
)


# helper to return raw text of top-k docs 
def retrieve_for_agent(query: str) -> str: 
    docs = retriever.get_relevant_documents(query)
    summary = summarizer.run(docs)
    # return "\n\n".join(d.page_content for d in docs)
    return summary[:2000]
    

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

    # # wrapping into a retriever & QA chain
    retriever = db.as_retriever(search_kwargs={"k": 1})

    tool = Tool(
        name="Retriever", 
        func=retrieve_for_agent, 
        description=(
        "Returns a short summary (≤2000 chars) of the most relevant "
        "publications to answer the question."
    ),
    )

    agent = initialize_agent(
        tools=[tool], 
        llm=chat, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, # displays actions/observation
        handle_parsing_errors=True
    )


    query = "What is VAE?"
    answer = agent.run(query)

    print("---\nAnswer:\n", answer)
    

except DocumentLoaderException as e: 
    print(f"Error: {e}")