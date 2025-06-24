from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel


def create_agent(
    docs,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    persist_directory: str = "doc_emb",
    top_k: int = 1,
) -> tuple:
 
    summarizer = load_summarize_chain(llm, chain_type="map_reduce")

    
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    retriever: VectorStoreRetriever = db.as_retriever(search_kwargs={"k": top_k})

    def retrieve_for_agent(query: str) -> str:
        results = retriever.get_relevant_documents(query)
        summary = summarizer.run(results)
        return summary[:2000]

    tool = Tool(
        name="Retriever",
        func=retrieve_for_agent,
        description="Returns a short summary (≤2000 chars) of the most relevant publications to answer the question."
    )

    # Create and return the agent
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, retriever
