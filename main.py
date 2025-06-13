import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool, AgentType
from document_loader import DocumentLoader, DocumentLoaderException
from dotenv import load_dotenv

load_dotenv()


chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
summarizer = load_summarize_chain(chat, chain_type="map_reduce")


st.set_page_config(page_title="RAG Chat", page_icon="🧠")
st.title("📚 Ready Tensor RAG Chat")
st.caption("Ask questions about the Ready Tensor publications.")

# Setup session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    try:
        # Load and split docs
        loader = DocumentLoader(
            file_path="./project_1_publications.json",
            jq_schema='.[] | {title: .title, body: .publication_description}',
            text_content=False
        )
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=100,
            chunk_overlap=5
        )
        docs = loader.load_and_split(text_splitter=text_splitter)

        # Embed and store in vector DB
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="doc_emb")
        retriever = db.as_retriever(search_kwargs={"k": 1})

        # Tool for summarization
        def retrieve_for_agent(query: str) -> str:
            docs = retriever.get_relevant_documents(query)
            summary = summarizer.run(docs)
            return summary[:2000]

        tool = Tool(
            name="Retriever",
            func=retrieve_for_agent,
            description="Returns a short summary (≤2000 chars) of the most relevant publications to answer the question."
        )

        # Create agent
        st.session_state.agent = initialize_agent(
            tools=[tool],
            llm=chat,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    except DocumentLoaderException as e:
        st.error(f"Loader error: {e}")
    except Exception as e:
        st.error(f"Startup failed: {e}")

if st.session_state.chat_history:
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.markdown(msg)


query = st.chat_input("Ask me anything about the Ready Tensor publications...")


if query and "agent" in st.session_state:
    with st.chat_message("user"):
        st.markdown(query)
        st.session_state.chat_history.append(("You", query))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.agent.run(query)
                st.markdown(answer)
                st.session_state.chat_history.append(("AI", answer))
            except Exception as e:
                st.error(f"Agent error: {e}")
