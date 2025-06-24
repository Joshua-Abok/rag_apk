import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# THIS HAS BEEN ENCLOSED
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, OUTPUTS_DIR, LOADER_DIR, AGENTS_DIR


from loaders.document_loader import DocumentLoader, DocumentLoaderException
from agents.agent import create_agent

load_dotenv()


chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()


#st.set_page_config(page_title="RAG Chat", page_icon="🧠")
st.set_page_config(page_title="Ready Tensor Publication Explorer", page_icon="🧠")
#st.title("📚 Ready Tensor RAG Chat")
st.title("📚 RAG-Based Ready Tensor Publication Explorer")
st.caption("Ask questions about the Ready Tensor publications.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    try:
        # Load and split docs
        loader = DocumentLoader(
            #file_path="./data/project_1_publications.json",
            file_path=f"{DATA_DIR}/project_1_publications.json",
            jq_schema='.[] | {title: .title, body: .publication_description}',
            text_content=False
        )
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=100,
            chunk_overlap=5
        )
        docs = loader.load_and_split(text_splitter=text_splitter)

        # Create agent and retriever
        agent, retriever = create_agent(
            docs=docs,
            llm=chat,
            embeddings=embeddings,
            persist_directory="doc_emb"
        )
        st.session_state.agent = agent

    except DocumentLoaderException as e:
        st.error(f"Loader error: {e}")
    except Exception as e:
        st.error(f"Startup failed: {e}")

# Show chat history
if st.session_state.chat_history:
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.markdown(msg)

# Accept user query
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
