import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


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
#st.caption("Ask questions about the Ready Tensor publications.")

import json

# Load publications from JSON
with open(f"{DATA_DIR}/project_1_publications.json", "r", encoding="utf-8") as f:
    publications = json.load(f)

# Select publication title
#titles = [pub.get("title", "No Title") for pub in filtered_pubs]
titles = [pub.get("title", "No Title") for pub in publications]
selected_title = st.selectbox(" 🔍Select a publication to view details", [""] + titles)

# Show selected publication content
if selected_title and selected_title in titles:
    selected_pub = next(pub for pub in publications if pub.get("title", "No Title") == selected_title)
    st.markdown(f"### {selected_pub.get('title', 'No Title')}")
    st.markdown(selected_pub.get("publication_description", "No Description"))
    st.markdown("---")


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

st.caption("Ask questions about the sample dataset Ready Tensor publications.")

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
                
                # Save response to output/ folder
                from datetime import datetime
                import os
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"response_{timestamp}.txt"
                filepath = os.path.join(OUTPUTS_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as fn:
                    fn.write(f"Query: {query}\n\nAnswer: {answer}")

            except Exception as e:
                st.error(f"Agent error: {e}")
