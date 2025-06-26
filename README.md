# RAG Pipeline
This repository is part of the **Agentic AI Developer Certification program** by [Ready Tensor](https://www.readytensor.ai)
and it is linked to the publication:**"Agentic AI Developer Certification: RAG-based AI assistant for Exploring Ready Tensor Publications"** on [Ready Tensor](https://www.readytensor.ai)

## Project Description
This repository provides a simple, extensible Retrieval-Augmented Generation (RAG) pipeline using LangChain and Chroma. It includes a generic `DocumentLoader` class that supports JSON and PDF inputs, handles JQ extraction for JSON, and performs text splitting, embedding, and similarity search.

## Features
* **DocumentLoader**: Unified loader for `.json` and `.pdf` files.
* **JSON extraction**: Built-in JQ filtering to extract specific fields (e.g., title, description).
* **Text normalization**: Converts dict-based content into strings before splitting.
* **Text splitting**: Uses `CharacterTextSplitter` to chunk large documents.
* **Embedding & Vectorstore**: Integrates with `OpenAIEmbeddings` and Chroma for storage and retrieval.
* **Similarity Search**: Example querying for semantic search over your documents.

## Repository Structure
```
rag_apk/
├── app/                            
│   ├── main.py              # Core LLM & vector DB implementation
│   └── path.py              # File path configurations
├── loaders/                        
│   └── document_loader.py   # Core DocumentLoader implementation
├── agents/                      
│   └── agent.py             # Agent logic
├── data/                         
│   └── project_1_publications.json  # Sample publications 
├── .env.example             # Environment variables template
├── .gitignore
├── requirements.txt         # Python dependencies
├── README.md
├── LICENSE
├── usage_example.png        # Example usage screenshot
```
## Prerequisites
* Python 3.10+
* A valid OpenAI API key (OPENAI_API_KEY environment variable)

## Installation
1. **Clone the repo** and be sure you're on the `main` branch:

   ```bash
   git clone https://github.com/Joshua-Abok/rag_apk
   cd rag_apk
   ```
2. **Install dependencies**   
   Install required packages (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```
3. **Create and activate a virtual environment (recommended):**      
   
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux / macOS
   .\.venv\Scripts\activate      # Windows
   ```
3. **Set up environment variables**  
   Add your OpenAI API key to a `.env` file in your the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
## Running the Application  
1. **Prepare data**    
   Ensure `project_1_publications.json` is present in the data/ directory (or your configured DATA_DIR).  

2. **Launch the App**     
   Run Streamlit from the project root:  
  
   ```
   streamlit run app/main.py
   ```
   
3. **Access the Interface**          
   Open your browser to the local Streamlit URL (usually http://localhost:8501).        

You can now interact with the Ready Tensor Publication Explorer!  

## Usage Examples 
![Usage Example](usage_example.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
