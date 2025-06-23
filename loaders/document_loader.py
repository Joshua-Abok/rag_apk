from typing import Any 
import os 

from langchain.document_loaders import (
    JSONLoader, PyPDFLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def _select_loader_class(file_path: str): 
    '''Return the appropriate loader class for the given file extension.'''
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json": 
        return JSONLoader
    if ext == ".pdf": 
        return PyPDFLoader 
    raise DocumentLoaderException(f"unsupported file type: {ext}")


class DocumentLoaderException(Exception): 
    pass


class DocumentLoader: 
    '''document loader for .json and .pdf sources'''

    def __init__(self, file_path: str, **kwargs: Any):
        self.file_path = file_path
        self.kwargs = kwargs 
        LoaderClass = _select_loader_class(file_path)
        # self.loader = self._select_loader()
        self.loader = LoaderClass(file_path=file_path, **kwargs)

    def load(self) -> list[Document]: 
        '''load raw documents'''
        return self.loader.load()
    
    def load_and_split(self, text_splitter: CharacterTextSplitter):
        '''Load documents & split into chunks'''
        raw_docs = self.load()
        processed: list[Document] = []

        for doc in raw_docs: 
            content = doc.page_content
            if isinstance(content, dict): 
                title = content.get("title", "")
                body = content.get("body", "")
                text = f"{title}\n\n{body}"
            else: 
                text = str(content)
            processed.append(Document(page_content=text, metadata=doc.metadata))
        return text_splitter.split_documents(processed)
