import os
from typing import List

from langchain.embeddings.base import Embeddings
from pinecone import Pinecone


class PineconeEmbeddings(Embeddings):
    """Custom wrapper for Pinecone embeddings."""

    def __init__(
        self,
        api_key: str = os.environ.get("PINECONE_API_KEY"),
        model_name="multilingual-e5-large",
    ):
        self.pc = Pinecone(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Pinecone's embedding model."""
        embeddings =  self.pc.inference.embed(
            model=self.model_name, inputs=texts, parameters={"input_type": "passage"}
        )
        return [e['values'] for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.pc.inference.embed(
            model=self.model_name, inputs=[text], parameters={"input_type": "query"}
        )
        return [e['values'] for e in embeddings]