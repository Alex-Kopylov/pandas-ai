import uuid
from typing import Any, Callable, Iterable, List, Optional, Union

import pinecone

from pandasai.helpers.logger import Logger
from pandasai.vectorstores.vectorstore import VectorStore


class Pinecone(VectorStore):
    """
    Implementation of Pinecone vector store
    """

    _logger: Logger

    def __init__(
        self,
        api_key: str,
        index: Union[str, pinecone.Index] = "pandasai",
        embedding_function: Optional[Callable[[List[str]], List[float]]] = None,
        dimensions=1536,
        metric="cosine",
        pool_threads: int = 1,
        specs: pinecone.ServerlessSpec = None,
        max_samples: int = 1,
        similary_threshold: int = 1.5,
        logger: Optional[Any] = None,
    ) -> None:
        self._logger = Logger() if logger is None else logger
        self._logger.log("Initializing Pinecone vector store")
        self._max_samples = max_samples
        self._similarity_threshold = similary_threshold
        self._api_key = api_key

        self._metatext_key = "text"

        self._embedding_function = embedding_function

        # Initialize these as None first
        self._pinecone = None
        self._index = None

        try:
            self._pinecone = pinecone.Pinecone(
                api_key=api_key, pool_threads=pool_threads
            )

            if isinstance(index, str):
                if index not in self._pinecone.list_indexes().names():
                    self._index = self._pinecone.create_index(
                        name=index,
                        dimension=dimensions,
                        metric=metric,
                        spec=specs
                        or pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
                    )

                self._index = self._pinecone.Index(name=index)
            else:
                self._index = index

            self._logger.log("Successfully initialized index")
        except Exception as e:
            self.cleanup()
            raise e

    def cleanup(self):
        """Clean up Pinecone resources"""
        if hasattr(self, "_index") and self._index is not None:
            self._index = None
        if hasattr(self, "_pinecone") and self._pinecone is not None:
            self._pinecone = None

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted"""
        self.cleanup()

    def add_question_answer(
        self,
        queries: Iterable[str],
        codes: Iterable[str],
        ids: Optional[Iterable[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if len(queries) != len(codes):
            raise ValueError(
                f"Queries and codes dimension doesn't match {len(queries)} != {len(codes)}"
            )

        if ids is None:
            ids = [f"{str(uuid.uuid4())}-qa" for _ in queries]

        metadatas = metadatas or [{} for _ in ids]

        qa_str = [self._format_qa(query, code) for query, code in zip(queries, codes)]

        for index, metadata in enumerate(metadatas):
            metadata[self._metatext_key] = qa_str[index]

        vector_data = [
            {"id": ids[index], "values": qa, "metadata": metadatas[index]}
            for index, qa in enumerate(self._embedding_function(qa_str))
        ]

        self._index.upsert(vectors=vector_data, namespace="qa")

        return ids

    def add_docs(
        self,
        docs: Iterable[str],
        ids: Optional[Iterable[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if not isinstance(docs, list):
            raise ValueError("Docs must be list of strings!")

        if ids is None:
            ids = [f"{str(uuid.uuid4())}-docs" for _ in docs]

        metadatas = metadatas or [{} for _ in ids]

        doc_embeddings = self._embedding_function(docs)

        for index, metadata in enumerate(metadatas):
            metadata[self._metatext_key] = docs[index]

        vector_data = [
            {"id": ids[index], "values": doc, "metadata": metadatas[index]}
            for index, doc in enumerate(doc_embeddings)
        ]

        self._index.upsert(vectors=vector_data, namespace="docs")

        return ids

    def update_question_answer(
        self,
        ids: Iterable[str],
        queries: Iterable[str],
        codes: Iterable[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if len(queries) != len(codes):
            raise ValueError(
                f"Queries and codes dimension doesn't match {len(queries)} != {len(codes)}"
            )

        qa_str = [self._format_qa(query, code) for query, code in zip(queries, codes)]

        metadatas = metadatas or [{} for _ in ids]

        for index, metadata in enumerate(metadatas):
            metadata[self._metatext_key] = qa_str[index]

        for index, qa in enumerate(self._embedding_function(qa_str)):
            self._index.update(
                id=ids[index], values=qa, set_metadata=metadatas[index], namespace="qa"
            )

    def update_docs(
        self,
        ids: Iterable[str],
        docs: Iterable[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        doc_embeddings = self._embedding_function(docs)

        metadatas = metadatas or [{} for _ in ids]

        for index, metadata in enumerate(metadatas):
            metadata[self._metatext_key] = docs[index]

        for index, doc in enumerate(doc_embeddings):
            self._index.update(
                id=ids[index],
                values=doc,
                set_metadata=metadatas[index],
                namespace="docs",
            )

    def delete_question_and_answers(
        self, ids: Optional[List[str]] = None
    ) -> Optional[bool]:
        self._index.delete(ids=ids, namespace="qa")
        return True

    def delete_docs(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        self._index.delete(ids=ids, namespace="docs")
        return True

    def get_relevant_question_answers(
        self, question: str, k: Union[int, None] = None
    ) -> List[dict]:
        k = k or self._max_samples

        questions = self._embedding_function([question])

        results = self._index.query(
            vector=questions,
            top_k=k,
            include_metadata=True,
            namespace="qa",
            include_values=True,
        )

        return self._filter_docs_based_on_distance(results, self._similarity_threshold)

    def get_relevant_docs(self, question: str, k: int = None) -> List[dict]:
        k = k or self._max_samples

        questions = self._embedding_function([question])

        results = self._index.query(
            vector=questions,
            top_k=k,
            include_metadata=True,
            namespace="docs",
            include_values=True,
        )

        return self._filter_docs_based_on_distance(results, self._similarity_threshold)

    def get_relevant_question_answers_by_id(self, ids: Iterable[str]) -> List[dict]:
        return self._index.fetch(id=ids, namespace="qa")

    def get_relevant_docs_by_id(self, ids: Iterable[str]) -> List[dict]:
        return self._index.fetch(id=ids, namespace="docs")

    def get_relevant_qa_documents(self, question: str, k: int = None) -> List[str]:
        return self.get_relevant_question_answers(question, k)["documents"][0]

    def get_relevant_docs_documents(self, question: str, k: int = None) -> List[str]:
        return self.get_relevant_docs(question, k)["documents"][0]

    def _filter_docs_based_on_distance(self, documents, threshold: int) -> List[str]:
        filtered_data = [
            (
                document["metadata"][self._metatext_key],
                document["score"],
                document["metadata"],
                document["id"],
            )
            for document in documents["matches"]
            if document["score"] < threshold
        ]

        return {
            key: [[data[i] for data in filtered_data]]
            for i, key in enumerate(["documents", "distances", "metadata", "ids"])
        }

    def _format_qa(self, query: str, code: str) -> str:
        """Format question and answer for storage"""
        return f"Q: {query}\nA: {code}"
