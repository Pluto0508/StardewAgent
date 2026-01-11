from typing import (
    List,
    Dict,
    Union,
    Optional,
    Any
)
import os

from stardojo.config import Config
from stardojo.log import Logger
from stardojo.memory.base import BaseMemory, Image
from stardojo.memory.vector_store import VectorStore
from stardojo.utils.json_utils import load_json, save_json

config = Config()
logger = Logger()


class BasicVectorMemory(BaseMemory):

    storage_filename = "vector_memory.json"

    def __init__(
        self,
        memory_path: str,
        vectorstores: VectorStore,
        embedding_provider: Any,
        memory: Optional[Dict] = None,
        #knowledge_base_path: Optional[str] = None
    ):
        if memory is None:
            self.memory: Dict = {}
        else:
            self.memory = memory
        self.memory_path = memory_path
        self.vectorstores = vectorstores
        self.embedding_provider = embedding_provider
        #self.knowledge_base = None
        #if knowledge_base_path and os.path.exists(knowledge_base_path):
            #self.knowledge_base = KnowledgeBaseMemory(knowledge_base_path, embedding_provider)


    def add(
        self,
        data: Dict[str, Union[str, Image]],
    ) -> None:
        """Add data to memory.

        Args:
            data: the mapping from unique name (id) to text/image.
        """

        keys: List[str] = list(data.keys())
        embeddings = []

        for k in keys:
            embeddings.append(self.embedding_provider.embed_query(data[k]["description"]))
            instruction = data[k]["instruction"]
            screenshot = data[k]["screenshot"]
            timestep = data[k]["timestep"]
            description = data[k]["description"]
            inventory = data[k]["inventory"]

            self.memory[k] = {
                "instruction": instruction,
                "screenshot": screenshot,
                "timestep": timestep,
                "description": description,
                "inventory": inventory,
            }

        self.vectorstores['description'].add_embeddings(keys, embeddings)


    def similarity_search(
        self,
        data: Union[str, Image],
        top_k: int = 3,
        **kwargs,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the vectorstore.

        Args:
            data: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """
        query_embedding = self.embedding_provider.embed_query(data)
        key_and_score = self.vectorstores['description'].similarity_search(query_embedding, top_k)

        return [self.memory[k] for k, score in key_and_score]


    def recent_search(
        self,
        recent_k: int = 3,
        **kwargs,
    ) -> List[Union[str, Image]]:
        """Retrieve the recent k keys

        Args:
            recent_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values of the recent k memory.
        """

        keys = list(self.memory.keys()) # the order of adding
        recent_k = min(recent_k,len(keys))
        return [self.memory[k] for k in keys[len(keys) - recent_k : len(keys)]]
    
    ##def query_knowledge_and_memory(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        results = {
            "knowledge_base_results": [],
            "memory_results": [],
            "recent_memory_results": []
        }
        
        if self.knowledge_base:
            results["knowledge_base_results"] = self.knowledge_base.query_knowledge(query, top_k)
        
        memory_results = self.similarity_search(query, top_k)
        results["memory_results"] = memory_results
        
        recent_results = self.recent_search(top_k)
        results["recent_memory_results"] = recent_results
        
        return results
    
    def add_experience(self, 
                      state: Dict, 
                      action: str, 
                      result: str, 
                      success: bool,
                      reasoning: str = "") -> None:
        experience_id = f"exp_{int(time.time())}_{len(self.memory)}"
        experience_data = {
            "state": state,
            "action": action,
            "result": result,
            "success": success,
            "reasoning": reasoning,
            "timestamp": time.time(),
            "description": f"Action: {action}, Result: {result}, Success: {success}"
        }
        
        self.memory[experience_id] = experience_data
        embedding = self.embedding_provider.embed_query(experience_data["description"])
        self.vectorstores['description'].add_embeddings([experience_id], [embedding])


    def load(self):
        self.load()


    def load(
        cls,
        memory_path: str,
        vectorstore: VectorStore,
        embedding_provider: Any,
    ) -> "BasicVectorMemory":
        """Load the memory from the local file."""

        memory = load_json(os.path.join(cls.memory_path, cls.storage_filename))

        return cls(
            memory_path=memory_path,
            vectorstore=vectorstore,
            embedding_provider=embedding_provider,
            memory=memory,
        )


    def save(self) -> None:
        """Save the memory to the local file."""
        save_json(file_path = os.path.join(self.memory_path, self.storage_filename), json_dict = self.memory, indent = 4)
        self.vectorstores.save()

##class KnowledgeBaseMemory:
    
    def __init__(self, knowledge_path: str, embedding_provider: Any):
        self.knowledge_path = knowledge_path
        self.embedding_provider = embedding_provider
        self.knowledge_data = self._load_knowledge_base()
        self.vectorstore = self._build_knowledge_vectorstore()
    
    def _load_knowledge_base(self) -> Dict:
        try:
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return {}
    
    def _build_knowledge_vectorstore(self) -> VectorStore:
        vectorstore = VectorStore()
        if not self.knowledge_data:
            return vectorstore
        
        knowledge_chunks = self._chunk_knowledge_data()
        keys = []
        embeddings = []
        
        for i, chunk in enumerate(knowledge_chunks):
            key = f"knowledge_{i}"
            keys.append(key)
            embedding = self.embedding_provider.embed_query(chunk)
            embeddings.append(embedding)
        
        vectorstore.add_embeddings(keys, embeddings)
        return vectorstore
    
    def _chunk_knowledge_data(self) -> List[str]:
        chunks = []
        
        def extract_content(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        extract_content(value, current_prefix)
                    else:
                        chunks.append(f"{current_prefix}: {value}")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    extract_content(item, f"{prefix}[{i}]")
        
        extract_content(self.knowledge_data)
        return chunks
    
    def query_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.knowledge_data:
            return []
        
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.vectorstore.similarity_search(query_embedding, top_k)
        
        return [{"content": result, "score": score} for result, score in results]