import logging
import os
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass


class RAGLangchain:
  def __init__(self, interrupt_count: SynchronizedClass):
    self.logger = logging.getLogger("speech_to_speech.rag_langchain")
    self.interrupt_count = interrupt_count

    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.db = Chroma(
      collection_name="RAG",
      embedding_function=self.embeddings
      )
    self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=125, is_separator_regex=False)
    
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    core_memory_dir = os.path.join(project_root_dir, "data", "core_memory_db")
    self.core_memory_db = Chroma(
      collection_name="core_memory_db",
      embedding_function=self.embeddings,
      persist_directory=core_memory_dir
    )


  def add_memory(self, text, timestamp):
    text = text.strip()

    if text == "None":
      return
    
    self.logger.debug(f"Adding memory: {datetime.fromtimestamp(timestamp).strftime("%m-%d-%y %H:%M:%S")} - {text}")

    self.core_memory_db.add_texts(
      texts=[text],
      metadatas=[{"timestamp": timestamp}],
      ids=[f"memory_{timestamp}"])


  def get_memories(self):
    memories = self.core_memory_db.get()
    all_memories = [
      (doc, datetime.fromtimestamp(meta["timestamp"]).strftime("%m-%d-%y %H:%M:%S")) 
      for doc, meta in zip(memories["documents"], memories["metadatas"])
    ]
    
    all_memories.sort(key=lambda x: x[1])
    
    memories = []
    for memory,time in memories:
      memories.append(f"{time} - {memory}")

    return memories


  def add_document(self, document):
    documents = []

    for chunk in self.splitter.split_text(document["content"]):
      documents.append(Document(chunk, metadata={"source": document["source"]}))

    for table in document["tables"]:
      documents.append(Document(table, metadata={"source": document["source"]}))

    self.db.add_documents(documents)


  def query(self, prompt):
    results = []
    
    search_results = self.db.similarity_search_with_relevance_scores(
      query=prompt,
      k = 7
    )

    # print(search_results)

    for document, score in search_results:
      results.append({
        "content": document.page_content,
        "source": document.metadata["source"],
        "score": score
      })

    return results
  