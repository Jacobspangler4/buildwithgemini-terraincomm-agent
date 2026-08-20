import os
import vertexai
from vertexai.preview import rag
from vertexai.preview.rag.utils import resources as rr

PROJECT_ID = "qwiklabs-gcp-04-2a75196ccefd"
LOCATION = "us-central1"  # Serverless RAG Engine is us-central1 only
GCS_PATH = "gs://terraincomm-assets-042a75196ccefd/rag/pg49513.txt"

PARSING_PROMPT = (
    "Extract all relevant technical, historical, and factual details from this document. "
    "Ignore Project Gutenberg headers, footers, and boilerplate text. "
    "Output clean prose."
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

# 1. Switch region's RAG managed DB to serverless mode
cfg = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragEngineConfig"
try:
    rag.update_rag_engine_config(
        rag_engine_config=rag.RagEngineConfig(
            name=cfg,
            rag_managed_db_config=rag.RagManagedDbConfig(mode=rr.Serverless()),
        )
    )
    print("Updated ragEngineConfig to Serverless mode.")
except Exception as e:
    print(f"Notice on update_rag_engine_config: {e}")

# 2. Create serverless RAG corpus
corpus = rag.create_corpus(
    display_name="terraincomm-gutenberg-corpus",
    embedding_model_config=rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-005"
    ),
)
print(f"CORPUS_NAME={corpus.name}")

# 3. Import file, chunk, and embed
resp = rag.import_files(
    corpus_name=corpus.name,
    paths=[GCS_PATH],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
    ),
    llm_parser=rag.LlmParserConfig(
        model_name="gemini-2.5-flash",
        custom_parsing_prompt=PARSING_PROMPT,
    ),
)
print(f"Imported RAG files count: {resp.imported_rag_files_count}")
