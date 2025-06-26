import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import os
import re
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chroma_db.chroma_client import get_chroma_client, get_or_create_collection

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_MAIN_JAVA = Path("/Users/tanmay/Desktop/AMRIT/Common-API/src/main/java")
SRC_TEST_JAVA = Path("/Users/tanmay/Desktop/AMRIT/Common-API/src/test/java")
CHROMA_COLLECTION = 'test_examples_collection'

EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'

# --- UTILS ---
def find_java_files(root_dir, pattern):
    return [p for p in root_dir.rglob(pattern)]

def extract_class_name(java_code):
    match = re.search(r'class\s+(\w+)', java_code)
    return match.group(1) if match else None

def extract_method_signatures(java_code):
    pattern = re.compile(r'(public|protected|private)\s+[\w<>,\[\]]+\s+(\w+)\s*\([^)]*\)')
    return [m.group(0) for m in pattern.finditer(java_code)]

# --- MAIN ---
def main():
    print(f"[RAG INDEX] Scanning for test/source pairs...")
    test_files = find_java_files(SRC_TEST_JAVA, '*Test.java')
    print(f"[RAG INDEX] Found {len(test_files)} test files.")

    # Setup embeddings and ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chroma_client = get_chroma_client()
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings
    )

    for test_file in test_files:
        try:
            test_code = test_file.read_text(encoding='utf-8')
            test_class_name = extract_class_name(test_code)
            # Guess source file path (same package, remove 'Test' suffix)
            if not test_class_name or not test_class_name.endswith('Test'):
                continue
            source_class_name = test_class_name[:-4]
            # Find package declaration
            package_match = re.search(r'package\s+([\w\.]+);', test_code)
            package_path = package_match.group(1).replace('.', '/') if package_match else ''
            source_file = SRC_MAIN_JAVA / package_path / f'{source_class_name}.java'
            if not source_file.exists():
                print(f"[RAG INDEX] Source file not found for test: {test_file}")
                continue
            source_code = source_file.read_text(encoding='utf-8')
            method_signatures = extract_method_signatures(source_code)
            # Compose metadata
            metadata = {
                'type': 'test_example',
                'test_class_name': test_class_name,
                'source_class_name': source_class_name,
                'package': package_match.group(1) if package_match else '',
                'test_file': str(test_file),
                'source_file': str(source_file),
                'method_signatures': '\n'.join(method_signatures)
            }
            # Embed and add to ChromaDB
            doc_text = f"SOURCE:\n{source_code}\n\nTEST:\n{test_code}"
            vectorstore.add_texts([doc_text], metadatas=[metadata])
            print(f"[RAG INDEX] Indexed test/source pair: {test_class_name} <-> {source_class_name}")
        except Exception as e:
            print(f"[RAG INDEX] Error processing {test_file}: {e}")

    print("[RAG INDEX] Indexing complete.")

def retrieve_similar_test_examples(class_code, method_signatures, vectorstore, top_n=3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    query_embedding = embeddings.embed_documents([class_code + '\n' + '\n'.join(method_signatures)])[0]
    results = vectorstore.similarity_search_by_vector(query_embedding, k=top_n, filter={'type': 'test_example'})
    return results  # Each result: {'page_content': ..., 'metadata': ...}

if __name__ == '__main__':
    main() 