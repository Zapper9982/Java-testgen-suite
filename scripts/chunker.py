import os
import json
import sys

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    USE_LANGCHAIN = True
except ImportError:
    print("Warning: langchain package not found. Falling back to simple chunking.")
    USE_LANGCHAIN = False

# config marker 
INPUT_DIR = 'processed_output'      
OUTPUT_DIR = 'chunked_output'      
CHUNK_SIZE = 800                    
CHUNK_OVERLAP = 100              


os.makedirs(OUTPUT_DIR, exist_ok=True)


if USE_LANGCHAIN:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(text.split()),
    )
else:
    def splitter_split_text(text):
        words = text.split()
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = words[i:i + CHUNK_SIZE]
            chunks.append(" ".join(chunk))
        return chunks


all_chunks = []

for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.endswith('.txt'):
            continue
        file_path = os.path.join(root, fname)
        rel_path = os.path.relpath(file_path, INPUT_DIR).replace(os.sep, '/')

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if USE_LANGCHAIN:
            chunks = splitter.split_text(content)
        else:
            chunks = splitter_split_text(content)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                'metadata': {
                    'source_path': rel_path,
                    'chunk_index': idx,
                },
                'text': chunk
            })

output_file = os.path.join(OUTPUT_DIR, 'langchain_chunks.json')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as out:
    json.dump(all_chunks, out, indent=2)

print(f"Chunking complete: {len(all_chunks)} chunks saved to {output_file}")