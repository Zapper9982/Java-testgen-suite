import os
import re
import yaml
import json 
from pathlib import Path
from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import uuid 
import shutil 

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    USE_LANGCHAIN_REC_SPLITTER = True
except ImportError:
    print("Warning: langchain RecursiveCharacterTextSplitter not found. Large semantic chunks will not be further split.")
    USE_LANGCHAIN_REC_SPLITTER = False

#INITIALISATION 
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
global_semantic_chunker = SemanticChunker(embeddings=embeddings)

# CONFIG

GENERIC_CHUNK_SIZE = 2000
GENERIC_CHUNK_OVERLAP = 100

JAVA_FILE_MAX_CHARS_FOR_SINGLE_CHUNK = 10000 
LARGE_JAVA_CHUNK_THRESHOLD = 15000 
METHOD_GROUP_MAX_CHARS = 7500 
MIN_CHUNK_CONTENT_LENGTH = 50 


TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent
PROCESSED_INPUT_DIR = TESTGEN_AUTOMATION_ROOT / 'processed_output'
CHUNKED_OUTPUT_DIR = TESTGEN_AUTOMATION_ROOT / 'chunked_output'
CHUNKED_OUTPUT_FILE = CHUNKED_OUTPUT_DIR / 'chunks.json'



def _infer_file_type_from_content(filename_base: str, content: str) -> str:
 
    if filename_base == "package-info":
        return 'ignored_metadata' 


    if (filename_base.lower() in ["application", "bootstrap"] and
        (content.strip().startswith("spring.") or content.strip().startswith("server."))):
        return 'config'

    if "---" in content or any(line.strip().startswith(key) for key in ["server:", "spring:", "database:", "management:"] for line in content.splitlines() if line.strip()):
        return 'config'
    if any(line.strip().startswith(key) and '=' in line for key in ["spring.", "server.", "my."] for line in content.splitlines() if line.strip() and not line.strip().startswith("#")):
        return 'config'

    if (re.search(r"\b(package|import|class|interface|enum)\b", content) and
        re.search(r"\b(public|private|protected)\b", content) and
        re.search(r"\{|\}", content)):
        return 'java'

    return 'unknown'

def load_files_from_txt_directory(input_dir: Path) -> List[Dict[str, str]]:

    loaded_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                path = Path(root) / file 
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        original_filename_base = path.stem 

                        inferred_type = _infer_file_type_from_content(original_filename_base, content)

                        loaded_files.append({
                            "filepath": str(path),
                            "original_filename_base": original_filename_base,
                            "content": content,
                            "inferred_type": inferred_type
                        })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return loaded_files

def parse_java_code_with_metadata(code_content: str, filename_base: str) -> Dict[str, Any]:
    """
    Parses Java code to extract class-level info and individual methods.
    Returns structured data that can be used to form chunks.
    """
    parsed_data: Dict[str, Any] = {
        "filename": f"{filename_base}.java", 
        "class_name": "UnknownClass",
        "class_annotations": [],
        "class_header_content": "",
        "methods": [],
        "fields": []
    }
    

    class_match = re.search(
        r"(?:^@\w+\s*)*\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*(class|interface|enum)\s+(\w+)\s*(?:extends\s+[\w\.]+)?\s*(?:implements\s+[\w\.,\s]+)?\s*\{",
        code_content, re.MULTILINE
    )

    if class_match:
        parsed_data["class_name"] = class_match.group(2)
        annotations_block = code_content[:class_match.start()]
        parsed_data["class_annotations"] = re.findall(r"@(\w+)(?:\([^)]*\))?", annotations_block)
        
  
        class_body_start_index = code_content.find('{', class_match.end()) # Find the brace after class signature
        if class_body_start_index != -1:
          
            first_member_match = re.search(
                r"(?:(?:public|private|protected|static|final|abstract|transient|volatile|synchronized|native|strictfp)\s+)*[\w\.<>\[\]]+\s+\w+[\s;({]",
                code_content[class_body_start_index+1:] # Search within class body
            )
            if first_member_match:
                parsed_data["class_header_content"] = code_content[class_match.start(): class_body_start_index + 1 + first_member_match.start()].strip()
            else:
                
                parsed_data["class_header_content"] = code_content[class_match.start():].strip()
        else:
            parsed_data["class_header_content"] = code_content.strip() # whole file if no class body found


    field_pattern = re.compile(
        r"@(Autowired|Inject)\s+(?:public|private|protected)?\s+([\w\.<>\[\]]+)\s+(\w+);",
        re.MULTILINE
    )
    for match in field_pattern.finditer(code_content):
        parsed_data["fields"].append({
            "annotation": match.group(1),
            "type": match.group(2),
            "name": match.group(3)
        })


    method_pattern = re.compile(
    r"(?:^@\w+\s*)*\s*(public|private|protected|static|final|abstract|synchronized|native|strictfp)?\s*(?:<[\w,\s]+>)?\s*([\w\.<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+(?:[\w\.]+(?:,\s*[\w\.]+)*))?\s*\{",
    re.MULTILINE
     )
    method_matches = list(method_pattern.finditer(code_content))
    for i, match in enumerate(method_matches):
        method_start = match.start()
       
        annotations_block_start = code_content.rfind('\n', 0, method_start) + 1
        method_annotations = re.findall(r"@(\w+)(?:\([^)]*\))?", code_content[annotations_block_start:method_start])

        brace_balance = 1
        method_end = match.end()
        while method_end < len(code_content) and brace_balance > 0:
            if code_content[method_end] == '{':
                brace_balance += 1
            elif code_content[method_end] == '}':
                brace_balance -= 1
            method_end += 1

        method_content = code_content[method_start:method_end].strip()

        parsed_data["methods"].append({
            "method_name": match.group(3),
            "return_type": match.group(2),
            "parameters": match.group(4),
            "content": method_content,
            "annotations": method_annotations,
            "start_char": method_start,
            "end_char": method_end
        })
    return parsed_data


def parse_config_file(content: str, filename_base: str) -> List[Dict[str, str]]:
    """Parses config file content into smaller chunks based on properties/YAML structure."""
    config_chunks = []
    
   
    is_yaml = "---" in content or any(line.strip().startswith(key) for key in ["server:", "spring:", "management:"] for line in content.splitlines() if line.strip())

    if is_yaml:
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
           
                def flatten_dict(d, parent_key=''):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key).items())
                        else:
                            items.append((new_key, str(v)))
                    return dict(items)
                
                flat_data = flatten_dict(data)
                for key, value in flat_data.items():
                    config_chunks.append({"type": "yaml_entry", "content": f"{key}: {value}"})
            else: 
                config_chunks.append({"type": "yaml_full", "content": content})
        except yaml.YAMLError as e:
            print(f"Error parsing YAML content from {filename_base}.txt: {e}. Chunking as full content.")
            config_chunks.append({"type": "yaml_full", "content": content})
    else: 
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                config_chunks.append({"type": "property", "content": line})
                
    return config_chunks


def semantic_chunk_documents(docs: List[Document], chunker: SemanticChunker) -> List[Document]:

    return chunker.split_documents(docs)


def process_codebase_from_txt_for_chunking(input_dir: Path, output_json_path: Path) -> List[Dict[str, Any]]:

    all_final_chunks: List[Dict[str, Any]] = []
    chunk_type_counts: Dict[str, int] = {} # For logging chunk types

    
    generic_splitter = None
    if USE_LANGCHAIN_REC_SPLITTER:
        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=GENERIC_CHUNK_SIZE,
            chunk_overlap=GENERIC_CHUNK_OVERLAP,
            length_function=len,
        )

    loaded_txt_files = load_files_from_txt_directory(input_dir)


    if CHUNKED_OUTPUT_DIR.exists():
        print(f"Clearing previous chunked output directory: {CHUNKED_OUTPUT_DIR}")
        shutil.rmtree(CHUNKED_OUTPUT_DIR)
    CHUNKED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for file_info in loaded_txt_files:
        filepath_txt = Path(file_info["filepath"]) 
        original_filename_base = file_info["original_filename_base"] 
        content = file_info["content"]
        inferred_type = file_info["inferred_type"]

        if inferred_type == 'ignored_metadata':
            print(f"Skipping metadata-only file: {filepath_txt.name}")
            continue

        common_metadata_base = {
            "filename": f"{original_filename_base}.java" if inferred_type == 'java' else f"{original_filename_base}.txt",
            "filepath_txt": str(filepath_txt) # Store as string
        }

        if inferred_type == 'java':
           
            if len(content) <= JAVA_FILE_MAX_CHARS_FOR_SINGLE_CHUNK:
                if len(content) >= MIN_CHUNK_CONTENT_LENGTH:
                    print(f"Treating small Java file '{filepath_txt.name}' as a single chunk. Length: {len(content)}")
                    all_final_chunks.append({
                        "chunk_id": uuid.uuid4().hex,
                        "chunk_content": content,
                        "chunk_metadata": { 
                            **common_metadata_base,
                            "type": "java_file_full",
                            "class_name": original_filename_base 
                        }
                    })
                    chunk_type_counts["java_file_full"] = chunk_type_counts.get("java_file_full", 0) + 1
                else:
                    print(f"Skipping tiny Java file chunk from {filepath_txt.name}. Length: {len(content)}")
                continue # Move to the next file


            parsed_java_data = parse_java_code_with_metadata(content, original_filename_base)
            

            if parsed_java_data["class_header_content"]:
                class_header_content = parsed_java_data["class_header_content"]
                class_header_metadata = common_metadata_base.copy()
                class_header_metadata.update({
                    "type": "java_class_header",
                    "class_name": parsed_java_data["class_name"],
                    "class_annotations": ", ".join(parsed_java_data["class_annotations"]) if parsed_java_data["class_annotations"] else "",
                    "fields": json.dumps(parsed_java_data["fields"]) # Store fields as JSON string
                })
               
                if len(class_header_content) > LARGE_JAVA_CHUNK_THRESHOLD and USE_LANGCHAIN_REC_SPLITTER:
                    print(f"Applying semantic chunking to large class header from {filepath_txt.name}.")
                    temp_doc = Document(page_content=class_header_content, metadata=class_header_metadata)
                    semantically_split_docs = semantic_chunk_documents([temp_doc], global_semantic_chunker)
                    for doc in semantically_split_docs:
                        if len(doc.page_content) >= MIN_CHUNK_CONTENT_LENGTH:
                            all_final_chunks.append({
                                "chunk_id": uuid.uuid4().hex,
                                "chunk_content": doc.page_content,
                                "chunk_metadata": doc.metadata # Renamed to chunk_metadata
                            })
                            chunk_type_counts[doc.metadata.get("type", "java_class_header_semantic")] = chunk_type_counts.get(doc.metadata.get("type", "java_class_header_semantic"), 0) + 1
                        else:
                            print(f"Skipping tiny semantic sub-chunk from {filepath_txt.name} (class header). Length: {len(doc.page_content)}")
                else:
                    if len(class_header_content) >= MIN_CHUNK_CONTENT_LENGTH:
                        all_final_chunks.append({
                            "chunk_id": uuid.uuid4().hex,
                            "chunk_content": class_header_content,
                            "chunk_metadata": class_header_metadata # Renamed to chunk_metadata
                        })
                        chunk_type_counts[class_header_metadata.get("type", "java_class_header")] = chunk_type_counts.get(class_header_metadata.get("type", "java_class_header"), 0) + 1
                    else:
                        print(f"Skipping tiny class header chunk from {filepath_txt.name}. Length: {len(class_header_content)}")


            # --- Subsequent Chunks: Grouped Methods ---
            current_method_group_content = ""
            current_method_group_method_names = [] # To store names of methods in this group
            
            for i, method_data in enumerate(parsed_java_data["methods"]):
                method_content = method_data["content"]
                
                # Append method content and metadata to current group
                current_method_group_content += method_content + "\n\n" # Add separators
                current_method_group_method_names.append(method_data["method_name"])

                # If current group exceeds threshold or it's the last method
                if len(current_method_group_content) > METHOD_GROUP_MAX_CHARS or i == len(parsed_java_data["methods"]) - 1:
                    # Create a single metadata for the combined chunk
                    combined_metadata = common_metadata_base.copy()
                    combined_metadata.update({
                        "type": "java_method_group",
                        "class_name": parsed_java_data["class_name"],
                        "method_count": len(current_method_group_method_names),
                        "methods_in_group": ", ".join(current_method_group_method_names) # Joined method names
                    })

                    # Apply semantic chunking if the combined method group is very large
                    if len(current_method_group_content) > LARGE_JAVA_CHUNK_THRESHOLD and USE_LANGCHAIN_REC_SPLITTER:
                        print(f"Applying semantic chunking to large method group from {filepath_txt.name}.")
                        temp_doc = Document(page_content=current_method_group_content, metadata=combined_metadata)
                        semantically_split_docs = semantic_chunk_documents([temp_doc], global_semantic_chunker)
                        for doc in semantically_split_docs:
                            if len(doc.page_content) >= MIN_CHUNK_CONTENT_LENGTH:
                                all_final_chunks.append({
                                    "chunk_id": uuid.uuid4().hex,
                                    "chunk_content": doc.page_content,
                                    "chunk_metadata": doc.metadata # Renamed to chunk_metadata
                                })
                                chunk_type_counts[doc.metadata.get("type", "java_method_group_semantic")] = chunk_type_counts.get(doc.metadata.get("type", "java_method_group_semantic"), 0) + 1
                            else:
                                print(f"Skipping tiny semantic sub-chunk from {filepath_txt.name} (method group). Length: {len(doc.page_content)}")
                    else:
                        if len(current_method_group_content) >= MIN_CHUNK_CONTENT_LENGTH:
                            all_final_chunks.append({
                                "chunk_id": uuid.uuid4().hex,
                                "chunk_content": current_method_group_content.strip(), # Trim final whitespace
                                "chunk_metadata": combined_metadata # Renamed to chunk_metadata
                            })
                            chunk_type_counts[combined_metadata.get("type", "java_method_group")] = chunk_type_counts.get(combined_metadata.get("type", "java_method_group"), 0) + 1
                        else:
                            print(f"Skipping tiny method group chunk from {filepath_txt.name}. Length: {len(current_method_group_content)}")
                    
                    # Reset for the next group
                    current_method_group_content = ""
                    current_method_group_method_names = []


        elif inferred_type == 'config':
            parsed_config_chunks = parse_config_file(content, original_filename_base)
            
            # For config files, determine a proper filename metadata (e.g., .yml, .properties)
            config_meta_filename = original_filename_base
            is_yaml_content = "---" in content or any(line.strip().startswith(key) for key in ["server:", "spring:"] for line in content.splitlines() if line.strip())
            if is_yaml_content:
                config_meta_filename += ".yml"
            else:
                config_meta_filename += ".properties" # Default to .properties if not clearly YAML

            for config_chunk_data in parsed_config_chunks:
                config_chunk_content = config_chunk_data["content"]
                
                config_metadata = common_metadata_base.copy()
                config_metadata.update({
                    "type": f"config_{config_chunk_data['type']}",
                    "filename": config_meta_filename, # Override with inferred config filename
                    "file_type": os.path.splitext(config_meta_filename)[1].lstrip('.') or "config"
                })

                # Config chunks are usually small, but if a very large one occurs, split generically
                if generic_splitter and len(config_chunk_content) > GENERIC_CHUNK_SIZE:
                    sub_chunks = generic_splitter.split_text(config_chunk_content)
                    for i, sub_chunk_text in enumerate(sub_chunks):
                        if len(sub_chunk_text) >= MIN_CHUNK_CONTENT_LENGTH:
                            sub_chunk_metadata = config_metadata.copy() 
                            sub_chunk_metadata["type"] = f"{sub_chunk_metadata['type']}_sub_chunk"
                            sub_chunk_metadata["sub_chunk_index"] = i
                            all_final_chunks.append({
                                "chunk_id": uuid.uuid4().hex,
                                "chunk_content": sub_chunk_text,
                                "chunk_metadata": sub_chunk_metadata # Renamed to chunk_metadata
                            })
                            chunk_type_counts[sub_chunk_metadata.get("type", "config_sub_chunk")] = chunk_type_counts.get(sub_chunk_metadata.get("type", "config_sub_chunk"), 0) + 1
                        else:
                            print(f"Skipping tiny generic sub-chunk from {filepath_txt.name} (config). Length: {len(sub_chunk_text)}")
                else:
                    if len(config_chunk_content) >= MIN_CHUNK_CONTENT_LENGTH:
                        all_final_chunks.append({
                            "chunk_id": uuid.uuid4().hex, # Unique ID for each config chunk
                            "chunk_content": config_chunk_content,
                            "chunk_metadata": config_metadata # Renamed to chunk_metadata
                        })
                        chunk_type_counts[config_metadata.get("type", "config")] = chunk_type_counts.get(config_metadata.get("type", "config"), 0) + 1
                    else:
                        print(f"Skipping tiny config chunk from {filepath_txt.name}. Length: {len(config_chunk_content)}")
        else: # Unclassified / unknown files
            print(f"Generic chunking for unclassified file: {filepath_txt.name}")
            # If a file is unclassified, use generic splitter if available, otherwise treat as one chunk
            if generic_splitter:
                chunks = generic_splitter.split_text(content)
                for i, chunk_content in enumerate(chunks):
                    if len(chunk_content) >= MIN_CHUNK_CONTENT_LENGTH:
                        all_final_chunks.append({
                            "chunk_id": uuid.uuid4().hex, 
                            "chunk_content": chunk_content,
                            "chunk_metadata": { # Renamed to chunk_metadata
                                "type": "unclassified_text_chunk",
                                "filename": common_metadata_base["filename"],
                                "filepath_txt": common_metadata_base["filepath_txt"],
                                "sub_chunk_index": i
                            }
                        })
                        chunk_type_counts["unclassified_text_chunk"] = chunk_type_counts.get("unclassified_text_chunk", 0) + 1
                    else:
                        print(f"Skipping tiny generic chunk from {filepath_txt.name} (unclassified). Length: {len(chunk_content)}")
            else: # Fallback: if no generic splitter or content too small for splitting, treat as one chunk
                if len(content) >= MIN_CHUNK_CONTENT_LENGTH:
                    all_final_chunks.append({
                        "chunk_id": uuid.uuid4().hex,
                        "chunk_content": content,
                        "chunk_metadata": { # Renamed to chunk_metadata
                            "type": "unclassified_full_file",
                            "filename": common_metadata_base["filename"],
                            "filepath_txt": common_metadata_base["filepath_txt"]
                        }
                    })
                    chunk_type_counts["unclassified_full_file"] = chunk_type_counts.get("unclassified_full_file", 0) + 1
                else:
                    print(f"Skipping tiny unclassified full file chunk from {filepath_txt.name}. Length: {len(content)}")


    # Save all chunks to a single JSON file
    os.makedirs(CHUNKED_OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
    with open(output_json_path, 'w', encoding='utf-8') as out:
        json.dump(all_final_chunks, out, indent=2)

    print(f"\nTotal chunks generated: {len(all_final_chunks)} chunks saved to {output_json_path}")
    print("\n--- Chunk Type Distribution ---")
    for chunk_type, count in sorted(chunk_type_counts.items()):
        print(f"- {chunk_type}: {count}")
    print("-----------------------------\n")
    return all_final_chunks


# --- Usage (Main execution block) ---
if __name__ == '__main__':
    # Input directory is relative to TESTGEN_AUTOMATION_ROOT
    input_dir_path = PROCESSED_INPUT_DIR
    output_json_file_path = CHUNKED_OUTPUT_FILE

    # Run the processing
    processed_chunks = process_codebase_from_txt_for_chunking(input_dir_path, output_json_file_path)
