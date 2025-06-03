import os
import re
import yaml
import json # Import json for serializing complex metadata
from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Import the RecursiveCharacterTextSplitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    USE_LANGCHAIN_REC_SPLITTER = True
except ImportError:
    print("Warning: langchain RecursiveCharacterTextSplitter not found. Large semantic chunks will not be further split.")
    USE_LANGCHAIN_REC_SPLITTER = False

# Configuration for RecursiveCharacterTextSplitter
GENERIC_CHUNK_SIZE = 800
GENERIC_CHUNK_OVERLAP = 100

# --- File Loading Functions ---

def _infer_file_type_from_content(filename_base: str, content: str) -> str:
    """
    Infers the original file type (java, config, or unknown) based on content and filename hints.
    """
    # 1. Check for common config file names (even without full extension)
    # Added more robust checks for config file content signatures
    if (filename_base.lower() in ["application", "bootstrap"] and
        (content.strip().startswith("spring.") or content.strip().startswith("server."))):
        return 'config'
    # More general check for YAML or properties content
    if "---" in content or any(line.strip().startswith(key) for key in ["server:", "spring:", "database:", "management:"] for line in content.splitlines() if line.strip()):
        return 'config'
    if any(line.strip().startswith(key) and '=' in line for key in ["spring.", "server.", "my."] for line in content.splitlines() if line.strip() and not line.strip().startswith("#")):
        return 'config'


    # 2. Check for Java content
    # Look for common Java keywords and structures
    if (re.search(r"\b(package|import|class|interface|enum)\b", content) and
        re.search(r"\b(public|private|protected)\b", content) and
        re.search(r"\{|\}", content)):
        return 'java'

    return 'unknown'

def load_files_from_txt_directory(input_dir: str) -> List[Dict[str, str]]:
    """
    Loads all .txt files from a given directory and its subdirectories,
    and infers the original file type based on content.
    """
    loaded_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Get the filename without the .txt extension
                        original_filename_base = os.path.splitext(file)[0] # e.g., 'UserController' from 'UserController.txt'

                        inferred_type = _infer_file_type_from_content(original_filename_base, content)

                        loaded_files.append({
                            "filepath": path, # The path to the .txt file
                            "original_filename_base": original_filename_base, # e.g., 'UserController'
                            "content": content,
                            "inferred_type": inferred_type
                        })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return loaded_files

# --- Code Cleaning and Parsing Functions (remain mostly the same) ---
def remove_comments_and_clean_code(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)
    code = "\n".join([line.strip() for line in code.splitlines() if line.strip()])
    return code

def parse_java_code_with_metadata(code_content: str, filename_base: str) -> Dict[str, Any]:
    # filename_base will now be like 'UserController'
    parsed_data: Dict[str, Any] = {
        "filename": f"{filename_base}.java", # Re-add .java for better metadata context if it's Java
        "class_name": "UnknownClass",
        "class_annotations": [],
        "class_content": code_content,
        "methods": [],
        "fields": []
    }
    class_match = re.search(
        r"(?:^@\w+\s*)*\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*(class|interface|enum)\s+(\w+)",
        code_content, re.MULTILINE
    )
    if class_match:
        parsed_data["class_name"] = class_match.group(2)
        annotations_block = code_content[:class_match.start()]
        parsed_data["class_annotations"] = re.findall(r"@(\w+)(?:\([^)]*\))?", annotations_block)

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
        r"(?:^@\w+\s*)*\s*(public|private|protected|static|final|abstract|synchronized|native|strictfp)?\s*(?:<[\w,\s]+>)?\s*([\w\.<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w\.]+)?\s*\{",
        re.MULTILINE
    )
    method_matches = list(method_pattern.finditer(code_content))
    for i, match in enumerate(method_matches):
        method_start = match.start()
        method_signature_end = match.end()
        annotations_block_start = code_content.rfind('\n', 0, method_start) + 1
        method_annotations = re.findall(r"@(\w+)(?:\([^)]*\))?", code_content[annotations_block_start:method_start])

        brace_balance = 1
        method_end = method_signature_end
        while method_end < len(code_content) and brace_balance > 0:
            if code_content[method_end] == '{':
                brace_balance += 1
            elif code_content[method_end] == '}':
                brace_balance -= 1
            method_end += 1

        if method_end < len(code_content) and code_content[method_end-1] == '}':
            pass
        elif method_end < len(code_content) and code_content[method_end] == '}':
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
    # Attempt to infer original config extension for better parsing logic
    # Default to properties if no clear sign
    original_extension = ".properties"
    if "server:" in content or "spring:" in content and "---" in content:
        original_extension = ".yml" # Heuristic for YAML
    elif filename_base.lower() == "application" and "spring.datasource" in content:
         original_extension = ".properties" # Strong hint for properties

    filename_with_inferred_ext = f"{filename_base}{original_extension}"

    config_chunks = []
    if filename_with_inferred_ext.endswith(".properties"):
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                config_chunks.append({"type": "property", "content": line})
    elif filename_with_inferred_ext.endswith((".yml", ".yaml")):
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
            print(f"Error parsing YAML content from {filename_base}.txt: {e}")
            config_chunks.append({"type": "yaml_full", "content": content})
    return config_chunks

# --- Semantic Chunking Function ---
def semantic_chunk_with_metadata(texts: List[str], metadatas: List[Dict[str, Any]]) -> List[Document]:
    if len(texts) != len(metadatas):
        raise ValueError("Texts and metadatas lists must have the same length.")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    chunker = SemanticChunker(embeddings=embeddings) 

    # Pass texts and metadatas directly to create_documents
    semantically_chunked_docs = chunker.create_documents(texts, metadatas=metadatas)
    
    return semantically_chunked_docs

# --- Main Processing Function ---

def process_codebase_from_txt_for_chunking(input_dir: str, output_json_path: str) -> List[Dict[str, Any]]:
    """
    Processes codebase content read from .txt files, inferring their type,
    extracting rich chunks with metadata, and applies a generic character-based chunking.
    """
    all_final_chunks: List[Dict[str, Any]] = []

    generic_splitter = None
    if USE_LANGCHAIN_REC_SPLITTER:
        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=GENERIC_CHUNK_SIZE,
            chunk_overlap=GENERIC_CHUNK_OVERLAP,
            length_function=len,
        )

    loaded_txt_files = load_files_from_txt_directory(input_dir)

    for file_info in loaded_txt_files:
        filepath = file_info["filepath"] # Path to the .txt file
        original_filename_base = file_info["original_filename_base"] # e.g., 'UserController'
        content = file_info["content"]
        inferred_type = file_info["inferred_type"]

        if inferred_type == 'java':
            parsed_java_data = parse_java_code_with_metadata(content, original_filename_base)
            
            contents_for_semantic_chunking = []
            metadatas_for_semantic_chunking = []

            # Sanitize class_annotations and fields for ChromaDB metadata compatibility
            # Change `else None` to `else ""` for string-expected fields
            sanitized_class_annotations = ", ".join(parsed_java_data["class_annotations"]) if parsed_java_data["class_annotations"] else ""
            sanitized_fields = json.dumps(parsed_java_data["fields"]) if parsed_java_data["fields"] else "" # Serialize list of dicts to JSON string

            # Add class header (with a .java extension for filename in metadata)
            class_header_content = f"Class: {parsed_java_data['class_name']}\n"
            if sanitized_class_annotations:
                class_header_content += f"Annotations: {sanitized_class_annotations}\n"
            if sanitized_fields:
                class_header_content += "Fields:\n" + "\n".join([f"  @{f['annotation']} {f['type']} {f['name']}" for f in parsed_java_data["fields"]]) + "\n"
            
            class_header_metadata = {
                "type": "java_class_header",
                "filename": parsed_java_data['filename'], # Will be like 'UserController.java'
                "filepath_txt": filepath, # Path to the original .txt file
                "class_name": parsed_java_data["class_name"],
                "class_annotations": sanitized_class_annotations, # Now a string or ""
                "fields": sanitized_fields # Now a JSON string or ""
            }
            contents_for_semantic_chunking.append(class_header_content)
            metadatas_for_semantic_chunking.append(class_header_metadata)

            for method_data in parsed_java_data["methods"]:
                # Sanitize method_annotations
                sanitized_method_annotations = ", ".join(method_data["annotations"]) if method_data["annotations"] else ""

                method_content = method_data["content"]
                method_metadata = {
                    "type": "java_method",
                    "filename": parsed_java_data['filename'], # Will be like 'UserController.java'
                    "filepath_txt": filepath,
                    "class_name": parsed_java_data["class_name"],
                    "method_name": method_data["method_name"],
                    "return_type": method_data["return_type"],
                    "parameters": method_data["parameters"],
                    "method_annotations": sanitized_method_annotations, # Now a string or ""
                    "start_char": method_data["start_char"],
                    "end_char": method_data["end_char"]
                }
                contents_for_semantic_chunking.append(method_content)
                metadatas_for_semantic_chunking.append(method_metadata)
                
            semantically_chunked_docs = semantic_chunk_with_metadata(contents_for_semantic_chunking, metadatas_for_semantic_chunking)
            
            for doc in semantically_chunked_docs:
                if generic_splitter and len(doc.page_content) > GENERIC_CHUNK_SIZE:
                    sub_chunks = generic_splitter.split_text(doc.page_content)
                    for i, sub_chunk_text in enumerate(sub_chunks):
                        sub_chunk_metadata = doc.metadata.copy()
                        sub_chunk_metadata["original_type"] = sub_chunk_metadata["type"]
                        sub_chunk_metadata["type"] = f"{sub_chunk_metadata['type']}_sub_chunk"
                        sub_chunk_metadata["sub_chunk_index"] = i
                        all_final_chunks.append({
                            "filename": sub_chunk_metadata.get("filename", f"{original_filename_base}.java"),
                            "filepath_txt": sub_chunk_metadata.get("filepath_txt", filepath),
                            "chunk_content": sub_chunk_text,
                            "chunk_metadata": sub_chunk_metadata
                        })
                else:
                    all_final_chunks.append({
                        "filename": doc.metadata.get("filename", f"{original_filename_base}.java"),
                        "filepath_txt": doc.metadata.get("filepath_txt", filepath),
                        "chunk_content": doc.page_content,
                        "chunk_metadata": doc.metadata
                    })

        elif inferred_type == 'config':
            parsed_config_chunks = parse_config_file(content, original_filename_base)
            
            # For config files, we'll try to guess a proper extension for the 'filename' metadata
            # based on content and original_filename_base for better context.
            config_meta_filename = original_filename_base
            # Refined heuristic for YAML or properties
            if "server:" in content or "spring:" in content and "---" in content:
                config_meta_filename += ".yml"
            elif any(line.strip().startswith(key) and '=' in line for key in ["spring.", "server.", "my."] for line in content.splitlines() if line.strip() and not line.strip().startswith("#")):
                config_meta_filename += ".properties"
            else:
                config_meta_filename += ".txt" # Fallback if no clear config type

            for config_chunk in parsed_config_chunks:
                if generic_splitter and len(config_chunk["content"]) > GENERIC_CHUNK_SIZE:
                    sub_chunks = generic_splitter.split_text(config_chunk["content"])
                    for i, sub_chunk_text in enumerate(sub_chunks):
                        sub_chunk_metadata = {
                            "type": f"config_{config_chunk['type']}_sub_chunk",
                            "filename": config_meta_filename,
                            "filepath_txt": filepath,
                            "file_type": os.path.splitext(config_meta_filename)[1] or "config", # Get original extension if possible
                            "sub_chunk_index": i
                        }
                        all_final_chunks.append({
                            "filename": config_meta_filename,
                            "filepath_txt": filepath,
                            "chunk_content": sub_chunk_text,
                            "chunk_metadata": sub_chunk_metadata
                        })
                else:
                    all_final_chunks.append({
                        "filename": config_meta_filename,
                        "filepath_txt": filepath,
                        "chunk_content": config_chunk["content"],
                        "chunk_metadata": {
                            "type": f"config_{config_chunk['type']}",
                            "filename": config_meta_filename,
                            "filepath_txt": filepath,
                            "file_type": os.path.splitext(config_meta_filename)[1] or "config"
                        }
                    })
        else:
            print(f"Skipping or generic-chunking unknown file type from {filepath}: {original_filename_base}. No specific parsing applied.")
            # Apply generic text splitting for unclassified types
            if generic_splitter:
                chunks = generic_splitter.split_text(content)
                for i, chunk_content in enumerate(chunks):
                    all_final_chunks.append({
                        "filename": original_filename_base + ".txt", # Use .txt suffix for unknown files
                        "filepath_txt": filepath,
                        "chunk_content": chunk_content,
                        "chunk_metadata": {
                            "type": "unclassified_text_chunk",
                            "filename": original_filename_base + ".txt",
                            "filepath_txt": filepath,
                            "sub_chunk_index": i
                        }
                    })


    # Save all chunks to a single JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as out:
        json.dump(all_final_chunks, out, indent=2)

    print(f"Total chunks generated: {len(all_final_chunks)} chunks saved to {output_json_path}")
    return all_final_chunks


# --- Example Usage ---

input_directory_for_this_script = 'processed_output'
final_output_json_path = './chunked_output/intelligent_chunks_from_txt.json'

# --- Create dummy processed_output with .txt files *without* .java suffix for testing ---
# In your actual workflow, these would be generated by your first chunker.
os.makedirs(os.path.join(input_directory_for_this_script, "com", "example", "app"), exist_ok=True)
os.makedirs(os.path.join(input_directory_for_this_script, "config"), exist_ok=True)
os.makedirs(os.path.join(input_directory_for_this_script, "com", "iemr", "tm", "service", "schedule"), exist_ok=True) # Ensure this path exists

with open(os.path.join(input_directory_for_this_script, "com", "example", "app", "UserController.txt"), "w") as f:
    f.write("""
package com.example.app;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    /**
     * Retrieves a user by ID.
     * @param id The user ID.
     * @return The user object.
     */
    @GetMapping("/{id}")
    public User getUserById(@RequestParam Long id) {
        // Some logic here
        return userService.findUser(id);
    }

    @PostMapping("/")
    public User createUser(@RequestBody User user) {
        // Another logic block
        return userService.saveUser(user);
    }

    private void internalHelperMethod() {
        System.out.println("This is an internal helper.");
    }
}
""")

with open(os.path.join(input_directory_for_this_script, "com", "example", "app", "UserService.txt"), "w") as f:
    f.write("""
package com.example.app;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Transactional
    public User findUser(Long id) {
        // Complex business logic to find a user
        return userRepository.findById(id).orElse(null);
    }

    public User saveUser(User user) {
        // Logic to save a user
        return userRepository.save(user);
    }
}
""")

with open(os.path.join(input_directory_for_this_script, "config", "application.txt"), "w") as f: # Now just 'application.txt'
    f.write("""
spring.datasource.url=jdbc:h2:mem:testdb
spring.datasource.driverClassName=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=password
server.port=8080
# This is a comment
my.custom.property=someValue
""")

with open(os.path.join(input_directory_for_this_script, "config", "some-other-config.txt"), "w") as f:
    f.write("""
server:
    port: 8081
    servlet:
        context-path: /app
database:
    type: postgres
    host: localhost
    port: 5432
""")

with open(os.path.join(input_directory_for_this_script, "README.txt"), "w") as f:
    f.write("This is a readme file. It should be treated as unclassified text.")

# Add dummy content for SchedulingServiceImpl.txt to match your previous errors
with open(os.path.join(input_directory_for_this_script, "com", "iemr", "tm", "service", "schedule", "SchedulingServiceImpl.txt"), "w") as f:
    f.write("""
package com.iemr.tm.service.schedule;

public class SchedulingServiceImpl {
    public void scheduleTask1() {
        // Task 1 details
    }
    public void scheduleTask2() {
        // Task 2 details
    }
    public void scheduleTask3() {
        // Task 3 details
    }
    public void scheduleTask4() {
        // Task 4 details
    }
}
""")
# Add dummy content for application.properties.txt (if it was treated as a separate file)
# Assuming 'application.properties.txt' is an alias for 'config/application.txt' if you had it.
# If it's a separate file, create it. Otherwise, this part is redundant.
if not os.path.exists(os.path.join(input_directory_for_this_script, "src", "main", "resources", "application.properties.txt")):
    os.makedirs(os.path.join(input_directory_for_this_script, "src", "main", "resources"), exist_ok=True)
    with open(os.path.join(input_directory_for_this_script, "src", "main", "resources", "application.properties.txt"), "w") as f:
        f.write("some.property=value\nanother.property=another_value")


# Run the processing
processed_chunks = process_codebase_from_txt_for_chunking(input_directory_for_this_script, final_output_json_path)

# You can uncomment the following to print a few chunks
# for i, chunk in enumerate(processed_chunks[:10]): # Print first 10 chunks for review
#     print(f"\n--- Chunk {i+1} ---")
#     print(f"Filename: {chunk['filename']}")
#     print(f"Filepath (txt): {chunk['filepath_txt']}")
#     print(f"Metadata: {chunk['chunk_metadata']}")
#     print(f"Content:\n{chunk['chunk_content']}")
#     print("-" * 30)

# Optional: Clean up dummy files and directories after testing
# import shutil
# shutil.rmtree(input_directory_for_this_script)