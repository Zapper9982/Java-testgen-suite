import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

#config 
SPRING_BOOT_PROJECT_ROOT_ENV = os.getenv("SPRING_BOOT_PROJECT_PATH")
if not SPRING_BOOT_PROJECT_ROOT_ENV:
    print("ERROR: SPRING_BOOT_PROJECT_PATH environment variable is not set.")
    print("Please set it in your run.sh script (e.g., 'export SPRING_BOOT_PROJECT_PATH=\"/path/to/your/project\"') before running this script.")
    exit(1)

CODEBASE_SOURCE_DIR = Path(SPRING_BOOT_PROJECT_ROOT_ENV) / "src" / "main" / "java"

PROCESSED_OUTPUT_DIR = Path("./processed_output")

INCLUDED_FILE_EXTENSIONS = [
    ".java",       
    ".properties", 
    ".yml",       
    ".yaml",       
    ".xml",        
    ".sql",              
]
EXCLUDED_FILE_NAMES = [
    "pom.xml",
    "README.md",
    "LICENSE",
    ".gitignore"
]

def clean_content(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = [line for line in lines if line.strip()]
    return "".join(cleaned_lines)

def process_and_save_file(full_path: Path, relative_path: Path, output_base_dir: Path):
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {full_path}: {e}")
        return

    cleaned_content = remove_comments(str(full_path), content) 
    output_filename = relative_path.with_suffix('.txt')
    output_path = output_base_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(cleaned_content + '\n')
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")

def remove_comments(filepath: str, code_content: str) -> str:

    _, ext = os.path.splitext(filepath)

    if ext == '.java':

        code_content = re.sub(r'/\*.*?\*/', '', code_content, flags=re.DOTALL)

        code_content = re.sub(r'//.*', '', code_content)
    elif ext in ('.properties', '.yml', '.yaml', '.sql', '.md', '.py'):

        code_content = re.sub(r'^\s*#.*', '', code_content, flags=re.MULTILINE)
    elif ext == '.xml':
      
        code_content = re.sub(r'<!--.*?-->', '', code_content, flags=re.DOTALL)

    cleaned_lines = [line_obj.strip() for line_obj in code_content.splitlines() if line_obj.strip()]
    return '\n'.join(cleaned_lines)

def walk_and_preprocess_codebase(source_directory: Path, output_directory: Path, extensions_to_process: list = INCLUDED_FILE_EXTENSIONS):

    if not source_directory.exists() or not source_directory.is_dir():
        print(f"Error: Codebase source directory not found: {source_directory}")
        exit(1) 

    if output_directory.exists():
        print(f"Clearing previous processed output directory: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Starting codebase preprocessing from: {source_directory}")
    print(f"Cleaned output will be saved to: {output_directory}")
    print(f"Looking for file types: {', '.join(extensions_to_process)}")

    processed_count = 0
    skipped_count = 0

    for file_path in source_directory.rglob("*"):
        if file_path.is_file():

            if file_path.suffix in extensions_to_process and file_path.name not in EXCLUDED_FILE_NAMES:
                relative_path = file_path.relative_to(source_directory)
                
                process_and_save_file(file_path, relative_path, output_directory)
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\nFinished preprocessing. {processed_count} relevant files were processed.")
    print(f"{skipped_count} files were skipped (not matching desired extensions or excluded names).")
    print(f"\nAll matching files processed. Cleaned content saved to: {output_directory}")

if __name__ == '__main__':
    walk_and_preprocess_codebase(CODEBASE_SOURCE_DIR, PROCESSED_OUTPUT_DIR, INCLUDED_FILE_EXTENSIONS)

