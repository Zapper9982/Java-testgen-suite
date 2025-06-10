import os
import re
# No yaml or json needed in this preprocessing step, as these are handled by the chunker

# Output directory remains constant
OUTPUT_DIR = './processed_output'

# Define file extensions to process.
# This list can be expanded or refined based on what "important stuff" means for your RAG pipeline.
FILE_EXTENSIONS_TO_PROCESS = (
    '.java',       # Java source code
    '.properties', # Spring Boot properties files
    '.yml',        # YAML configuration files
    '.yaml',       # YAML configuration files
    '.xml',        # XML configuration/Spring XML files
    '.sql',        # SQL scripts/schema definitions
    '.md',         # Markdown documentation
    '.js',         # JavaScript files (e.g., frontend components if relevant)
    '.ts',         # TypeScript files
    '.py',         # Python scripts
    '.html',       # HTML templates
    '.css'         # CSS stylesheets
)

def remove_comments(filepath: str, code_content: str) -> str:
    """
    Removes comments from the given code content based on the file extension.
    This function uses common regex patterns for different languages.
    For complex cases, more sophisticated parsing might be needed.
    """
    _, ext = os.path.splitext(filepath)

    if ext == '.java':
        # Remove block comments /* ... */
        code_content = re.sub(r'/\*.*?\*/', '', code_content, flags=re.DOTALL)
        # Remove single-line comments // ...
        code_content = re.sub(r'//.*', '', code_content)
    elif ext in ('.properties', '.yml', '.yaml', '.sql', '.md', '.py'):
        # Remove single-line comments starting with #
        code_content = re.sub(r'^\s*#.*', '', code_content, flags=re.MULTILINE)
    elif ext == '.xml':
        # Remove XML comments <!-- ... -->
        code_content = re.sub(r'<!--.*?-->', '', code_content, flags=re.DOTALL)
    elif ext in ('.js', '.ts', '.css'):
        # Remove JavaScript/TypeScript/CSS block comments /* ... */
        code_content = re.sub(r'/\*.*?\*/', '', code_content, flags=re.DOTALL)
        # Remove JavaScript/TypeScript single-line comments // ...
        if ext in ('.js', '.ts'):
            code_content = re.sub(r'//.*', '', code_content)
    elif ext == '.html':
        # Remove HTML comments <!-- ... -->
        code_content = re.sub(r'<!--.*?-->', '', code_content, flags=re.DOTALL)

    # After removing comments, clean up leading/trailing whitespace and remove empty lines
    # FIX: Changed 'line.strip()' to 'line_obj.strip()'
    cleaned_lines = [line_obj.strip() for line_obj in code_content.splitlines() if line_obj.strip()]
    return '\n'.join(cleaned_lines)

def process_and_save_file(full_path: str, relative_path: str, output_base_dir: str):
    """
    Reads a source file, removes comments, and saves its cleaned content
    as a .txt file in a mirrored directory structure within the output directory.
    The .txt extension is used consistently for the output,
    and the chunker script will infer the original file type from its content.
    """
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {full_path}: {e}")
        return

    # Remove comments based on the original file type
    cleaned_content = remove_comments(full_path, content)

    # Construct the output path for the .txt file
    # The output filename will always have a .txt extension.
    output_filename = os.path.splitext(relative_path)[0] + '.txt'
    output_path = os.path.join(output_base_dir, output_filename)

    # Ensure the output directory structure exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            # Ensure a newline at the end if content exists to prevent issues with concatenation
            out.write(cleaned_content + '\n')
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")

def walk_and_preprocess_codebase(source_directory: str, output_directory: str, extensions_to_process: tuple = FILE_EXTENSIONS_TO_PROCESS):
    """
    Walks through the specified source directory (including all subfolders)
    and processes files with the defined extensions.
    """
    print(f"Starting codebase preprocessing from: {source_directory}")
    print(f"Cleaned output will be saved to: {output_directory}")
    print(f"Looking for file types: {', '.join(extensions_to_process)}")

    processed_count = 0
    skipped_count = 0
    for root, _, files in os.walk(source_directory):
        for file in files:
            # Check if the file has one of the desired extensions
            if any(file.endswith(ext) for ext in extensions_to_process):
                full_path = os.path.join(root, file)
                # Get the path relative to the user-provided source directory
                # This ensures the output structure mirrors the input structure
                relative_path = os.path.relpath(full_path, source_directory)
                
                process_and_save_file(full_path, relative_path, output_directory)
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\nFinished preprocessing. {processed_count} relevant files were processed.")
    print(f"{skipped_count} files were skipped (not matching desired extensions).")

if __name__ == '__main__':
    # Prompt the user for the root directory of their codebase
    user_input_dir = input("Enter the path to the codebase source directory (e.g., 'my_spring_project/src/main'): ").strip()

    if not user_input_dir:
        print("Error: No source directory provided. Please enter a valid path.")
        exit(1)
    elif not os.path.isdir(user_input_dir):
        print(f"Error: The provided source directory '{user_input_dir}' does not exist or is not a directory.")
        exit(1)

    SOURCE_DIR_TO_USE = user_input_dir

    # Run the comprehensive preprocessing function
    walk_and_preprocess_codebase(SOURCE_DIR_TO_USE, OUTPUT_DIR)
    print(f'\nAll matching files processed. Cleaned content saved to: {OUTPUT_DIR}')