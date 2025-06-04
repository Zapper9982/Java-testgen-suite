import os
import re

# Output directory remains constant
OUTPUT_DIR = './processed_output'

def is_controller_or_service(content):
    return '@RestController' in content or '@Service' in content

def remove_comments(code):
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code

def process_java_file(full_path, relative_path, output_base_dir):
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {full_path}: {e}")
        return

    if not is_controller_or_service(content):
        return

    cleaned = remove_comments(content)
    lines = cleaned.strip().split('\n')

    # Construct the output path
    # Replace .java extension with .txt
    output_filename = os.path.splitext(relative_path)[0] + '.txt'
    output_path = os.path.join(output_base_dir, output_filename)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            for line in lines:
                stripped = line.strip()
                if stripped: # Write non-empty, stripped lines
                    out.write(stripped + '\n')
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")

def walk_and_process(source_directory: str, output_directory: str):

    print(f"Starting processing from: {source_directory}")
    print(f"Output will be saved to: {output_directory}")

    processed_count = 0
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.java'):
                full_path = os.path.join(root, file)
                # Get the path relative to the user-provided source directory
                relative_path = os.path.relpath(full_path, source_directory)
                process_java_file(full_path, relative_path, output_directory)
                processed_count += 1
    print(f"Finished processing. {processed_count} Java files were checked.")


if __name__ == '__main__':

    user_input_dir = input("Enter the path to the Java source directory: ").strip()

    if not user_input_dir:
        print("Error: No source directory provided. Please enter a valid path.")
        exit(1)
    elif not os.path.isdir(user_input_dir):
        print(f"Error: The provided source directory '{user_input_dir}' does not exist or is not a directory.")
        exit(1)

    SOURCE_DIR_TO_USE = user_input_dir

    walk_and_process(SOURCE_DIR_TO_USE, OUTPUT_DIR)
    print(f'All matching files processed. Output saved to: {OUTPUT_DIR}')