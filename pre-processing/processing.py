import os
import re


SOURCE_DIR = '/Users/tanmay/Desktop/C4GT/Scheduler-API/src/main/java'

OUTPUT_DIR = './processed_output'

def is_controller_or_service(content):
    return '@RestController' in content or '@Service' in content

def remove_comments(code):
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code

def process_java_file(full_path, relative_path):
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not is_controller_or_service(content):
        return

    cleaned = remove_comments(content)
    lines = cleaned.strip().split('\n')

    output_path = os.path.join(OUTPUT_DIR, relative_path)
    output_path = os.path.splitext(output_path)[0] + '.txt'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out:
        for line in lines:
            stripped = line.strip()
            if stripped:
                out.write(stripped + '\n')

def walk_and_process():
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.java'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, SOURCE_DIR)
                process_java_file(full_path, relative_path)

if __name__ == '__main__':
    walk_and_process()
    print(f'All matching files processed. Output saved to: {OUTPUT_DIR}')
