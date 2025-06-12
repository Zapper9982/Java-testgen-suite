import re
from pathlib import Path
from typing import List, Dict, Any, Union

def extract_custom_imports_from_chunk_file(processed_filepath_txt: Path) -> List[str]:

    custom_imports = set()
    import_pattern = re.compile(r"^\s*import\s+(com\.iemr\..*?);", re.MULTILINE)

    try:
        with open(processed_filepath_txt, 'r', encoding='utf-8') as f:
            content = f.read()
            for match in import_pattern.findall(content):
                custom_imports.add(f"import {match};")
    except FileNotFoundError:
        print(f"WARNING: Processed file not found: {processed_filepath_txt}")
    except Exception as e:
        print(f"ERROR: Could not read or parse imports from {processed_filepath_txt}: {e}")
    
    return sorted(list(custom_imports))

class CodeAnalyser:
    """
    Analyzes a Java source file to identify its class name, package,
    and internal project dependencies (other .java files it imports/uses).
    """
    def __init__(self, project_main_java_dir: Path):
        self.project_main_java_dir = project_main_java_dir
        self.internal_import_pattern = re.compile(r"import\s+(com\.iemr\.[a-zA-Z0-9\._]+);")
        self.package_pattern = re.compile(r"package\s+([a-zA-Z0-9\._]+);")
        self.class_pattern = re.compile(r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+([a-zA-Z0-9_]+)")

    def analyze_dependencies(self, java_file_path: Path) -> Dict[str, Union[str, List[str]]]:
        """
        Analyzes a given Java file to extract class name, package name,
        and infer its internal dependencies (filenames).
        """
        if not java_file_path.exists():
            print(f"ERROR: Java source file not found for analysis: {java_file_path}")
            return {"class_name": "N/A", "package_name": "N/A", "dependent_filenames": []}

        content = java_file_path.read_text(encoding='utf-8')
        
        class_name = "N/A"
        package_name = "N/A"
        dependent_filenames = set()

        package_match = self.package_pattern.search(content)
        if package_match:
            package_name = package_match.group(1)

        class_match = self.class_pattern.search(content)
        if class_match:
            class_name = class_match.group(1)

        for match in self.internal_import_pattern.finditer(content):
            full_import_path = match.group(1)
            dependent_class_name = full_import_path.split('.')[-1]
            dependent_filenames.add(f"{dependent_class_name}.java")
        
        return {
            "class_name": class_name,
            "package_name": package_name,
            "dependent_filenames": sorted(list(dependent_filenames))
        }

class SpringBootAnalyser:

    def __init__(self, project_main_java_dir: Path, processed_output_root: Path):
        self.project_main_java_dir = project_main_java_dir
        self.processed_output_root = processed_output_root
        self.service_annotation_pattern = re.compile(r"@Service")
        self.controller_annotation_pattern = re.compile(r"@(?:Rest)?Controller")
        self.code_analyser = CodeAnalyser(project_main_java_dir)

    def discover_targets(self) -> List[Dict[str, str]]:

        discovered_targets = []
        print(f"\nScanning Spring Boot project for Services and Controllers in: {self.project_main_java_dir}")

        for java_file_path in self.project_main_java_dir.rglob("*.java"):
            try:
                content = java_file_path.read_text(encoding='utf-8')
                is_service = self.service_annotation_pattern.search(content)
                is_controller = self.controller_annotation_pattern.search(content)

                if is_service or is_controller:
                    print(f"  Found Spring Boot target: {java_file_path.name}")
                    
                    analysis_result = self.code_analyser.analyze_dependencies(java_file_path)
                    class_name = analysis_result["class_name"]
                    package_name = analysis_result["package_name"]
                    dependent_filenames = analysis_result["dependent_filenames"]

                    relative_path_from_main_java = java_file_path.relative_to(self.project_main_java_dir)
                    relative_processed_txt_path = relative_path_from_main_java.parent / (relative_path_from_main_java.stem + ".txt")
                    
                    absolute_processed_txt_path = self.processed_output_root / relative_processed_txt_path
                    custom_imports = extract_custom_imports_from_chunk_file(absolute_processed_txt_path)


                    discovered_targets.append({
                        "java_file_path_abs": java_file_path,
                        "relative_processed_txt_path": str(relative_processed_txt_path),
                        "class_name": class_name,
                        "package_name": package_name,
                        "dependent_filenames": dependent_filenames,
                        "custom_imports": custom_imports
                    })
            except Exception as e:
                print(f"  Error processing file {java_file_path}: {e}")
        
        print(f"Finished scanning. Discovered {len(discovered_targets)} Spring Boot targets.")
        return discovered_targets

