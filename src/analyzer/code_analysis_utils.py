import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# This file (code_analysis_utils.py) is a utility module within the 'analyzer' package.
# It does not define root paths; those are passed from the calling scripts.

def extract_custom_imports_from_chunk_file(processed_filepath_txt: Path) -> List[str]:
    """
    Reads a processed_output .txt file and extracts unique 'com.iemr.' import statements.

    Args:
        processed_filepath_txt: The absolute Path object to the processed .txt file.

    Returns:
        A list of unique import statements starting with 'com.iemr.'.
    """
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

def extract_all_imports_from_java_file(java_file_path: Path) -> List[str]:
    """
    Extracts all import statements (project and external) from a Java file.
    Returns a list of import strings (e.g., 'import org.slf4j.Logger;').
    """
    all_imports = set()
    import_pattern = re.compile(r'^\s*import\s+([\w\.\*]+);', re.MULTILINE)
    try:
        with open(java_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for match in import_pattern.findall(content):
                all_imports.add(f"import {match};")
    except FileNotFoundError:
        print(f"WARNING: Java file not found: {java_file_path}")
    except Exception as e:
        print(f"ERROR: Could not read or parse imports from {java_file_path}: {e}")
    return sorted(list(all_imports))

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
        and infer its internal dependencies (filenames), and all imports.
        """
        if not java_file_path.exists():
            print(f"ERROR: Java source file not found for analysis: {java_file_path}")
            return {"class_name": "N/A", "package_name": "N/A", "dependent_filenames": [], "all_imports": []}

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
        
        all_imports = extract_all_imports_from_java_file(java_file_path)

        return {
            "class_name": class_name,
            "package_name": package_name,
            "dependent_filenames": sorted(list(dependent_filenames)),
            "all_imports": all_imports
        }

    def _detect_db_dependency_in_content(self, java_code: str, class_name: str) -> bool:
        """
        Detects if a Java class likely requires a test database for integration testing.
        Checks for common indicators like @Repository, @Transactional, and Spring Data JPA dependencies.
        """
        # 1. Check for @Repository annotation on the class itself
        if re.search(r"@Repository\s*(?:public\s+)?(?:class|interface|enum)\s+" + re.escape(class_name), java_code):
            return True

        # 2. Check for @Transactional annotation on class or methods
        if re.search(r"@Transactional", java_code):
            return True

        # 3. Check for injection of Spring Data JPA Repositories or similar persistence interfaces
        repository_injection_pattern = re.compile(
            r"@(?:Autowired|Inject)\s+private\s+(?:final\s+)?([a-zA-Z0-9_]+(?:Repository|Dao|Mapper))\s+\w+;",
            re.MULTILINE
        )
        if repository_injection_pattern.search(java_code):
            return True
        
        # 4. Check for direct usage of EntityManager or JdbcTemplate
        if re.search(r"\bEntityManager\b|\bJdbcTemplate\b", java_code):
            return True

        return False


class SpringBootAnalyser:
    """
    Scans a Spring Boot project's 'src/main/java' directory to discover
    Service and Controller classes and extracts their metadata.
    """
    def __init__(self, project_main_java_dir: Path, processed_output_root: Path):
        self.project_main_java_dir = project_main_java_dir
        self.processed_output_root = processed_output_root
        self.service_annotation_pattern = re.compile(r"@Service")
        self.controller_annotation_pattern = re.compile(r"@(?:Rest)?Controller")
        self.code_analyser = CodeAnalyser(project_main_java_dir) # Reuse CodeAnalyser's logic

    def discover_targets(self) -> List[Dict[str, str]]:
        """
        Discovers Spring Boot Service and Controller classes and
        extracts their file name, location, and internal imports.
        Only includes targets explicitly identified as Service or Controller.
        """
        discovered_targets = []
        print(f"\nScanning Spring Boot project for Services and Controllers in: {self.project_main_java_dir}")

        for java_file_path in self.project_main_java_dir.rglob("*.java"):
            try:
                # First, determine the path to the *processed* .txt file
                relative_path_from_main_java = java_file_path.relative_to(self.project_main_java_dir)
                relative_processed_txt_path = relative_path_from_main_java.parent / (relative_path_from_main_java.stem + ".txt")
                absolute_processed_txt_path = self.processed_output_root / relative_processed_txt_path
                
                if not absolute_processed_txt_path.exists():
                    print(f"  WARNING: Processed .txt file not found for {java_file_path.name}. Skipping.")
                    continue

                # Read the content from the *processed* .txt file for annotation detection
                # This ensures comments are removed before checking for @Service/@Controller
                cleaned_content = absolute_processed_txt_path.read_text(encoding='utf-8')
                is_service = self.service_annotation_pattern.search(cleaned_content)
                is_controller = self.controller_annotation_pattern.search(cleaned_content)

                # Filter here: Only process if it's explicitly a Service or Controller based on cleaned content
                if is_service or is_controller:
                    
                    analysis_result = self.code_analyser.analyze_dependencies(java_file_path)
                    class_name = analysis_result["class_name"]
                    package_name = analysis_result["package_name"]
                    dependent_filenames = analysis_result["dependent_filenames"]
                    all_imports = analysis_result.get("all_imports", [])

                    custom_imports = extract_custom_imports_from_chunk_file(absolute_processed_txt_path)

                    # Determine the type
                    target_type = "Controller" if is_controller else "Service"
                    
                    # Detect DB dependency using the helper function on original content (for richer context)
                    # We use java_file_path.read_text() here to get the full context for DB detection regexes
                    # which might look for specific injection patterns etc.
                    original_java_content = java_file_path.read_text(encoding='utf-8')
                    requires_db_test = self.code_analyser._detect_db_dependency_in_content(original_java_content, class_name)


                    print(f"  Found {target_type} target: {java_file_path.name}") # Explicitly print type
                    discovered_targets.append({
                        "java_file_path_abs": str(java_file_path), # Convert Path to string for JSON serialization
                        "relative_processed_txt_path": str(relative_processed_txt_path),
                        "class_name": class_name,
                        "package_name": package_name,
                        "dependent_filenames": dependent_filenames,
                        "custom_imports": custom_imports,
                        "all_imports": all_imports,
                        "type": target_type, # Add the determined type
                        "requires_db_test": requires_db_test # Add the DB dependency flag
                    })
            except Exception as e:
                print(f"  Error processing file {java_file_path}: {e}")
        
        print(f"Finished scanning. Discovered {len(discovered_targets)} Spring Boot targets (Services/Controllers).")
        return discovered_targets

def resolve_transitive_dependencies(java_file_path: Path, project_main_java_dir: Path) -> List[str]:
    """
    Recursively resolve all transitive dependencies (filenames) for a given Java file.
    Returns a unique list of all .java filenames (including the target file itself).
    """
    visited = set()
    to_visit = [java_file_path]
    all_filenames = set()
    analyser = CodeAnalyser(project_main_java_dir)

    while to_visit:
        current_path = to_visit.pop()
        if not current_path.exists() or current_path in visited:
            continue
        visited.add(current_path)
        analysis = analyser.analyze_dependencies(current_path)
        filename = current_path.name
        all_filenames.add(filename)
        for dep_filename in analysis.get('dependent_filenames', []):
            dep_path = None
            # Try to find the dependency file in the project
            for found in project_main_java_dir.rglob(dep_filename):
                dep_path = found
                break
            if dep_path and dep_path not in visited:
                to_visit.append(dep_path)
    return sorted(all_filenames)


def resolve_dependency_path(dep_filename, main_class_code, project_root):
    import re, os
    dep_basename = dep_filename.replace('.java', '')
    # Try to find the import statement for this dependency
    pattern = re.compile(r'import\s+([\w\.]+)\.' + re.escape(dep_basename) + r';')
    match = pattern.search(main_class_code)
    package_path = None
    if match:
        package_path = match.group(1).replace('.', os.sep)
        candidate = os.path.join(project_root, package_path, dep_filename)
        if os.path.exists(candidate):
            return candidate
    # Fallback: recursive search for the file in the project tree
    found = None
    for root, dirs, files in os.walk(project_root):
        if dep_filename in files:
            full_path = os.path.join(root, dep_filename)
            # Prefer a file whose path includes the expected package
            if package_path and package_path in root:
                return full_path
            if not found:
                found = full_path
    return found

