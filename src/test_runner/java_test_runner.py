import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import os 
import re 

class JavaTestRunner:
    """
    Executes Java JUnit tests using Maven or Gradle and parses the test results.
    """
    def __init__(self, project_root: Path, build_tool: str = "maven"):
        self.project_root = project_root
        self.build_tool = build_tool.lower()

        if self.build_tool not in ["maven", "gradle"]:
            raise ValueError("Unsupported build tool. Please use 'maven' or 'gradle'.")

        print(f"Initialized JavaTestRunner with build tool: {self.build_tool} for project: {self.project_root}")

    def _parse_maven_errors(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parses Maven stdout/stderr for compilation errors and test failures, providing structured details and fix suggestions.
        """
        errors = {
            "compilation_errors": [],
            "test_failures": [],
            "general_messages": []
        }

        # Regex for compilation errors (e.g., [ERROR] /path/to/File.java:[line,col] error_message)
        compilation_error_pattern = re.compile(
            r"\[ERROR\]\s*(.+?\.java):\[(\d+),(\d+)\]\s*(.+)"
        )
        # Regex for cannot find symbol (extract symbol/class/method)
        cannot_find_symbol_pattern = re.compile(r"symbol:\s*(\w+)\s*(\w+)?")
        # Regex for missing import
        missing_import_pattern = re.compile(r"package\s+([\w\.]+)\s+does\s+not\s+exist")
        # Regex for incompatible types
        incompatible_types_pattern = re.compile(r"incompatible types: (.+) cannot be converted to (.+)")
        # Regex for more general compilation issues
        general_compilation_pattern = re.compile(
            r"\[ERROR\]\s*(.*(?:COMPILATION ERROR|error|symbol not found|package does not exist).*)", re.IGNORECASE
        )
        # Regex for assertion errors
        assertion_pattern = re.compile(r"expected:\s*(.+?)\s*but was:\s*(.+)", re.IGNORECASE)

        # Check for compilation errors first (from both stdout and stderr)
        for line in (stdout + stderr).splitlines():
            match_comp_error = compilation_error_pattern.search(line)
            if match_comp_error:
                file = match_comp_error.group(1)
                line_num = int(match_comp_error.group(2))
                column = int(match_comp_error.group(3))
                message = match_comp_error.group(4).strip()
                error_type = "COMPILATION_ERROR"
                symbol = None
                suggested_fix = None
                # Detect specific error types and suggest fixes
                if "cannot find symbol" in message:
                    error_type = "CANNOT_FIND_SYMBOL"
                    m = cannot_find_symbol_pattern.search(message)
                    if m:
                        symbol = m.group(2) or m.group(1)
                        suggested_fix = f"Declare or import symbol '{symbol}'. Check for typos or missing dependencies."
                    else:
                        suggested_fix = "Declare or import the missing symbol."
                elif "package" in message and "does not exist" in message:
                    error_type = "MISSING_IMPORT"
                    m = missing_import_pattern.search(message)
                    if m:
                        symbol = m.group(1)
                        suggested_fix = f"Add import statement for package '{symbol}'."
                    else:
                        suggested_fix = "Add the missing import statement."
                elif "incompatible types" in message:
                    error_type = "INCOMPATIBLE_TYPES"
                    m = incompatible_types_pattern.search(message)
                    if m:
                        from_type, to_type = m.groups()
                        suggested_fix = f"Change the type from '{from_type}' to '{to_type}' or cast appropriately."
                    else:
                        suggested_fix = "Fix the type mismatch."
                else:
                    suggested_fix = "Read the error message and fix the code accordingly."
                errors["compilation_errors"].append({
                    "file": file,
                    "line": line_num,
                    "column": column,
                    "location": f"{file}:{line_num}:{column}",
                    "message": message,
                    "error_type": error_type,
                    "symbol": symbol,
                    "suggested_fix": suggested_fix
                })
            else:
                match_general_comp = general_compilation_pattern.search(line)
                if match_general_comp:
                    if not any(match_general_comp.group(1) in e['message'] for e in errors["compilation_errors"]):
                        errors["general_messages"].append(match_general_comp.group(1).strip())

        # Test failure parsing (extract method, class, type, message, stack, expected/actual, suggested fix)
        test_failure_method_pattern = re.compile(r"\[ERROR\]\s+([\w.$]+)\(([\w.$]+)\)\s+Time elapsed:[\d.]+ s\s+<<< (?:FAILURE|ERROR)!?\s*\n(?P<exception_type>[a-zA-Z0-9_\.$]+(?:Exception|Error|Failure)?):?\s*(?P<message>.*?(?=\n\s+at|\n\s*\w+\.[\w.$]+:))", re.MULTILINE | re.DOTALL)
        for match in test_failure_method_pattern.finditer(stdout):
            test_method = match.group(1)
            test_class = match.group(2)
            failure_type = match.group('exception_type').strip()
            message = match.group('message').strip()
            expected = None
            actual = None
            suggested_fix = None
            stack_trace = None
            assertion_match = assertion_pattern.search(message)
            if assertion_match:
                expected = assertion_match.group(1).strip()
                actual = assertion_match.group(2).strip()
                suggested_fix = f"Check the assertion in '{test_method}'. Expected '{expected}', but got '{actual}'. Fix the test or the code under test."
            elif "NullPointerException" in failure_type:
                suggested_fix = f"A NullPointerException occurred in '{test_method}'. Ensure all mocks are properly stubbed and no object is null before use."
            elif "MockitoException" in failure_type:
                suggested_fix = f"A Mockito error occurred in '{test_method}'. Check your mock setup, especially for @Spy and argument matchers."
            elif "AssertionFailedError" in failure_type:
                suggested_fix = f"An assertion failed in '{test_method}'. Check the expected and actual values and update the test or code."
            else:
                suggested_fix = f"A test failure occurred in '{test_method}'. Read the error message and fix the test or code."
            # Try to extract stack trace (lines after the error message)
            stack_trace_lines = []
            lines = stdout.splitlines()
            start_idx = None
            for i, l in enumerate(lines):
                if test_method in l and failure_type in l:
                    start_idx = i + 1
                    break
            if start_idx is not None:
                for l in lines[start_idx:]:
                    if l.strip().startswith('at '):
                        stack_trace_lines.append(l.strip())
                    else:
                        break
                if stack_trace_lines:
                    stack_trace = '\n'.join(stack_trace_lines)
            errors["test_failures"].append({
                "test_class": test_class,
                "test_method": test_method,
                "failure_type": failure_type,
                "message": message,
                "expected": expected,
                "actual": actual,
                "stack_trace": stack_trace,
                "location": f"{test_class}.{test_method}",
                "suggested_fix": suggested_fix
            })
        # (General messages unchanged)
        if not errors["compilation_errors"] and not errors["test_failures"] and not errors["general_messages"] and ("BUILD FAILURE" in stdout or "BUILD ERROR" in stdout):
            errors["general_messages"].append("Maven build failed for an unparsed reason. Full output might be needed.")
        return errors


    def _run_maven_command(self, command_args: List[str]) -> Dict[str, Any]:
        """Internal helper to run Maven commands."""
        command = ["mvn"] + command_args
        print(f"Executing Maven command: {' '.join(command)} in {self.project_root}")
        
        try:
            process = subprocess.run(
                command,
                cwd=self.project_root, # Run Maven from the project root
                capture_output=True,
                text=True,
                check=False # Do not raise an exception for non-zero exit codes (we want to check stderr/stdout)
            )
            stdout = process.stdout
            stderr = process.stderr
            return_code = process.returncode

            detailed_errors = self._parse_maven_errors(stdout, stderr)

            status = "UNKNOWN"
            message = "An unexpected Maven execution state occurred."
            
            if return_code == 0 and "BUILD SUCCESS" in stdout:
                # Look for test results summary (common in Maven output)
                test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
                if test_summary_match:
                    total = int(test_summary_match.group(1))
                    failures = int(test_summary_match.group(2))
                    errors_count = int(test_summary_match.group(3))
                    skipped = int(test_summary_match.group(4))
                    
                    if failures == 0 and errors_count == 0:
                        status = "SUCCESS"
                        message = "Tests passed successfully."
                    else:
                        status = "FAILED"
                        message = f"Tests completed with {failures} failures, {errors_count} errors."
                    
                    summary = {"total": total, "failures": failures, "errors": errors_count, "skipped": skipped}
                else:
                    status = "SUCCESS" # Assume success if build success but no test summary (e.g., compile only)
                    message = "Maven build successful (no test summary found)." # More descriptive message
                    summary = None # No test summary available
            else: # Build failed or errored
                status = "FAILED"
                if detailed_errors["compilation_errors"]:
                    message = "Compilation errors detected."
                elif detailed_errors["test_failures"]:
                    message = "Test failures detected."
                elif detailed_errors["general_messages"]:
                    message = "General build errors detected."
                else:
                    message = f"Maven build failed with exit code {return_code} (no specific errors parsed)."
                summary = None # No successful test summary on failure

            return {
                "status": status,
                "message": message,
                "stdout": stdout,
                "stderr": stderr,
                "summary": summary,
                "detailed_errors": detailed_errors # NEW: Structured error details
            }

        except FileNotFoundError:
            return {
                "status": "ERROR",
                "message": "Maven (mvn) command not found. Please ensure Maven is installed and in your PATH.",
                "stdout": "",
                "stderr": "Maven not found.",
                "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": ["Maven command not found."]}
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"An unexpected error occurred during Maven execution: {e}",
                "stdout": "",
                "stderr": str(e),
                "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": [f"Unexpected error: {e}"]}
            }

    def _run_gradle_command(self, command_args: List[str]) -> Dict[str, Any]:
        """Internal helper to run Gradle commands."""
        command = ["gradle"] + command_args
        print(f"Executing Gradle command: {' '.join(command)} in {self.project_root}")

        try:
            process = subprocess.run(
                command,
                cwd=self.project_root, # Run Gradle from the project root
                capture_output=True,
                text=True,
                check=False
            )
            stdout = process.stdout
            stderr = process.stderr
            return_code = process.returncode

            # Simplified error parsing for Gradle for now, can be expanded later
            detailed_errors = {
                "compilation_errors": [],
                "test_failures": [],
                "general_messages": []
            }
            if "BUILD FAILED" in stdout or "BUILD FAILED" in stderr:
                detailed_errors["general_messages"].append("Gradle build failed. Check stdout/stderr for details.")
                # You can add regex here later to parse specific Gradle errors.
            
            status = "UNKNOWN"
            message = "An unexpected Gradle execution state occurred."

            if return_code == 0 and "BUILD SUCCESSFUL" in stdout:
                if "BUILD FAILED" in stdout or "FAILURE" in stdout: 
                     status = "FAILED"
                     message = "Gradle build successful but tests failed."
                else:
                    status = "SUCCESS"
                    message = "Tests passed successfully."
            else:
                status = "FAILED"
                message = f"Gradle build failed with exit code {return_code}."

            return {
                "status": status,
                "message": message,
                "stdout": stdout,
                "stderr": stderr,
                "summary": None, # Summary parsing for Gradle is more complex, omit for now
                "detailed_errors": detailed_errors
            }

        except FileNotFoundError:
            return {
                "status": "ERROR",
                "message": "Gradle (gradle) command not found. Please ensure Gradle is installed and in your PATH.",
                "stdout": "",
                "stderr": "Gradle not found.",
                "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": ["Gradle command not found."]}
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"An unexpected error occurred during Gradle execution: {e}",
                "stdout": "",
                "stderr": str(e),
                "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": [f"Unexpected error: {e}"]}
            }

    def run_test(self, test_file_path: Path) -> Dict[str, Any]:
        """
        Runs a specific Java JUnit test file.

        Args:
            test_file_path: The absolute Path to the generated Java test file.

        Returns:
            A dictionary containing test execution status, message, stdout, stderr,
            summary (if applicable), and detailed_errors.
        """
        if not test_file_path.exists():
            return {
                "status": "ERROR",
                "message": f"Test file not found: {test_file_path}",
                "stdout": "", "stderr": "", "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": [f"Test file not found: {test_file_path}"]}
            }

        print(f"\nRunning single test: {test_file_path}")

        if self.build_tool == "maven":
            try:
                # --- NEW FQN Extraction Logic ---
                # This aims to be more robust for extracting the fully qualified class name
                # E.g., for /Users/.../project_root/src/test/java/com/example/MyTest.java
                # It should extract "com.example.MyTest"
                path_parts = test_file_path.parts
                src_test_java_index = -1
                for i, part in enumerate(path_parts):
                    if part == "src" and i + 2 < len(path_parts) and path_parts[i+1] == "test" and path_parts[i+2] == "java":
                        src_test_java_index = i + 3 # Index right after 'java'
                        break
                
                if src_test_java_index != -1:
                    fqn_parts = path_parts[src_test_java_index:]
                    # Remove .java extension and join with dots
                    test_class_name = ".".join(fqn_parts).replace(".java", "")
                    print(f"DEBUG: Calculated FQN (test_class_name) for Maven: {test_class_name}") 
                else:
                    # Fallback to stem if 'src/test/java' structure not found (though it should be)
                    test_class_name = test_file_path.stem 
                    print(f"WARNING: Could not determine full class name for {test_file_path}, using simple name {test_class_name}. Path structure not as expected for FQN.")

            except Exception as e:
                test_class_name = test_file_path.stem 
                print(f"ERROR: Exception during FQN calculation for {test_file_path}: {e}. Using simple name {test_class_name}.")


            command_args = ["test", f"-Dtest={test_class_name}", "-DfailIfNoTests=false"]
            return self._run_maven_command(command_args)
        elif self.build_tool == "gradle":
            relative_test_path = test_file_path.relative_to(self.project_root / "src" / "test" / "java")
            test_class_name = str(relative_test_path.with_suffix('')).replace(os.sep, ".")
            command_args = ["test", "--tests", test_class_name]
            return self._run_gradle_command(command_args)
        else:
            return {
                "status": "ERROR",
                "message": "Invalid build tool configured.",
                "stdout": "", "stderr": "", "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": ["Invalid build tool."]}
            }

    def run_project_tests(self, is_full_verify: bool = False) -> Dict[str, Any]:
        """
        Runs all tests for the entire project.

        Args:
            is_full_verify: If True, runs 'clean verify' for Maven or 'clean test' for Gradle.
                            If False, runs 'test' (without clean) for Maven/Gradle.

        Returns:
            A dictionary containing overall test execution status, message, stdout, stderr,
            summary (if applicable), and detailed_errors.
        """
        print(f"\nRunning {'full project verification' if is_full_verify else 'all project tests'} for project: {self.project_root}")
        
        if self.build_tool == "maven":
            if is_full_verify:
                command_args = ["clean", "verify"]
            else:
                command_args = ["test"]
            return self._run_maven_command(command_args)
        elif self.build_tool == "gradle":
            if is_full_verify:
                command_args = ["clean", "test"]
            else:
                command_args = ["test"]
            return self._run_gradle_command(command_args)
        else:
            return {
                "status": "ERROR",
                "message": "Invalid build tool configured.",
                "stdout": "", "stderr": "", "summary": None,
                "detailed_errors": {"compilation_errors": [], "test_failures": [], "general_messages": ["Invalid build tool."]}
            }


if __name__ == "__main__":
    # Example Usage
    project_root_env = os.getenv("SPRING_BOOT_PROJECT_PATH")
    if not project_root_env:
        print("ERROR: SPRING_BOOT_PROJECT_PATH environment variable is not set for example usage.")
        exit(1)
    
    project_root_example = Path(project_root_env)
    
    # Example for single test execution (existing functionality)
    # Ensure this points to a real (or dummy) test file in your project
    example_test_file = project_root_example / "src" / "test" / "java" / "com" / "iemr" / "mmu" / "service" / "quickConsultation" / "QuickConsultationServiceImplTest.java" 
    
    # Create a dummy test file for testing the runner if it doesn't exist
    if not example_test_file.exists():
        example_test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(example_test_file, 'w') as f:
            f.write("""
package com.iemr.mmu.service.quickConsultation; // Ensure package matches path

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse; // Added for a potential failure example
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class QuickConsultationServiceImplTest { // Ensure this matches example_test_file name
    @Test
    void testDummySuccess() {
        assertTrue(true, "This test should always pass.");
    }

    // Uncomment this test to simulate a failure and see error parsing
    /*
    @Test
    void testDummyFailure() {
        assertFalse(true, "This test is intentionally designed to fail.");
    }
    */
}
""")
        print(f"Created a dummy test file for runner testing: {example_test_file}")
    
    runner = JavaTestRunner(project_root=project_root_example, build_tool="maven") # or "gradle"
    
    print("\n--- Running Single Test ---")
    single_test_results = runner.run_test(example_test_file)
    print(f"Status: {single_test_results['status']}")
    print(f"Message: {single_test_results['message']}")
    if single_test_results['stdout']:
        print("\n--- STDOUT ---")
        print(single_test_results['stdout'])
    if single_test_results['stderr']:
        print("\n--- STDERR ---")
        print(single_test_results['stderr'])
    if single_test_results['summary']:
        print(f"Summary: {single_test_results['summary']}")
    if single_test_results['detailed_errors']["compilation_errors"]:
        print("\n--- DETAILED COMPILATION ERRORS ---")
        for err in single_test_results['detailed_errors']['compilation_errors']:
            print(f"  File: {err['file']}, Line: {err['line']}, Col: {err['column']}, Msg: {err['message']}")
    if single_test_results['detailed_errors']["test_failures"]:
        print("\n--- DETAILED TEST FAILURES ---")
        for failure in single_test_results['detailed_errors']['test_failures']:
            print(f"  Summary: {failure['summary']}")
            for detail in failure['details']:
                print(f"    - {detail}")
    if single_test_results['detailed_errors']["general_messages"]:
        print("\n--- GENERAL ERROR MESSAGES ---")
        for msg in single_test_results['detailed_errors']['general_messages']:
            print(f"  - {msg}")


    print("\n--- Running Full Project Verification ---")
    full_verify_results = runner.run_project_tests(is_full_verify=True)
    print(f"Status: {full_verify_results['status']}")
    print(f"Message: {full_verify_results['message']}")
    if full_verify_results['stdout']:
        print("\n--- STDOUT ---")
        # print(full_verify_results['stdout']) # Suppressing full output for brevity for project
    if full_verify_results['stderr']:
        print("\n--- STDERR ---")
        # print(full_verify_results['stderr']) # Suppressing full output for brevity for project
    if full_verify_results['summary']:
        print(f"Summary: {full_verify_results['summary']}")
    if full_verify_results['detailed_errors']["compilation_errors"]:
        print("\n--- DETAILED COMPILATION ERRORS (Project) ---")
        for err in full_verify_results['detailed_errors']['compilation_errors']:
            print(f"  File: {err['file']}, Line: {err['line']}, Col: {err['column']}, Msg: {err['message']}")
    if full_verify_results['detailed_errors']["test_failures"]:
        print("\n--- DETAILED TEST FAILURES (Project) ---")
        for failure in full_verify_results['detailed_errors']['test_failures']:
            print(f"  Summary: {failure['summary']}")
            for detail in failure['details']:
                print(f"    - {detail}")
    if full_verify_results['detailed_errors']["general_messages"]:
        print("\n--- GENERAL ERROR MESSAGES (Project) ---")
        for msg in full_verify_results['detailed_errors']['general_messages']:
            print(f"  - {msg}")

