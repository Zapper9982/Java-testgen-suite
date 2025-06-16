import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import os
class JavaTestRunner:

    def __init__(self, project_root: Path, build_tool: str = "maven"):
        self.project_root = project_root
        self.build_tool = build_tool.lower()

        if self.build_tool not in ["maven", "gradle"]:
            raise ValueError("Unsupported build tool. Please use 'maven' or 'gradle'.")

        print(f"Initialized JavaTestRunner with build tool: {self.build_tool} for project: {self.project_root}")

    def _run_maven_test(self, test_file_path: Path) -> Dict[str, Any]:
        relative_test_path = test_file_path.relative_to(self.project_root)
        test_class_name = str(relative_test_path).replace(".java", "").replace(os.sep, ".")

        command = [
            "mvn",
            "test",
            f"-Dtest={test_class_name}",
            "-DfailIfNoTests=false" 
        ]
        
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

            # Simple parsing for success/failure (can be improved)
            if return_code == 0 and "BUILD SUCCESS" in stdout:
                # Look for test results summary
                if "Tests run: " in stdout:
                    test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
                    if test_summary_match:
                        total = int(test_summary_match.group(1))
                        failures = int(test_summary_match.group(2))
                        errors = int(test_summary_match.group(3))
                        skipped = int(test_summary_match.group(4))
                        
                        if failures == 0 and errors == 0:
                            return {"status": "SUCCESS", "message": "Tests passed successfully.", "stdout": stdout, "stderr": stderr}
                        else:
                            return {"status": "FAILED", "message": "Tests failed or had errors.", "stdout": stdout, "stderr": stderr}
                else:
                    return {"status": "UNKNOWN", "message": "Maven build successful but no test summary found.", "stdout": stdout, "stderr": stderr}
            else:
                return {"status": "ERROR", "message": f"Maven build failed with exit code {return_code}.", "stdout": stdout, "stderr": stderr}

        except FileNotFoundError:
            return {"status": "ERROR", "message": "Maven (mvn) command not found. Please ensure Maven is installed and in your PATH.", "stdout": "", "stderr": "Maven not found."}
        except Exception as e:
            return {"status": "ERROR", "message": f"An unexpected error occurred during Maven execution: {e}", "stdout": "", "stderr": str(e)}

 

    def run_test(self, test_file_path: Path) -> Dict[str, Any]:

        if not test_file_path.exists():
            return {"status": "ERROR", "message": f"Test file not found: {test_file_path}", "stdout": "", "stderr": ""}

        print(f"\nRunning test: {test_file_path}")

        if self.build_tool == "maven":
            return self._run_maven_test(test_file_path)
        elif self.build_tool == "gradle":
            return self._run_gradle_test(test_file_path)
        else:
            return {"status": "ERROR", "message": "Invalid build tool configured.", "stdout": "", "stderr": ""}

if __name__ == "__main__":

    project_root = Path("/Users/tanmay/Desktop/AMRIT/BeneficiaryID-Generation-API") 
    example_test_file = project_root / "src" / "test" / "java" / "com" / "iemr" / "common" / "bengen" / "service" / "GenerateBeneficiaryServiceTest.java" # Replace with a real test file



    runner = JavaTestRunner(project_root=project_root, build_tool="maven") # or "gradle"
    results = runner.run_test(example_test_file)

    print("\n--- Test Results ---")
    print(f"Status: {results['status']}")
    print(f"Message: {results['message']}")
    if results['stdout']:
        print("\n--- STDOUT ---")
        print(results['stdout'])
    if results['stderr']:
        print("\n--- STDERR ---")
        print(results['stderr'])
