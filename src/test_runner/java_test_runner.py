import subprocess
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import os # For env vars if used in main example
import re # For parsing test output from single test run

# Setup basic logging if no handlers are configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JavaTestRunner:
    def __init__(self, project_root: Path, build_tool: str = "maven"):
        self.project_root = project_root
        self.build_tool = build_tool.lower()
        self.logger = logging.getLogger(__name__) # Use a named logger

        if self.build_tool == "maven":
            self.jacoco_report_path = self.project_root / "target" / "site" / "jacoco" / "jacoco.xml"
            # `install` is more likely to trigger all phases including JaCoCo report generation
            # `verify` might also work if the lifecycle is configured correctly.
            self.build_command = ["mvn", "clean", "install"]
            self.single_test_command_prefix = ["mvn", "test"] # `-Dtest=...` will be added
        elif self.build_tool == "gradle":
            self.jacoco_report_path = self.project_root / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"
            # `build` usually includes `test` and `jacocoTestReport` if configured.
            # `jacocoTestReport` explicitly runs the JaCoCo report task.
            self.build_command = ["./gradlew", "clean", "build", "jacocoTestReport"]
            self.single_test_command_prefix = ["./gradlew", "test"] # `--tests ...` will be added
        else:
            raise ValueError(f"Unsupported build tool: {self.build_tool}. Supported: 'maven', 'gradle'.")

        self.logger.info(f"Initialized JavaTestRunner with build tool: {self.build_tool} for project: {self.project_root}")
        self.logger.info(f"JaCoCo report path set to: {self.jacoco_report_path}")
        self.logger.info(f"Build command set to: {' '.join(self.build_command)}")


    def _run_maven_single_test(self, test_file_path: Path) -> Dict[str, Any]:
        relative_test_path = test_file_path.relative_to(self.project_root / "src" / "test" / "java")
        test_class_name = str(relative_test_path).replace(".java", "").replace(os.sep, ".")
        
        command = self.single_test_command_prefix + [f"-Dtest={test_class_name}", "-DfailIfNoTests=false"]

        self.logger.info(f"Executing Maven single test command: {' '.join(command)} in {self.project_root}")
        
        try:
            process = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            stdout = process.stdout
            stderr = process.stderr
            return_code = process.returncode

            if return_code == 0 and "BUILD SUCCESS" in stdout:
                if "Tests run: " in stdout:
                    test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
                    if test_summary_match:
                        failures = int(test_summary_match.group(2))
                        errors = int(test_summary_match.group(3))
                        if failures == 0 and errors == 0:
                            return {"status": "SUCCESS", "message": "Test passed.", "stdout": stdout, "stderr": stderr}
                        else:
                            return {"status": "FAILED", "message": "Test failed or had errors.", "stdout": stdout, "stderr": stderr}
                return {"status": "UNKNOWN", "message": "Maven build successful but no test summary found.", "stdout": stdout, "stderr": stderr}
            else:
                return {"status": "ERROR", "message": f"Maven test execution failed with exit code {return_code}.", "stdout": stdout, "stderr": stderr}
        except FileNotFoundError:
            return {"status": "ERROR", "message": "Maven (mvn) command not found.", "stdout": "", "stderr": "Maven not found."}
        except Exception as e:
            self.logger.error(f"Unexpected error during Maven single test execution: {e}", exc_info=True)
            return {"status": "ERROR", "message": f"An unexpected error occurred: {e}", "stdout": "", "stderr": str(e)}

    def _run_gradle_single_test(self, test_file_path: Path) -> Dict[str, Any]:
        relative_test_path = test_file_path.relative_to(self.project_root / "src" / "test" / "java")
        test_class_name = str(relative_test_path).replace(".java", "").replace(os.sep, ".")

        command = self.single_test_command_prefix + [f"--tests", test_class_name]

        self.logger.info(f"Executing Gradle single test command: {' '.join(command)} in {self.project_root}")

        try:
            process = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            stdout = process.stdout
            stderr = process.stderr
            return_code = process.returncode

            if return_code == 0 and "BUILD SUCCESSFUL" in stdout: # Gradle success message
                # Gradle's output for test results is less standardized in basic console output than Maven's.
                # Often, a separate HTML report is the primary source of detailed results.
                # For simplicity, we'll assume if the build is successful and tests were run, it's a pass.
                # More robust parsing would require looking for "X tests completed, Y failed" patterns.
                if f"> Task :test" in stdout and "FAILED" not in stdout: # Basic check
                     return {"status": "SUCCESS", "message": "Test passed.", "stdout": stdout, "stderr": stderr}
                elif "FAILED" in stdout: # If FAILED appears anywhere
                     return {"status": "FAILED", "message": "Test failed or had errors based on 'FAILED' in output.", "stdout": stdout, "stderr": stderr}
                else: # If no clear failure, but also no clear success signal for tests.
                     return {"status": "UNKNOWN", "message": "Gradle build successful, but test pass/fail status unclear from stdout.", "stdout": stdout, "stderr": stderr}

            elif "NO-SOURCE" in stdout or "No tests found for given includes" in stdout : # Gradle specific for no tests
                return {"status": "SUCCESS", "message": "No tests found or executed, but build did not fail.", "stdout": stdout, "stderr": stderr}

            else: # Build failed
                return {"status": "ERROR", "message": f"Gradle test execution failed with exit code {return_code}.", "stdout": stdout, "stderr": stderr}
        except FileNotFoundError:
            return {"status": "ERROR", "message": "Gradle (./gradlew) command not found.", "stdout": "", "stderr": "Gradle not found."}
        except Exception as e:
            self.logger.error(f"Unexpected error during Gradle single test execution: {e}", exc_info=True)
            return {"status": "ERROR", "message": f"An unexpected error occurred: {e}", "stdout": "", "stderr": str(e)}


    def run_test(self, test_file_path: Path) -> Dict[str, Any]:
        """
        Runs a single test file using the configured build tool.
        This is used for quick feedback by TestCaseGenerator.
        """
        if not test_file_path.exists():
            return {"status": "ERROR", "message": f"Test file not found: {test_file_path}", "stdout": "", "stderr": ""}

        self.logger.info(f"Running single test: {test_file_path} using {self.build_tool}")
        if self.build_tool == "maven":
            return self._run_maven_single_test(test_file_path)
        elif self.build_tool == "gradle":
            return self._run_gradle_single_test(test_file_path)
        else: # Should have been caught in __init__
            return {"status": "ERROR", "message": "Invalid build tool configured.", "stdout": "", "stderr": ""}


    def run_build_and_generate_jacoco_report(self) -> bool:
        self.logger.info(f"Running full build with command: {' '.join(self.build_command)} in {self.project_root}")
        try:
            process = subprocess.run(
                self.build_command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            # Log stdout/stderr regardless of success for debugging
            if process.stdout:
                self.logger.info(f"Build process stdout:\n{process.stdout}")
            if process.stderr:
                self.logger.warning(f"Build process stderr:\n{process.stderr}")

            if process.returncode == 0:
                self.logger.info(f"Build command completed successfully.")
                return True
            else:
                self.logger.error(f"Build command failed with return code {process.returncode}.")
                return False
        except FileNotFoundError:
            self.logger.error(f"Build command '{self.build_command[0]}' not found. Ensure '{self.build_tool}' is installed and in PATH (or gradlew is executable).")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during build: {e}", exc_info=True)
            return False

    def parse_jacoco_report(self) -> dict | None:
        if not self.jacoco_report_path.exists():
            self.logger.error(f"JaCoCo report not found at: {self.jacoco_report_path}. Ensure your build tool is configured to generate it.")
            return None

        self.logger.info(f"Parsing JaCoCo report: {self.jacoco_report_path}")
        try:
            tree = ET.parse(self.jacoco_report_path)
            root = tree.getroot()

            coverage_data = {
                "overall_line_coverage": 0.0,
                "methods": [] # To store method-level coverage details
            }

            # Overall line coverage from the first 'report > counter[type="LINE"]'
            report_line_counter = root.find('./counter[@type="LINE"]')
            if report_line_counter is not None:
                missed_lines = int(report_line_counter.get('missed', 0))
                covered_lines = int(report_line_counter.get('covered', 0))
                total_lines = missed_lines + covered_lines
                if total_lines > 0:
                    coverage_data["overall_line_coverage"] = covered_lines / total_lines
            else:
                self.logger.warning("Could not find overall LINE counter in JaCoCo report root. Looking for session totals if any.")
                # Attempt to sum up all counters if no root counter (less common for standard reports)
                all_line_counters = root.findall('.//counter[@type="LINE"]')
                total_missed = sum(int(c.get('missed',0)) for c in all_line_counters)
                total_covered = sum(int(c.get('covered',0)) for c in all_line_counters)
                grand_total = total_missed + total_covered
                if grand_total > 0:
                     coverage_data["overall_line_coverage"] = total_covered / grand_total
                else:
                    self.logger.warning("No LINE counters found anywhere in the report.")


            # Method-level coverage: report > package > class > method > counter[@type="LINE"]
            for package_element in root.findall('./package'):
                package_name = package_element.get('name').replace('/', '.')
                for class_element in package_element.findall('./class'):
                    # class_name = class_element.get('name').split('/')[-1] # Simple name
                    # For full_class_name, JaCoCo often provides it with slashes, convert to dots
                    full_class_name_from_report = package_name + '.' + class_element.get('name').replace('/', '.').split('.')[-1]

                    for method_element in class_element.findall('./method'):
                        method_name = method_element.get('name')
                        method_desc = method_element.get('desc') # Signature
                        line_num_str = method_element.get('line') # Starting line number

                        method_line_counter = method_element.find('./counter[@type="LINE"]')
                        if method_line_counter is not None:
                            missed = int(method_line_counter.get('missed', 0))
                            covered = int(method_line_counter.get('covered', 0))
                            method_total_lines = missed + covered
                            method_coverage = 0.0
                            if method_total_lines > 0:
                                method_coverage = covered / method_total_lines

                            coverage_data["methods"].append({
                                "class_name": full_class_name_from_report,
                                "method_name": method_name,
                                "method_signature": method_desc,
                                "start_line": int(line_num_str) if line_num_str and line_num_str.isdigit() else 0,
                                "line_coverage": method_coverage,
                                "covered_lines": covered,
                                "missed_lines": missed,
                                "total_lines": method_total_lines
                            })

            self.logger.info(f"Successfully parsed JaCoCo report. Overall line coverage: {coverage_data['overall_line_coverage']:.2%}")
            return coverage_data

        except ET.ParseError as e:
            self.logger.error(f"Error parsing JaCoCo XML report: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during JaCoCo report parsing: {e}", exc_info=True)
            return None

    def get_coverage(self) -> dict | None:
        self.logger.info("Attempting to run build, generate JaCoCo report, and parse coverage...")
        build_success = self.run_build_and_generate_jacoco_report()
        if build_success:
            self.logger.info("Build successful. Proceeding to parse JaCoCo report.")
            return self.parse_jacoco_report()
        else:
            self.logger.error("Build failed, cannot get coverage. Check build logs for details.")
            return None

if __name__ == '__main__':
    # Ensure basicConfig is called for the main block if running standalone for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    logger_main = logging.getLogger("JavaTestRunnerExample") # Specific logger for this block

    try:
        PROJECT_ROOT_STR = os.getenv("SPRING_BOOT_PROJECT_ROOT")
        if not PROJECT_ROOT_STR:
            raise ValueError("SPRING_BOOT_PROJECT_ROOT environment variable not set for example execution.")

        project_root_path = Path(PROJECT_ROOT_STR)

        build_tool_env = os.getenv("BUILD_TOOL", "maven").lower() # Default to maven for example
        logger_main.info(f"Using build tool: {build_tool_env} for JavaTestRunner example from SPRING_BOOT_PROJECT_ROOT: {project_root_path}")

        if not project_root_path.exists():
            raise FileNotFoundError(f"Project root path does not exist: {project_root_path}")

        runner = JavaTestRunner(project_root=project_root_path, build_tool=build_tool_env)

        # --- Example: Running a single test (if you have a specific test to run) ---
        # This part is more for illustration, as TestCaseGenerator uses run_test internally.
        # You'd need an actual test file path.
        # example_test_file_str = os.getenv("EXAMPLE_TEST_FILE_PATH")
        # if example_test_file_str:
        #    example_test_file = Path(example_test_file_str)
        #    if example_test_file.exists():
        #        logger_main.info(f"Attempting to run single test: {example_test_file}")
        #        single_test_results = runner.run_test(example_test_file)
        #        logger_main.info(f"Single test run results: {single_test_results}")
        #    else:
        #        logger_main.warning(f"EXAMPLE_TEST_FILE_PATH '{example_test_file_str}' does not exist. Skipping single test run example.")
        # else:
        #    logger_main.info("EXAMPLE_TEST_FILE_PATH not set. Skipping single test run example.")


        # --- Example: Get overall coverage ---
        logger_main.info("Attempting to get coverage information...")
        coverage_info = runner.get_coverage()

        if coverage_info:
            logger_main.info(f"Overall Line Coverage: {coverage_info['overall_line_coverage']:.2%}")

            low_coverage_methods = [m for m in coverage_info.get("methods", []) if m['line_coverage'] < 0.90 and m['total_lines'] > 0]
            if low_coverage_methods:
                logger_main.info("\nMethods with < 90% coverage:")
                for method_cov in low_coverage_methods:
                    logger_main.info(
                        f"  Class: {method_cov['class_name']}, Method: {method_cov['method_name']}{method_cov['method_signature']}"
                        f" - Coverage: {method_cov['line_coverage']:.2%} ({method_cov['covered_lines']}/{method_cov['total_lines']} lines)"
                        f" - Starts at line: {method_cov['start_line']}"
                    )
            else:
                logger_main.info("All reported methods have >= 90% coverage or no methods reported with lines.")
        else:
            logger_main.error("Failed to get coverage information from the run.")

    except ValueError as e:
        logger_main.error(f"Configuration error in example: {e}")
    except FileNotFoundError as e:
        logger_main.error(f"File not found error in example: {e}")
    except Exception as e:
        logger_main.error(f"An unexpected error occurred in example usage: {e}", exc_info=True)

# For reference, the previous content from read_files was:
# [start of src/test_runner/java_test_runner.py]
# import subprocess
# from pathlib import Path
# from typing import Dict, Any, Optional
#
# class JavaTestRunner:
#
#     def __init__(self, project_root: Path, build_tool: str = "maven"):
#         self.project_root = project_root
#         self.build_tool = build_tool.lower()
#
#         if self.build_tool not in ["maven", "gradle"]:
#             raise ValueError("Unsupported build tool. Please use 'maven' or 'gradle'.")
#
#         print(f"Initialized JavaTestRunner with build tool: {self.build_tool} for project: {self.project_root}")
#
#     def _run_maven_test(self, test_file_path: Path) -> Dict[str, Any]:
#         relative_test_path = test_file_path.relative_to(self.project_root)
#         test_class_name = str(relative_test_path).replace(".java", "").replace(os.sep, ".")
#
#         command = [
#             "mvn",
#             "test",
#             f"-Dtest={test_class_name}",
#             "-DfailIfNoTests=false"
#         ]
#
#         print(f"Executing Maven command: {' '.join(command)} in {self.project_root}")
#
#         try:
#             process = subprocess.run(
#                 command,
#                 cwd=self.project_root, # Run Maven from the project root
#                 capture_output=True,
#                 text=True,
#                 check=False # Do not raise an exception for non-zero exit codes (we want to check stderr/stdout)
#             )
#             stdout = process.stdout
#             stderr = process.stderr
#             return_code = process.returncode
#
#             # Simple parsing for success/failure (can be improved)
#             if return_code == 0 and "BUILD SUCCESS" in stdout:
#                 # Look for test results summary
#                 if "Tests run: " in stdout:
#                     test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
#                     if test_summary_match:
#                         total = int(test_summary_match.group(1))
#                         failures = int(test_summary_match.group(2))
#                         errors = int(test_summary_match.group(3))
#                         skipped = int(test_summary_match.group(4))
#
#                         if failures == 0 and errors == 0:
#                             return {"status": "SUCCESS", "message": "Tests passed successfully.", "stdout": stdout, "stderr": stderr}
#                         else:
#                             return {"status": "FAILED", "message": "Tests failed or had errors.", "stdout": stdout, "stderr": stderr}
#                 else:
#                     return {"status": "UNKNOWN", "message": "Maven build successful but no test summary found.", "stdout": stdout, "stderr": stderr}
#             else:
#                 return {"status": "ERROR", "message": f"Maven build failed with exit code {return_code}.", "stdout": stdout, "stderr": stderr}
#
#         except FileNotFoundError:
#             return {"status": "ERROR", "message": "Maven (mvn) command not found. Please ensure Maven is installed and in your PATH.", "stdout": "", "stderr": "Maven not found."}
#         except Exception as e:
#             return {"status": "ERROR", "message": f"An unexpected error occurred during Maven execution: {e}", "stdout": "", "stderr": str(e)}
#
#
#
#     def run_test(self, test_file_path: Path) -> Dict[str, Any]:
#
#         if not test_file_path.exists():
#             return {"status": "ERROR", "message": f"Test file not found: {test_file_path}", "stdout": "", "stderr": ""}
#
#         print(f"\nRunning test: {test_file_path}")
#
#         if self.build_tool == "maven":
#             return self._run_maven_test(test_file_path)
#         elif self.build_tool == "gradle":
#             return self._run_gradle_test(test_file_path) # This was missing in the provided old content
#         else:
#             return {"status": "ERROR", "message": "Invalid build tool configured.", "stdout": "", "stderr": ""}
#
# if __name__ == "__main__":
#
#     project_root = Path("/Users/tanmay/Desktop/AMRIT/BeneficiaryID-Generation-API")
#     example_test_file = project_root / "src" / "test" / "java" / "com" / "iemr" / "common" / "bengen" / "service" / "GenerateBeneficiaryServiceTest.java" # Replace with a real test file
#
#
#
#     runner = JavaTestRunner(project_root=project_root, build_tool="maven") # or "gradle"
#     results = runner.run_test(example_test_file)
#
#     print("\n--- Test Results ---")
#     print(f"Status: {results['status']}")
#     print(f"Message: {results['message']}")
#     if results['stdout']:
#         print("\n--- STDOUT ---")
#         print(results['stdout'])
#     if results['stderr']:
#         print("\n--- STDERR ---")
#         print(results['stderr'])
#
# [end of src/test_runner/java_test_runner.py]
