import sys
import os
from pathlib import Path
import json

# Setup import path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from analyzer.code_analysis_utils import extract_method_invocations_in_batch
except ImportError as e:
    print(f"[ERROR] Could not import extract_method_invocations_in_batch: {e}")
    sys.exit(1)

def create_dummy_java_file(filepath: Path):
    content = """
package com.example.demo;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class DummyService {

    @Autowired
    private UserRepository userRepository; // Used

    @Autowired
    private EmailService emailService; // Unused

    @Autowired
    private AuditLogger auditLogger; // Used transitively

    public void doSomething(String username) {
        userRepository.findByUsername(username);
        logAction("search");
    }

    public void doNothing() {
        // No dependencies used
    }

    private void logAction(String action) {
        auditLogger.log(action);
    }
}
"""
    with open(filepath, "w") as f:
        f.write(content)

def run_verification():
    dummy_file = PROJECT_ROOT / "DummyService.java"
    create_dummy_java_file(dummy_file)

    print(f"Created dummy file at {dummy_file}")

    try:
        # Test Case 1: doSomething (uses userRepository directly, auditLogger transitively)
        print("\n--- Test Case 1: doSomething ---")
        usage_data = extract_method_invocations_in_batch(dummy_file, ["doSomething"])
        used_types = usage_data.get("used_types", set())
        used_fields = usage_data.get("used_fields", set())

        print(f"Used Fields: {used_fields}")
        print(f"Used Types: {used_types}")

        errors = []
        if "UserRepository" not in used_types:
            errors.append("Missing direct dependency: UserRepository")
        if "AuditLogger" not in used_types:
            errors.append("Missing transitive dependency: AuditLogger")
        if "EmailService" in used_types:
            errors.append("Incorrectly included unused dependency: EmailService")

        if errors:
            print("[FAIL] Test Case 1 Failed:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("[PASS] Test Case 1 Passed")

        # Test Case 2: doNothing (no dependencies)
        print("\n--- Test Case 2: doNothing ---")
        usage_data_2 = extract_method_invocations_in_batch(dummy_file, ["doNothing"])
        used_types_2 = usage_data_2.get("used_types", set())
        
        print(f"Used Types: {used_types_2}")
        
        if used_types_2:
             print(f"[FAIL] Test Case 2 Failed: Expected empty set, got {used_types_2}")
        else:
             print("[PASS] Test Case 2 Passed")

    except Exception as e:
        print(f"[ERROR] Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if dummy_file.exists():
            os.remove(dummy_file)
            print(f"\nCleaned up {dummy_file}")

if __name__ == "__main__":
    run_verification()
