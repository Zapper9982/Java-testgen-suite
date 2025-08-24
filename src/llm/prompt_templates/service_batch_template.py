def get_service_batch_prompt(
    target_class_name: str,
    target_package_name: str,
    batch_number: int,
    method_list_str: str,
    context: str,
    previous_test_code: str = None,
    error_feedback: str = None
) -> str:
    """
    Generate prompt for service batch test generation.
    """
    prompt = ""
    
    if previous_test_code:
        prompt += f"PREVIOUS TEST CLASS OUTPUT:\n--- BEGIN PREVIOUS OUTPUT ---\n{previous_test_code}\n--- END PREVIOUS OUTPUT ---\n\n"
        prompt += f"Expand and fix the above test class by addressing the errors and adding/fixing tests for the following methods (do NOT duplicate or overwrite existing tests):\n"
        prompt += f'{method_list_str}\n\n'
    
    if error_feedback:
        prompt += f"ERROR FEEDBACK FROM PREVIOUS ATTEMPT:\n--- BEGIN ERROR OUTPUT ---\n{error_feedback}\n--- END ERROR OUTPUT ---\n\n"
    
    strictness = (
        "IMPORTANT: Only use methods, fields, and constructors that are present in the code blocks below. "
        "Do NOT invent or assume any methods, fields, or classes. "
        "If a method or field is not present in the provided code, do NOT use it in your test. "
        "If you are unsure, leave it out. "
        "If you hallucinate any code, you will be penalized and re-prompted. "
        "You must generate a compiling, passing, and style-compliant test class. "
        "If the class or its methods use a logger (e.g., org.slf4j.Logger, log.info, log.error, etc.), write tests that verify logging behavior where appropriate. "
        "Use Mockito or other suitable techniques to verify that logging statements are called as expected, especially for error or important info logs. "
        "If you are unsure how to verify logging, add a comment in the test indicating what should be checked. "
        "Generate tests for exception/negative paths (e.g., when repo throws). "
        "Cover edge cases and all branches (e.g., with/without working location). "
        "Do NOT define or create dummy DTOs, entities, or repository interfaces inside the test class. Use only the real classes provided in the context. If a class is missing, do NOT invent it—report an error instead. "
        "STRICT: Do NOT invent, assume, or hallucinate any code, types, or dependencies not present in the provided context. If something is missing, add a TODO comment or leave it out, but never guess."
    )
    
    prompt += f"""
You are an expert Java developer. You are to generate a complete JUnit 5 + Mockito test class for the MAIN CLASS below. The other classes are provided as context only (do NOT generate tests for them).

{context}

Instructions:
- Only generate tests for the MAIN CLASS.
- Include all necessary imports and annotations.
- Name the test class {target_class_name}Test_Batch{batch_number} and use the package {target_package_name}.
- Cover ONLY these public methods (do not skip any):\n{method_list_str}
- For each method, create at least one @Test method that tests its functionality.
- Avoid unnecessary stubbing or mocking. Only mock what is required for compilation or to isolate the class under test. Do NOT mock dependencies that are not used in the test method. Do NOT mock simple POJOs or value objects.
- Do NOT output explanations, markdown, or comments outside the code.
- Do NOT USE RelfectionTestUtils 
- Output ONLY the Java code for the test class, nothing else.
- Make sure to include @Test annotations on all test methods.

{strictness}
"""
    
    return prompt
