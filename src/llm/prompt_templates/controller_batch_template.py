def get_controller_batch_prompt(
    target_class_name: str,
    target_package_name: str,
    batch_number: int,
    method_list_str: str,
    context: str,
    previous_test_code: str = None,
    error_feedback: str = None
) -> str:
    """
    Generate prompt for controller batch test generation.
    """
    prompt = ""
    
    if previous_test_code:
        prompt += f"PREVIOUS TEST CLASS OUTPUT:\n--- BEGIN PREVIOUS OUTPUT ---\n{previous_test_code}\n--- END PREVIOUS OUTPUT ---\n\n"
        prompt += f"Expand and fix the above test class by addressing the errors and adding/fixing tests for the following methods (do NOT duplicate or overwrite existing tests):\n"
        prompt += f'{method_list_str}\n\n'
    
    if error_feedback:
        prompt += f"ERROR FEEDBACK FROM PREVIOUS ATTEMPT:\n--- BEGIN ERROR OUTPUT ---\n{error_feedback}\n--- END ERROR OUTPUT ---\n\n"
    
    # Add strictest warning at the top of the prompt
    strictest_warning = (
        "IMPORTANT: You MUST NOT invent, assume, or hallucinate any methods, fields, classes, dependencies, or behaviors that are not explicitly present in the provided code context. "
        "If something is missing, leave it out or add a TODO comment. If you are unsure, DO NOT guess. "
        "If you hallucinate, you will be penalized and re-prompted.\n\n"
    )
    prompt += strictest_warning
    
    strict_requirements = """
STRICT REQUIREMENTS:
- You MUST use standalone MockMvc with Mockito: use @ExtendWith(MockitoExtension.class), @InjectMocks for the controller, @Mock for dependencies, and initialize MockMvc in @BeforeEach using MockMvcBuilders.standaloneSetup(controller).
- Do NOT use @WebMvcTest, @MockBean, @Autowired, or @SpringBootTest for controllers.
- Do NOT use field injection for MockMvc or dependencies.
- Do NOT generate any code for dependencies. Only generate the test class for the controller. Assume all dependencies exist and are available for mocking.
- If you use any forbidden annotation or pattern, you will be penalized and re-prompted.
- Output ONLY compilable Java code, no explanations or markdown.
- All test methods must use mockMvc to perform HTTP requests and assert responses.
- STRICT: Do NOT invent, assume, or hallucinate any code, types, or dependencies not present in the provided context. If something is missing, add a TODO comment or leave it out, but never guess.
"""
    
    prompt += f"""
You are an expert Java developer. You are to generate a complete JUnit 5 + Mockito controller test class for the MAIN CLASS below. The other classes are provided as context only (do NOT generate tests for them).

{context}

Instructions:
- Only generate tests for the MAIN CLASS.
- Include all necessary imports and annotations.
- Name the test class {target_class_name}Test_Batch{batch_number} and use the package {target_package_name}.
- Cover ONLY these public methods (do not skip any):\n{method_list_str}
- For each method, create at least one @Test method that tests its functionality.
- Use standalone MockMvc for all HTTP request/response assertions.
- Use @ExtendWith(MockitoExtension.class), @InjectMocks for the controller, @Mock for dependencies, and initialize MockMvc in @BeforeEach using MockMvcBuilders.standaloneSetup(controller).
- Avoid unnecessary stubbing or mocking. Only mock what is required for compilation or to isolate the class under test. Do NOT mock dependencies that are not used in the test method. Do NOT mock simple POJOs or value objects.
- Do NOT output explanations, markdown, or comments outside the code.
- Output ONLY the Java code for the test class, nothing else.
- Make sure to include @Test annotations on all test methods.

{strict_requirements}
"""
    
    return prompt
