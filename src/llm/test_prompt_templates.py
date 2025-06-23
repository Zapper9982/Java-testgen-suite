"""
Ultra-strict prompt templates for LLM-based test generation for Spring Boot (JUnit5 + Mockito + MockMvc).
"""

def get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    formatted_custom_imports = "\n".join(custom_imports)
    dependency_context = ""
    if dependency_signatures:
        dependency_context += "<dependency_method_signatures>\n"
        for class_name, signatures in dependency_signatures.items():
            dependency_context += f"// Methods for dependency: {class_name}\n{signatures}\n\n"
        dependency_context += "</dependency_method_signatures>\n\n"
    return f"""
<persona>
You are an expert Senior Java Developer specializing in writing clean, maintainable, and robust unit tests for Spring Boot services using JUnit 5 and Mockito. Your code must be production-quality and must compile without errors.
</persona>
<task>
Generate a comprehensive JUnit 5 unit test class for the `{target_class_name}` class in the `{target_package_name}` package.
</task>
<rules>
<rule number="1" importance="critical">Output ONLY compilable Java code. Do NOT output explanations, comments, or text outside the code block.</rule>
<rule number="2" importance="critical">Use only methods, fields, and dependencies that exist in the provided context. Never hallucinate methods, fields, or imports.</rule>
<rule number="3" importance="critical">Use `@ExtendWith(MockitoExtension.class)` for JUnit 5.</rule>
<rule number="4" importance="critical">Declare dependencies to be mocked with `@Mock`.</rule>
<rule number="5" importance="critical">Declare the class under test with `@InjectMocks`.</rule>
<rule number="6" importance="high">If `{target_class_name}` calls its own public methods that need stubbing, annotate `@InjectMocks` with `@Spy`.</rule>
<rule number="7" importance="critical">When stubbing a `@Spy` object, ALWAYS use `doReturn(value).when(spyObject).method()` syntax. NEVER use `when(spyObject.method()).thenReturn(value)` for spies.</rule>
<rule number="8" importance="critical">Never use undefined variables, ambiguous types, or TODOs. Never use methods or fields not present in the context.</rule>
<rule number="9" importance="critical">Include all necessary imports. The target class is `import {target_package_name}.{target_class_name};`.</rule>
<rule number="10" importance="critical">Use `org.junit.jupiter.api.Assertions` for assertions.</rule>
<rule number="11" importance="high">Create a separate test method for each public method in the class under test. Test edge cases, null inputs, and exception paths.</rule>
<rule number="12" importance="high">Use meaningful test method names like `testMethod_WhenCondition_ShouldBehavior()`.</rule>
<rule number="13" importance="critical">Always use the correct constructor and dependency injection style as in the source.</rule>
<rule number="14" importance="critical">Never use ambiguous or generic types. Always use the exact types from the context.</rule>
</rules>
<context>
<custom_imports_from_source>
{formatted_custom_imports}
</custom_imports_from_source>
{dependency_context}
<retrieved_source_code>
{{context}}
</retrieved_source_code>
</context>
<instructions>
{additional_query_instructions}
Provide ONLY the complete Java code block for the test class, enclosed in ```java ... ```.
Do not include any explanations or conversational text.
</instructions>
"""

def get_controller_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    formatted_custom_imports = "\n".join(custom_imports)
    dependency_context = ""
    if dependency_signatures:
        dependency_context += "<dependency_method_signatures>\n"
        for class_name, signatures in dependency_signatures.items():
            dependency_context += f"// Methods for dependency: {class_name}\n{signatures}\n\n"
        dependency_context += "</dependency_method_signatures>\n\n"
    return f"""
<persona>
You are an expert Senior Java Developer specializing in writing robust, isolated unit tests for Spring Boot REST controllers using JUnit 5, Mockito, and MockMvc. Your code must be production-quality and must compile without errors.
</persona>
<task>
Generate a comprehensive JUnit 5 test class for the `{target_class_name}` controller in the `{target_package_name}` package. Use `@WebMvcTest({target_class_name}.class)` and `MockMvc` for endpoint testing.
</task>
<rules>
<rule number="1" importance="critical">Output ONLY compilable Java code. Do NOT output explanations, comments, or text outside the code block.</rule>
<rule number="2" importance="critical">Use only methods, fields, and dependencies that exist in the provided context. Never hallucinate methods, fields, or imports.</rule>
<rule number="3" importance="critical">Use `@WebMvcTest({target_class_name}.class)` for the test class. Do not use `@SpringBootTest`.</rule>
<rule number="4" importance="critical">Inject `MockMvc` using `@Autowired`.</rule>
<rule number="5" importance="critical">Mock all dependencies of the controller using `@MockBean`.</rule>
<rule number="6" importance="high">Write a separate test method for each endpoint (e.g., for each `@GetMapping`, `@PostMapping`, etc.).</rule>
<rule number="7" importance="critical">Use `mockMvc.perform(...)` to simulate HTTP requests and assert responses.</rule>
<rule number="8" importance="high">Test edge cases, invalid input, and error responses.</rule>
<rule number="9" importance="critical">Include all necessary imports. The target class is `import {target_package_name}.{target_class_name};`.</rule>
<rule number="10" importance="critical">Use `org.springframework.test.web.servlet.result.MockMvcResultMatchers` for assertions.</rule>
<rule number="11" importance="critical">Never use undefined variables, ambiguous types, or TODOs. Never use methods or fields not present in the context.</rule>
<rule number="12" importance="critical">Always use the correct constructor and dependency injection style as in the source.</rule>
<rule number="13" importance="critical">Never use ambiguous or generic types. Always use the exact types from the context.</rule>
</rules>
<context>
<custom_imports_from_source>
{formatted_custom_imports}
</custom_imports_from_source>
{dependency_context}
<retrieved_source_code>
{{context}}
</retrieved_source_code>
</context>
<instructions>
{additional_query_instructions}
Provide ONLY the complete Java code block for the test class, enclosed in ```java ... ```.
Do not include any explanations or conversational text.
</instructions>
"""

def get_repository_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    formatted_custom_imports = "\n".join(custom_imports)
    dependency_context = ""
    if dependency_signatures:
        dependency_context += "<dependency_method_signatures>\n"
        for class_name, signatures in dependency_signatures.items():
            dependency_context += f"// Methods for dependency: {class_name}\n{signatures}\n\n"
        dependency_context += "</dependency_method_signatures>\n\n"
    return f"""
<persona>
You are an expert Senior Java Developer specializing in writing clean, maintainable, and robust unit tests for Spring Boot repositories using JUnit 5 and Mockito. Your code must be production-quality and must compile without errors.
</persona>
<task>
Generate a comprehensive JUnit 5 unit test class for the `{target_class_name}` repository in the `{target_package_name}` package.
</task>
<rules>
<rule number="1" importance="critical">Output ONLY compilable Java code. Do NOT output explanations, comments, or text outside the code block.</rule>
<rule number="2" importance="critical">Use only methods, fields, and dependencies that exist in the provided context. Never hallucinate methods, fields, or imports.</rule>
<rule number="3" importance="critical">Use `@ExtendWith(MockitoExtension.class)` for the test class.</rule>
<rule number="4" importance="critical">Mock all dependencies using `@Mock`.</rule>
<rule number="5" importance="critical">Test repository methods in isolation, mocking any database interactions.</rule>
<rule number="6" importance="high">Write a separate test method for each public method in the repository.</rule>
<rule number="7" importance="critical">Include all necessary imports. The target class is `import {target_package_name}.{target_class_name};`.</rule>
<rule number="8" importance="critical">Use `org.junit.jupiter.api.Assertions` for assertions.</rule>
<rule number="9" importance="critical">Never use undefined variables, ambiguous types, or TODOs. Never use methods or fields not present in the context.</rule>
<rule number="10" importance="critical">Always use the correct constructor and dependency injection style as in the source.</rule>
<rule number="11" importance="critical">Never use ambiguous or generic types. Always use the exact types from the context.</rule>
</rules>
<context>
<custom_imports_from_source>
{formatted_custom_imports}
</custom_imports_from_source>
{dependency_context}
<retrieved_source_code>
{{context}}
</retrieved_source_code>
</context>
<instructions>
{additional_query_instructions}
Provide ONLY the complete Java code block for the test class, enclosed in ```java ... ```.
Do not include any explanations or conversational text.
</instructions>
""" 