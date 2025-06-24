"""
Ultra-strict prompt templates for LLM-based test generation for Spring Boot (JUnit5 + Mockito + MockMvc).
Includes few-shot positive and negative examples, and strict output instructions.
"""

# --- FEW-SHOT EXAMPLES (used in all templates) ---
POSITIVE_EXAMPLE = '''
// GOOD EXAMPLE (do this):
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    @Mock UserRepository ;
    @InjectMocks UserService userService;

    @Test
    void shouldReturnUser_whenUserExists() {
        User user = new User("alice");
        when(userRepository.findByName("alice")userRepository).thenReturn(user);
        User result = userService.findByName("alice");
        assertEquals("alice", result.getName());
    }
}
'''

NEGATIVE_EXAMPLE = '''
// BAD EXAMPLE (never do this):
// import com.fake.NonExistent;
// @Test void testFake() { fail(); }
// This test will not compile. Never invent imports or methods.
'''

STRICT_OUTPUT_INSTRUCTIONS = """
Output ONLY a single compilable Java code block. If you cannot generate a test for any method, output a comment in the code explaining why. Never output explanations or text outside the code block.
"""

def get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    return f"""
You are an expert Java Spring Boot test generator. Your task is to generate a JUnit 5 test class for the service `{target_class_name}` in the package `{target_package_name}`.

STRICT REQUIREMENTS:
- Output ONLY compilable Java code, no explanations or markdown.
- Use JUnit 5 (`org.junit.jupiter.api.*`) and Mockito (`org.mockito.*`).
- Import all required classes, including Spring annotations and Mockito utilities.
- Use `@ExtendWith(MockitoExtension.class)` for Mockito support.
- Mock all dependencies using `@Mock` and inject them using `@InjectMocks`.
- For each public method in the service, generate at least one test method that:
    - Mocks dependencies as needed for the method under test.
    - Asserts the return value and side effects using JUnit assertions (e.g., `assertEquals`, `assertThrows`).
    - Covers both typical and edge cases if possible.
- Use descriptive test method names (e.g., `shouldReturnXWhenY`).
- Include all necessary setup (e.g., `@BeforeEach` if needed).
- Do NOT hallucinate methods or imports—use only what is present in the provided context and imports.
- Always include `{custom_imports}` in the import section.
- {additional_query_instructions}
- If any dependencies or utility classes are required, use only those present in the context.
- The output must be a single, compilable Java test class.
"""

def get_controller_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    return f"""
You are an expert Java Spring Boot test generator. Your task is to generate a JUnit 5 test class for the controller `{target_class_name}` in the package `{target_package_name}`.

STRICT REQUIREMENTS:
- Output ONLY compilable Java code, no explanations or markdown.
- Use JUnit 5 (`org.junit.jupiter.api.*`), Mockito (`org.mockito.*`), and MockMvc (`org.springframework.test.web.servlet.*`).
- Import all required classes, including Spring annotations, MockMvc, and Mockito utilities.
- Use `@WebMvcTest({target_class_name}.class)` and proper Spring test configuration.
- Mock all dependencies using `@MockBean` or `@Mock` as appropriate.
- Use `@Autowired` for MockMvc and the controller under test.
- For each public endpoint/method in the controller, generate at least one test method that:
    - Uses MockMvc to perform the request (e.g., `.perform(get(...))`)
    - Asserts the response status, content, and any relevant side effects using `andExpect` and JUnit assertions.
    - Mocks dependencies as needed for each test.
- Use descriptive test method names (e.g., `shouldReturnXWhenY`).
- Include all necessary setup (e.g., `@BeforeEach` for MockMvc setup if needed).
- Do NOT hallucinate methods, endpoints, or imports—use only what is present in the provided context and imports.
- Always include `{custom_imports}` in the import section.
- {additional_query_instructions}
- If any dependencies or utility classes are required, use only those present in the context.
- The output must be a single, compilable Java test class.
"""

def get_repository_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    return f"""
You are an expert Java Spring Boot test generator. Your task is to generate a JUnit 5 test class for the repository `{target_class_name}` in the package `{target_package_name}`.

STRICT REQUIREMENTS:
- Output ONLY compilable Java code, no explanations or markdown.
- Use JUnit 5 (`org.junit.jupiter.api.*`) and Mockito (`org.mockito.*`).
- Import all required classes, including Spring annotations and Mockito utilities.
- Use `@DataJpaTest` for repository tests.
- Mock or provide test data as needed for repository methods.
- For each public method in the repository, generate at least one test method that:
    - Sets up the required data in the test database (use in-memory DB if possible).
    - Asserts the results of repository methods using JUnit assertions.
- Use descriptive test method names (e.g., `shouldReturnXWhenY`).
- Include all necessary setup (e.g., `@BeforeEach` if needed).
- Do NOT hallucinate methods or imports—use only what is present in the provided context and imports.
- Always include `{custom_imports}` in the import section.
- {additional_query_instructions}
- If any dependencies or utility classes are required, use only those present in the context.
- The output must be a single, compilable Java test class.
""" 