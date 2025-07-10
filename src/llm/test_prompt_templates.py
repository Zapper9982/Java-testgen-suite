"""
Ultra-strict prompt templates for LLM-based test generation for Spring Boot (JUnit5 + Mockito + MockMvc).
Includes few-shot positive and negative examples, and strict output instructions.

Prompt templates for modular, robust Spring Boot controller test generation and validation.
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

IMPORTANT ANTI-HALLUCINATION RULES:
- Do NOT invent or use any methods, fields, classes, or helpers not present in the provided code.
- Only use what is present in the provided code and imports.
- If you are unsure, leave it out or add a comment.
- If you hallucinate, you will be penalized and re-prompted.
"""

# --- Modular Prompt Templates ---

CONTROLLER_PRE_ANALYSIS_PROMPT = """
You are an expert Java Spring Boot developer and test designer.

Given the following controller class, analyze and output a JSON object with:
- "controllerName": Controller class name.
- "package": Package name.
- "dependencies": List of injected dependencies (type and field name).
- "endpoints": For each public method with HTTP mapping:
    - "methodName", "httpMethod", "path", "requestParams" (with types/annotations), "responseType", "validationRules", "authRequired", "businessLogicSummary".

Output ONLY the JSON object, no explanations or markdown.

--- BEGIN CONTROLLER CODE ---
{controller_code}
--- END CONTROLLER CODE ---
"""

CONTROLLER_TEST_GENERATION_PROMPT = """
You are an expert in Java Spring Boot testing. Using the controller analysis and code, generate a complete, idiomatic JUnit 5 test class using @WebMvcTest and MockMvc.

STRICT REQUIREMENTS:
- Use @WebMvcTest({controllerName}.class) on the test class.
- Inject MockMvc with @Autowired.
- Mock all controller dependencies with @MockBean.
- Organize tests using @Nested classes for:
    - HappyPathTests: Successful scenarios.
    - ValidationTests: Input validation failures.
    - ErrorHandlingTests: Exception and error scenarios.
- For each endpoint, generate:
    - At least one happy path test.
    - Tests for validation failures.
    - Tests for error/exception handling.
- Use MockMvc to perform HTTP requests and assert responses.
- Use JsonPath assertions for response body validation.
- Use test data builders or helper methods for request/response objects.
- Use AssertJ or Hamcrest for fluent assertions.
- Use descriptive test method names (e.g., methodName_WhenCondition_ShouldReturnExpected).
- Separate arrange, act, and assert steps clearly.
- Mock service/repository methods to return realistic test data.
- Mock external API/database interactions as needed.
- Cover authentication/authorization edge cases if present.
- Output ONLY the complete Java test class, no explanations or markdown.

--- BEGIN CONTROLLER ANALYSIS ---
{controller_analysis_json}
--- END CONTROLLER ANALYSIS ---

--- BEGIN CONTROLLER CODE ---
{controller_code}
--- END CONTROLLER CODE ---
"""

TEST_QUALITY_CHECK_PROMPT = """
You are a senior Java code reviewer.

Given the following test class and controller analysis, review the test class for:
- Correct use of @WebMvcTest, MockMvc, @MockBean.
- Proper use of @Nested classes for happy path, validation, error handling.
- Coverage of all endpoints with happy path, validation, and error tests.
- Use of JsonPath assertions, test data builders, descriptive method names, clear arrange/act/assert, realistic mocks, and coverage of auth/exception scenarios.
- No anti-patterns (e.g., string comparisons, direct method calls, missing mocks, hardcoded JSON).

Output a JSON object with:
- "score": 1-10.
- "issues": List of specific issues.
- "suggestions": List of concrete improvements.

--- BEGIN CONTROLLER ANALYSIS ---
{controller_analysis_json}
--- END CONTROLLER ANALYSIS ---

--- BEGIN TEST CLASS ---
{test_class_code}
--- END TEST CLASS ---
"""

BEST_PRACTICES_EXAMPLE = """
@WebMvcTest(UserController.class)
class UserControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UserService userService;

    @Nested
    class HappyPathTests {
        @Test
        void getUser_WhenUserExists_ShouldReturnUser() throws Exception {
            User user = User.builder().id(1L).name("Alice").build();
            when(userService.findById(1L)).thenReturn(Optional.of(user));

            mockMvc.perform(get("/users/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1L))
                .andExpect(jsonPath("$.name").value("Alice"));
        }
    }

    @Nested
    class ValidationTests {
        @Test
        void getUser_WhenIdIsInvalid_ShouldReturnBadRequest() throws Exception {
            mockMvc.perform(get("/users/invalid"))
                .andExpect(status().isBadRequest());
        }
    }

    @Nested
    class ErrorHandlingTests {
        @Test
        void getUser_WhenUserNotFound_ShouldReturnNotFound() throws Exception {
            when(userService.findById(99L)).thenReturn(Optional.empty());

            mockMvc.perform(get("/users/99"))
                .andExpect(status().isNotFound());
        }
    }
}
"""

def get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    imports_section = '\n'.join(custom_imports) if custom_imports else ''
    dependency_sig_section = ''
    if dependency_signatures:
        dependency_sig_section = '\n'.join([f"// Dependency: {k}: {v}" for k, v in dependency_signatures.items()])
    return f"""
You are an expert Java Spring Boot test generator. Your task is to generate a JUnit 5 test class for the service `{target_class_name}` in the package `{target_package_name}`.

STRICT REQUIREMENTS:
- Output ONLY compilable Java code, no explanations or markdown.
- Use JUnit 5 (`org.junit.jupiter.api.*`) and Mockito (`org.mockito.*`) to write tests.
- Import all required classes, including Spring annotations and Mockito utilities.
- Use `@ExtendWith(MockitoExtension.class)` for Mockito support.
- Mock all dependencies using `@Mock` and inject them using `@InjectMocks` (and `@Spy` if needed).
- Use `@BeforeEach` for setup if needed.
- For each public method in the service, generate at least one test method that:
    - Mocks dependencies as needed for the method under test using Mockito (`when`, `doReturn`, etc.).
    - Asserts the return value and side effects using JUnit assertions (e.g., `assertEquals`, `assertThrows`).
    - Covers both typical and edge cases, including exception paths.
- Use descriptive test method names (e.g., `shouldReturnXWhenY`).
- Never instantiate dependencies manually—always use dependency injection and mocking.
- Do NOT hallucinate methods, fields, helpers, or imports—use only what is present in the provided context and imports.
- Do NOT invent or use any methods, fields, classes, or helpers not present in the provided code.
- Only use what is present in the provided code and imports.
- If you are unsure, leave it out or add a comment.
- If you hallucinate, you will be penalized and re-prompted.
- Always include `{custom_imports}` in the import section.
- {additional_query_instructions}
- If any dependencies or utility classes are required, use only those present in the context.
- The output must be a single, compilable Java test class, similar in style and completeness to the following high-quality example.

--- BEGIN EXAMPLE TEST CLASS ---
package com.iemr.common.bengen.utils;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

@ExtendWith(MockitoExtension.class)
class CookieUtilTest {{
    @Mock
    HttpServletRequest request;

    @InjectMocks
    @Spy
    CookieUtil cookieUtil;

    @Test
    void getCookieValue_cookieExists() {{
        Cookie cookie = mock(Cookie.class);
        doReturn("myCookieName").when(cookie).getName();
        doReturn("myCookieValue").when(cookie).getValue();
        doReturn(new Cookie[]{{cookie}}).when(request).getCookies();

        Optional<String> result = cookieUtil.getCookieValue(request, "myCookieName");

        assertTrue(result.isPresent());
        assertEquals("myCookieValue", result.get());
    }}

    @Test
    void getCookieValue_cookieDoesNotExist() {{
        doReturn(new Cookie[0]).when(request).getCookies();
        Optional<String> result = cookieUtil.getCookieValue(request, "myCookieName");
        assertFalse(result.isPresent());
    }}

    @Test
    void getCookieValue_cookiesIsNull() {{
        doReturn(null).when(request).getCookies();
        Optional<String> result = cookieUtil.getCookieValue(request, "myCookieName");
        assertFalse(result.isPresent());
    }}

    @Test
    void getJwtTokenFromCookie_jwtCookieExists() {{
        Cookie jwtCookie = mock(Cookie.class);
        doReturn("Jwttoken").when(jwtCookie).getName();
        doReturn("myJwtToken").when(jwtCookie).getValue();
        doReturn(new Cookie[]{{jwtCookie}}).when(request).getCookies();

        String jwtToken = CookieUtil.getJwtTokenFromCookie(request);

        assertEquals("myJwtToken", jwtToken);
    }}

    @Test
    void getJwtTokenFromCookie_jwtCookieDoesNotExist() {{
        Cookie otherCookie = mock(Cookie.class);
        doReturn("otherCookie").when(otherCookie).getName();
        doReturn(new Cookie[]{{otherCookie}}).when(request).getCookies();

        String jwtToken = CookieUtil.getJwtTokenFromCookie(request);

        assertNull(jwtToken);
    }}

    @Test
    void getJwtTokenFromCookie_cookiesIsNull() {{
        doReturn(null).when(request).getCookies();
        String jwtToken = CookieUtil.getJwtTokenFromCookie(request);
        assertNull(jwtToken);
    }}
}}
--- END EXAMPLE TEST CLASS ---

--- BEGIN CLASS UNDER TEST ---
// The following is the code for the class you must write tests for. Do NOT repeat this code. Only write the test class.
// You may use only the methods, fields, and dependencies present in this code and its imports.
{{context}}
--- END CLASS UNDER TEST ---

ADDITIONAL INSTRUCTIONS:
- If the method under test declares 'throws' exceptions, your test must handle them (e.g., with try/catch or appropriate test annotations).
- Do NOT omit required exception handling. If you omit it, the code will not compile.
- If you are unsure, add a try/catch block or use the appropriate JUnit annotation (e.g., assertThrows).

# Add explicit logger verification instructions to all test generation prompt templates
"""

def get_controller_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    imports_section = '\n'.join(custom_imports) if custom_imports else ''
    dependency_sig_section = ''
    if dependency_signatures:
        dependency_sig_section = '\n'.join([f"// Dependency: {k}: {v}" for k, v in dependency_signatures.items()])
    return f"""
// GOOD EXAMPLE (DO THIS):
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class ExampleControllerTest {{
    private MockMvc mockMvc;

    @Mock
    private ExampleService exampleService;

    @InjectMocks
    private ExampleController controller;

    @BeforeEach
    void setUp() {{
        MockitoAnnotations.openMocks(this);
        mockMvc = MockMvcBuilders.standaloneSetup(controller).build();
    }}

    @Test
    void shouldReturnData_whenGetIsSuccessful() throws Exception {{
        when(exampleService.getData(anyString())).thenReturn("mockData");
        mockMvc.perform(get("/example/get/{id}", "1")
                .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data").value("mockData"));
    }}
}}

STRICT REQUIREMENTS for Standalone Controller Tests:
- Use @ExtendWith(MockitoExtension.class) for the test class.
- Use @InjectMocks for the controller under test.
- Use @Mock for all service/repository dependencies.
- Do NOT use @WebMvcTest, @MockBean, or @Autowired.
- Initialize MockMvc in a @BeforeEach method using MockMvcBuilders.standaloneSetup(controller).build();
- All test methods must use mockMvc.perform(...) to simulate HTTP requests.
- Do NOT instantiate the controller manually (use @InjectMocks).
- Do NOT use Spring context injection.
- Output only valid, compilable Java code for the test class, no explanations or markdown.
- Use descriptive test method names (e.g., shouldReturnXWhenY).
- Cover both typical and edge cases, including exception paths.
- Never instantiate dependencies manually—always use @Mock and @InjectMocks.
- Do NOT hallucinate methods, fields, helpers, or imports—use only what is present in the provided context and imports.
- Do NOT invent or use any methods, fields, classes, or helpers not present in the provided code.
- If you are unsure, leave it out or add a comment.
- If you hallucinate, you will be penalized and re-prompted.
- Always include `{custom_imports}` in the import section.
- {additional_query_instructions}
- If any dependencies or utility classes are required, use only those present in the context.
- The output must be a single, compilable Java test class, similar in style and completeness to the GOOD EXAMPLE above.

--- BEGIN CLASS UNDER TEST ---
// The following is the code for the class you must write tests for. Do NOT repeat this code. Only write the test class.
// You may use only the methods, fields, and dependencies present in this code and its imports.
{{context}}
--- END CLASS UNDER TEST ---
"""

def get_repository_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    # Use the same example and structure as above for consistency
    return get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures)

def get_best_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, class_code, dependency_signatures=None):
    """
    Returns the best test prompt template for the given class.
    - If the class is a controller (detected by @RestController, @Controller, or class name ending with 'Controller'), returns the MockMvc controller prompt.
    - Otherwise, returns the service or repository prompt as appropriate.
    """
    # Simple detection logic
    is_controller = False
    if class_code:
        if ('@RestController' in class_code or '@Controller' in class_code or
            'extends' in class_code and 'Controller' in class_code or
            'implements' in class_code and 'Controller' in class_code):
            is_controller = True
    if target_class_name.lower().endswith('controller'):
        is_controller = True
    if is_controller:
        return get_controller_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures)
    # Could add repository detection here if needed
    return get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures) 