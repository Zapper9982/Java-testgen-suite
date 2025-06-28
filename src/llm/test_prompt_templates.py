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

IMPORTANT ANTI-HALLUCINATION RULES:
- Do NOT invent or use any methods, fields, classes, or helpers not present in the provided code.
- Only use what is present in the provided code and imports.
- If you are unsure, leave it out or add a comment.
- If you hallucinate, you will be penalized and re-prompted.
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
    # Use the same example and structure as above for consistency
    return get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures)


def get_repository_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures=None):
    # Use the same example and structure as above for consistency
    return get_service_test_prompt_template(target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures) 