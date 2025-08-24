def get_merge_prompt(
    existing_test_class: str,
    new_batch_code: str,
    target_class_name: str,
    target_package_name: str,
    test_type: str
) -> str:
    """
    Generate prompt for merging new batch into existing test class.
    """
    merge_prompt = f"""
You are an expert Java test developer. Merge the new test methods into the existing test class.

EXISTING TEST CLASS:
{existing_test_class}

NEW BATCH TO ADD:
{new_batch_code}

REQUIREMENTS:
- Keep all existing imports, fields, and methods from the existing test class
- Add new test methods from the batch
- Ensure @InjectMocks field names are consistent (use 'sut' as canonical name)
- Deduplicate any duplicate imports
- Maintain proper class structure and annotations
- Keep all existing @BeforeEach/@AfterEach methods
- If test method names conflict, rename new ones with _BatchN suffix
- Preserve the original package: {target_package_name} (NOT {target_package_name}.{target_class_name}Test)
- Maintain test type consistency: {test_type} (controller/service)
- Ensure the package declaration is: package {target_package_name};

Output ONLY the complete merged test class, no explanations or markdown.
"""
    
    return merge_prompt
