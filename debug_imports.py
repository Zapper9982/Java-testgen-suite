import re

# Test the import pattern
test_imports = [
    'import com.iemr.common.notification.agent.DTO.AlertAndNotificationChangeStatusDTO;',
    'import com.iemr.common.notification.agent.DTO.AlertAndNotificationCountDTO;',
    'import com.iemr.common.notification.util.InputMapper;',
    'import com.iemr.common.utils.response.OutputResponse;'
]

# Current pattern
pattern = re.compile(r'^import\s+(com\.iemr\.[\w\.]+)\.([A-Z][A-Za-z0-9_]*)\;', re.MULTILINE)

print("Testing current pattern:")
for imp in test_imports:
    match = pattern.search(imp)
    if match:
        print(f'MATCH: {imp} -> {match.group(1)}.{match.group(2)}')
    else:
        print(f'NO MATCH: {imp}')

print("\nTesting alternative pattern:")
# Alternative pattern that might work better
alt_pattern = re.compile(r'^import\s+(com\.iemr\.[\w\.]+)\.([A-Z][A-Za-z0-9_]*)\;', re.MULTILINE)

for imp in test_imports:
    match = alt_pattern.search(imp)
    if match:
        print(f'MATCH: {imp} -> {match.group(1)}.{match.group(2)}')
    else:
        print(f'NO MATCH: {imp}') 