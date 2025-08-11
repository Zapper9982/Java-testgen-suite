# Incremental Test Class Merging

Copilot-style incremental merging that intelligently combines batch-generated test methods into a single, growing test class.

## How It Works

**Before (Original):**
```
Batch 1 → Write to file → Run tests → Success
Batch 2 → Write to file → Run tests → Success  
Batch 3 → Write to file → Run tests → Success
...
Final: Manual merge needed
```

**After (Incremental Merge):**
```
Batch 1 → Write to file → Run tests → Success → Keep as base test class
Batch 2 → Generate → LLM Merge with existing → Write merged → Run tests → Success
Batch 3 → Generate → LLM Merge with existing → Write merged → Run tests → Success
...
Final: Single merged test class ready
```

## Configuration

```bash
# Enable/disable incremental merging (default: true)
export ENABLE_INCREMENTAL_MERGE=true

# Keep batch files on merge failure (default: true)
export KEEP_BATCH_FILES_ON_MERGE_FAILURE=true
```

## File Structure

**With Merge Enabled:**
```
src/test/java/{package}/
├── {ClassName}Test.java (grows incrementally)
└── {ClassName}Test/ (batch files - for debugging)
    ├── {ClassName}Test_Batch1.java
    └── {ClassName}Test_Batch2.java
```

## What the LLM Merge Does

1. **Preserves Existing Code**: Keeps all imports, fields, and methods
2. **Adds New Methods**: Integrates new test methods from batch
3. **Unifies @InjectMocks**: Uses 'sut' as canonical SUT variable name
4. **Deduplicates Imports**: Removes duplicate import statements
5. **Handles Conflicts**: Renames duplicate test methods with `_BatchN` suffix

## Usage

```bash
# Enable incremental merging
export ENABLE_INCREMENTAL_MERGE=true

# Run test generation (will use incremental merging)
python3 src/llm/test_case_generator.py

# Test the merge functionality
python3 test_incremental_merge.py
```

## Benefits

- ✅ Intelligent integration of test methods
- ✅ Consistent single test class state
- ✅ Better error recovery with retry attempts
- ✅ No final merge step needed
- ✅ Backward compatible with existing workflow
- ✅ Automatic fallback if merge fails
