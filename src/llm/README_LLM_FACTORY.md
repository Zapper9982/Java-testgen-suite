# LLM Factory Documentation

## Overview

The LLM Factory provides a centralized, extensible way to instantiate different LLM providers for the Java Test Generation Suite. It supports multiple providers with graceful fallbacks for missing dependencies.

## Quick Start

```python
from llm.llm_factory import create_llm

# Use with environment variables
llm = create_llm()

# Or with custom config
llm = create_llm_with_config(
    provider="openai",
    model_name="gpt-4",
    temperature=0.0
)
```

## Supported Providers

| Provider | Models | Status | Installation |
|----------|---------|--------|-------------|
| **Google Gemini** | gemini-2.5-flash, gemini-2.5-pro | ✅ Default | Built-in |
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-3.5-turbo | ⚡ Optional | `pip install langchain-openai` |
| **Anthropic** | claude-3-sonnet, claude-3-haiku | ⚡ Optional | `pip install langchain-anthropic` |
| **Ollama** | codellama, llama2, mistral | ⚡ Optional | `pip install langchain-community` |
| **Azure OpenAI** | Your deployments | ⚡ Optional | `pip install langchain-openai` |

## Environment Configuration

Set these variables in your `.env` file:

```bash
# Provider Selection
LLM_PROVIDER=google              # google, openai, anthropic, ollama, azure
LLM_MODEL_NAME=gemini-2.5-flash  # Provider-specific model name
LLM_TEMPERATURE=0.1              # 0.0-1.0
LLM_MAX_TOKENS=8192              # Maximum tokens

# API Keys (provider-specific)
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Azure OpenAI Configuration  
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

## Adding New Providers

To add a new LLM provider:

1. **Add Import** (with try/except):
```python
try:
    from langchain_newprovider import ChatNewProvider
except ImportError:
    ChatNewProvider = None
```

2. **Add Configuration** to `LLMConfig.__init__()`:
```python
self.newprovider_api_key = os.getenv("NEWPROVIDER_API_KEY")
```

3. **Add Factory Method**:
```python
@staticmethod
def _create_newprovider_llm(config: LLMConfig) -> ChatNewProvider:
    if ChatNewProvider is None:
        raise ValueError("NewProvider not available. Install with: pip install langchain-newprovider")
    
    if not config.newprovider_api_key:
        raise ValueError("NEWPROVIDER_API_KEY environment variable is not set.")
    
    return ChatNewProvider(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.newprovider_api_key
    )
```

4. **Add Case** to `create_llm()`:
```python
elif provider == "newprovider":
    return LLMFactory._create_newprovider_llm(config)
```

5. **Update Available Providers**:
```python
if ChatNewProvider is not None:
    providers.append("newprovider")
```

## Testing

Test the factory with different configurations:

```python
from llm.llm_factory import LLMFactory

# Check available providers
print(LLMFactory.get_available_providers())

# Get configuration help
LLMFactory.print_configuration_help()

# Test creation
llm = LLMFactory.create_llm()
```

## Migration from Old Code

**Before:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
```

**After:**
```python
from llm.llm_factory import create_llm
llm = create_llm()  # Uses environment configuration
```

## Error Handling

The factory provides helpful error messages:

- Missing API keys → Clear instructions
- Unavailable providers → Installation guidance  
- Invalid configuration → Detailed error messages
- Auto-help display on failures

## Best Practices

1. **Use Environment Variables**: Keep credentials secure
2. **Start with Gemini**: It's the default and works well
3. **Test Locally**: Use Ollama for private/offline development
4. **Monitor Costs**: Different providers have different pricing
5. **Set Appropriate Temperature**: 0.0-0.2 for code generation

## Example Usage Patterns

```python
# Production: Use environment config
llm = create_llm()

# Development: Override specific settings
llm = create_llm_with_config(temperature=0.0)

# Testing: Use local model
llm = create_llm_with_config(provider="ollama", model_name="codellama")

# Cost optimization: Use cheaper model
llm = create_llm_with_config(
    provider="anthropic", 
    model_name="claude-3-haiku-20240307"
)
```
