# LLM Provider Configuration Guide

The Java Test Generation Suite now supports multiple LLM providers through a factory pattern. You can easily switch between different AI models based on your needs, budget, and preferences.

## Supported Providers

| Provider | Models Available | Strengths |
|----------|------------------|-----------|
| **Gemini** (default) | `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash-exp` | Large context window, good code understanding |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | Excellent code generation, reliable |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` | Strong reasoning, good at following instructions |
| **Groq** | `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768` | Fast inference, cost-effective |

## Configuration

### 1. Set the Provider

In your `.env` file, set the `LLM_PROVIDER` variable:

```bash
export LLM_PROVIDER=gemini  # Options: gemini, openai, anthropic, groq
```

### 2. Configure API Keys

Set the appropriate API key for your chosen provider:

```bash
# For Gemini (default)
export GOOGLE_API_KEY=your_google_api_key_here

# For OpenAI
export OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Groq
export GROQ_API_KEY=your_groq_api_key_here
```

### 3. Customize Model Settings

Each provider supports model and temperature customization:

```bash
# Gemini Configuration
export GEMINI_MODEL=gemini-1.5-pro
export GEMINI_TEMPERATURE=0.1

# OpenAI Configuration
export OPENAI_MODEL=gpt-4o
export OPENAI_TEMPERATURE=0.1

# Anthropic Configuration
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_TEMPERATURE=0.1

# Groq Configuration
export GROQ_MODEL=llama-3.1-70b-versatile
export GROQ_TEMPERATURE=0.1
```

## Quick Setup Examples

### Using Google Gemini (Default)
```bash
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your_google_api_key_here
export GEMINI_MODEL=gemini-1.5-pro
```

### Using OpenAI GPT-4
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key_here
export OPENAI_MODEL=gpt-4o
```

### Using Anthropic Claude
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Using Groq Llama
```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_groq_api_key_here
export GROQ_MODEL=llama-3.1-70b-versatile
```

## Installation

Make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

This will install all LLM provider libraries:
- `langchain-google-genai` for Gemini
- `langchain-openai` for OpenAI
- `langchain-anthropic` for Anthropic
- `langchain-groq` for Groq

## Getting API Keys

### Google Gemini
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Set it as `GOOGLE_API_KEY`

### OpenAI
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Set it as `OPENAI_API_KEY`

### Anthropic
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an API key
3. Set it as `ANTHROPIC_API_KEY`

### Groq
1. Go to [Groq Console](https://console.groq.com/)
2. Create an API key
3. Set it as `GROQ_API_KEY`

## Usage

Once configured, the system will automatically use your chosen provider:

```bash
# The system will detect your LLM_PROVIDER and use the appropriate model
python3 src/llm/test_case_generator.py --target-class "com.example.MyService"
```

You'll see output like:
```
🤖 Initializing OPENAI LLM...
   Model: gpt-4o
   Temperature: 0.1
```

## Troubleshooting

### Missing Dependency Error
```
❌ Missing dependency for OPENAI: No module named 'langchain_openai'
   Install required package: pip install langchain-openai
```

**Solution**: Install the specific package:
```bash
pip install langchain-openai  # or langchain-anthropic, langchain-groq
```

### API Key Error
```
❌ API key not found. Please set OPENAI_API_KEY environment variable for openai
```

**Solution**: Set the correct API key in your `.env` file.

### Network/Authentication Error
```
❌ Failed to initialize OPENAI LLM: Incorrect API key provided
```

**Solution**: Verify your API key is correct and has appropriate permissions.

## Model Recommendations

### For Production Use
- **OpenAI GPT-4o**: Most reliable and consistent results
- **Anthropic Claude 3.5 Sonnet**: Excellent instruction following

### For Development/Testing
- **Gemini 1.5 Flash**: Good balance of speed and quality
- **Groq Llama 3.1-8b-instant**: Very fast inference

### For Large Codebases
- **Gemini 1.5 Pro**: Large 2M token context window
- **Anthropic Claude 3.5 Sonnet**: 200k token context

## Cost Considerations

Approximate costs per 1M tokens (as of 2024):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Gemini | 1.5-pro | $3.50 | $10.50 |
| Gemini | 1.5-flash | $0.075 | $0.30 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | claude-3.5-sonnet | $3.00 | $15.00 |
| Groq | llama-3.1-70b | $0.59 | $0.79 |

*Prices subject to change - check provider websites for current rates*
