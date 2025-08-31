# 🚀 Quick Setup Guide

## Interactive Configuration Script

We've created an easy-to-use interactive setup script that will configure everything for you!

### Option 1: Run the Setup Script (Recommended)

```bash
# Make sure you're in the project directory
cd Java-testgen-suite

# Run the interactive setup
python3 configure_llm.py
# OR
bash setup_llm.sh
```

This interactive script will:
- ✅ Ask for your Spring Boot project path
- ✅ Auto-detect your build tool (Maven/Gradle)
- ✅ Let you choose your LLM provider
- ✅ Configure API keys and model settings
- ✅ Test the configuration
- ✅ Save everything to `.env` file

### Option 2: Manual Configuration

If you prefer manual setup, copy `.env.example` to `.env` and edit the values:

```bash
cp .env.example .env
# Edit .env with your preferred editor
```

## Available LLM Providers

| Provider | Cost | Speed | Code Quality | Setup Difficulty |
|----------|------|-------|--------------|------------------|
| **Google Gemini** ⭐ | 💰 Low | ⚡ Fast | ⭐⭐⭐⭐ Good | 🟢 Easy |
| **OpenAI GPT** | 💰💰 Medium | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | 🟢 Easy |
| **Anthropic Claude** | 💰💰 Medium | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | 🟢 Easy |
| **Ollama (Local)** | 🆓 Free | ⚡⚡⚡ Slow | ⭐⭐⭐ Good | 🟡 Medium |
| **Azure OpenAI** | 💰💰 Medium | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | 🔴 Hard |

⭐ **Recommended for beginners**: Google Gemini (fast, cheap, good quality)

## After Setup

Once configured, run the full pipeline:

```bash
bash run.sh
```

Or generate tests for specific targets:

```bash
python3 src/llm/test_case_generator.py
```

## Troubleshooting

**Configuration issues?**
- Re-run the setup script: `python3 configure_llm.py`
- Check your API keys are valid
- Ensure your Spring Boot project path is correct

**Build issues?**
- Make sure your project builds successfully: `mvn clean compile` or `gradle build`
- Check that `src/main/java` contains your Java source files

**Need help?** 
- Check the full README.md for detailed documentation
- Review the PROJECT_DOCUMENTATION.md for technical details
