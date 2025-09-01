"""
LLM Factory for Java Test Generation Suite

This factory provides a centralized way to instantiate different LLM providers
based on environment configuration. Adding new LLM providers is as simple as:
1. Adding the import (with try/except for optional dependencies)
2. Adding the provider configuration
3. Adding a case in the factory method

Supported Providers:
- Google Gemini (default)
- OpenAI GPT models
- Anthropic Claude
- Ollama (local models)
- Azure OpenAI
"""

import os
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel

# Core imports (always available)
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional imports (with graceful fallbacks)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
except ImportError:
    Ollama = None
    ChatOllama = None

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None


class LLMConfig:
    """Configuration class for LLM settings"""
    
    def __init__(self):
        # Provider and model selection
        self.provider = os.getenv("LLM_PROVIDER", "google").lower()
        self.model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        
        # Optional max_tokens - if not provided, let LLM use its default
        max_tokens_env = os.getenv("LLM_MAX_TOKENS")
        self.max_tokens = int(max_tokens_env) if max_tokens_env else None
        
        # API Keys and endpoints
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Azure OpenAI specific
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


class LLMFactory:
    """Factory class for creating LLM instances based on configuration"""
    
    @staticmethod
    def create_llm(config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        Create an LLM instance based on the configuration.
        
        Args:
            config: LLMConfig instance. If None, creates from environment variables.
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If provider is not supported or required credentials are missing
        """
        if config is None:
            config = LLMConfig()
        
        provider = config.provider.lower()
        
        if provider == "google" or provider == "gemini":
            return LLMFactory._create_google_llm(config)
        elif provider == "openai" or provider == "gpt":
            return LLMFactory._create_openai_llm(config)
        elif provider == "anthropic" or provider == "claude":
            return LLMFactory._create_anthropic_llm(config)
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm(config)
        elif provider == "azure" or provider == "azure-openai":
            return LLMFactory._create_azure_openai_llm(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: google, openai, anthropic, ollama, azure")
    
    @staticmethod
    def _create_google_llm(config: LLMConfig) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance"""
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. "
                           "Please set it for Gemini API calls.")
        
        tokens_info = f"max_tokens={config.max_tokens}" if config.max_tokens else "unlimited tokens"
        print(f"🤖 Creating Google Gemini LLM: {config.model_name} "
              f"(temp={config.temperature}, {tokens_info})")
        
        # Build parameters conditionally
        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "google_api_key": config.google_api_key
        }
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
            
        return ChatGoogleGenerativeAI(**params)
    
    @staticmethod
    def _create_openai_llm(config: LLMConfig) -> ChatOpenAI:
        """Create OpenAI GPT LLM instance"""
        if ChatOpenAI is None:
            raise ValueError("OpenAI provider not available. "
                           "Install with: pip install langchain-openai")
        
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. "
                           "Please set it for OpenAI API calls.")
        
        tokens_info = f"max_tokens={config.max_tokens}" if config.max_tokens else "unlimited tokens"
        print(f"🤖 Creating OpenAI LLM: {config.model_name} "
              f"(temp={config.temperature}, {tokens_info})")
        
        # Build parameters conditionally
        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "openai_api_key": config.openai_api_key
        }
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
            
        return ChatOpenAI(**params)
    
    @staticmethod
    def _create_anthropic_llm(config: LLMConfig) -> ChatAnthropic:
        """Create Anthropic Claude LLM instance"""
        if ChatAnthropic is None:
            raise ValueError("Anthropic provider not available. "
                           "Install with: pip install langchain-anthropic")
        
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set. "
                           "Please set it for Anthropic API calls.")
        
        tokens_info = f"max_tokens={config.max_tokens}" if config.max_tokens else "unlimited tokens"
        print(f"🤖 Creating Anthropic LLM: {config.model_name} "
              f"(temp={config.temperature}, {tokens_info})")
        
        # Build parameters conditionally
        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "anthropic_api_key": config.anthropic_api_key
        }
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
            
        return ChatAnthropic(**params)
    
    @staticmethod
    def _create_ollama_llm(config: LLMConfig) -> ChatOllama:
        """Create Ollama local LLM instance"""
        if ChatOllama is None:
            raise ValueError("Ollama provider not available. "
                           "Install with: pip install langchain-community")
        
        tokens_info = f"max_tokens={config.max_tokens}" if config.max_tokens else "unlimited tokens"
        print(f"🤖 Creating Ollama LLM: {config.model_name} at {config.ollama_base_url} "
              f"(temp={config.temperature}, {tokens_info})")
        
        # Build parameters conditionally (Ollama might not support max_tokens)
        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.ollama_base_url
        }
        # Note: Some Ollama models may not support max_tokens parameter
        if config.max_tokens:
            try:
                params["max_tokens"] = config.max_tokens
            except Exception:
                print("⚠️  Warning: This Ollama model may not support max_tokens parameter")
                
        return ChatOllama(**params)
    
    @staticmethod
    def _create_azure_openai_llm(config: LLMConfig) -> AzureChatOpenAI:
        """Create Azure OpenAI LLM instance"""
        if AzureChatOpenAI is None:
            raise ValueError("Azure OpenAI provider not available. "
                           "Install with: pip install langchain-openai")
        
        if not config.azure_api_key or not config.azure_endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                           "environment variables must be set for Azure OpenAI.")
        
        if not config.azure_deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable "
                           "must be set for Azure OpenAI.")
        
        tokens_info = f"max_tokens={config.max_tokens}" if config.max_tokens else "unlimited tokens"
        print(f"🤖 Creating Azure OpenAI LLM: {config.azure_deployment_name} "
              f"(temp={config.temperature}, {tokens_info})")
        
        # Build parameters conditionally
        params = {
            "azure_deployment": config.azure_deployment_name,
            "temperature": config.temperature,
            "azure_endpoint": config.azure_endpoint,
            "api_key": config.azure_api_key,
            "api_version": config.azure_api_version
        }
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
            
        return AzureChatOpenAI(**params)
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available LLM providers based on installed packages"""
        providers = ["google"]  # Always available
        
        if ChatOpenAI is not None:
            providers.extend(["openai", "azure"])
        if ChatAnthropic is not None:
            providers.append("anthropic")
        if ChatOllama is not None:
            providers.append("ollama")
            
        return providers
    
    @staticmethod
    def print_configuration_help():
        """Print help message for configuring different LLM providers"""
        print("\n" + "="*60)
        print("🚀 LLM FACTORY CONFIGURATION HELP")
        print("="*60)
        print("\n📋 Environment Variables:")
        print("LLM_PROVIDER       - Provider to use: google, openai, anthropic, ollama, azure")
        print("LLM_MODEL_NAME     - Model name (provider-specific)")
        print("LLM_TEMPERATURE    - Temperature (0.0-1.0, default: 0.1)")
        print("LLM_MAX_TOKENS     - Max tokens (default: 8192)")
        
        print("\n🔑 Google Gemini:")
        print("GOOGLE_API_KEY     - Your Google API key")
        print("Examples: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash")
        
        print("\n🔑 OpenAI:")
        print("OPENAI_API_KEY     - Your OpenAI API key") 
        print("Examples: gpt-4, gpt-4-turbo, gpt-3.5-turbo")
        
        print("\n🔑 Anthropic:")
        print("ANTHROPIC_API_KEY  - Your Anthropic API key")
        print("Examples: claude-3-sonnet-20240229, claude-3-haiku-20240307")
        
        print("\n🔑 Ollama (Local):")
        print("OLLAMA_BASE_URL    - Ollama server URL (default: http://localhost:11434)")
        print("Examples: llama2, codellama, mistral")
        
        print("\n🔑 Azure OpenAI:")
        print("AZURE_OPENAI_ENDPOINT      - Your Azure endpoint")
        print("AZURE_OPENAI_API_KEY       - Your Azure API key") 
        print("AZURE_OPENAI_DEPLOYMENT_NAME - Your deployment name")
        print("AZURE_OPENAI_API_VERSION   - API version (default: 2024-02-01)")
        
        print(f"\n✅ Available providers: {', '.join(LLMFactory.get_available_providers())}")
        print("="*60)


# Convenience functions for easy usage
def create_llm() -> BaseChatModel:
    """Create LLM with default configuration from environment variables"""
    return LLMFactory.create_llm()


def create_llm_with_config(**kwargs) -> BaseChatModel:
    """Create LLM with custom configuration parameters"""
    config = LLMConfig()
    
    # Override config with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    return LLMFactory.create_llm(config)


# Example usage and testing
if __name__ == "__main__":
    # Print configuration help
    LLMFactory.print_configuration_help()
    
    # Test LLM creation (will use environment variables)
    try:
        llm = create_llm()
        print(f"\n✅ Successfully created LLM: {type(llm).__name__}")
    except Exception as e:
        print(f"\n❌ Failed to create LLM: {e}")
