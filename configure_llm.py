#!/usr/bin/env python3
"""
Interactive LLM Configuration Script for Java Test Generation Suite

This script provides a user-friendly CLI to configure LLM providers.
It automatically updates the .env file based on user selections.
"""

import os
import sys
from pathlib import Path
import subprocess
from typing import Dict, Optional

# ANSI color codes for better CLI experience
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

class LLMConfigurator:
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.env_file = self.repo_root / ".env"
        self.env_example = self.repo_root / ".env.example"
        
        # LLM Provider configurations
        self.providers = {
            "1": {
                "name": "Google Gemini",
                "provider": "google",
                "models": [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro"
                ],
                "default_model": "gemini-2.5-flash",
                "api_key_var": "GOOGLE_API_KEY",
                "description": "Fast, reliable, and cost-effective (Recommended)",
                "setup_url": "https://makersuite.google.com/app/apikey"
            },
            "2": {
                "name": "OpenAI GPT",
                "provider": "openai",
                "models": [
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4o",
                    "gpt-3.5-turbo"
                ],
                "default_model": "gpt-4",
                "api_key_var": "OPENAI_API_KEY",
                "description": "Excellent code quality, higher cost",
                "setup_url": "https://platform.openai.com/api-keys"
            },
            "3": {
                "name": "Anthropic Claude",
                "provider": "anthropic",
                "models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ],
                "default_model": "claude-3-5-sonnet-20241022",
                "api_key_var": "ANTHROPIC_API_KEY",
                "description": "Great at following instructions, good for complex code",
                "setup_url": "https://console.anthropic.com/account/keys"
            },
            "4": {
                "name": "Ollama (Local)",
                "provider": "ollama",
                "models": [
                    "codellama",
                    "llama2",
                    "mistral",
                    "deepseek-coder",
                    "magicoder"
                ],
                "default_model": "codellama",
                "api_key_var": None,
                "description": "Free, private, runs locally (requires Ollama setup)",
                "setup_url": "https://ollama.com/download"
            },
            "5": {
                "name": "Azure OpenAI",
                "provider": "azure",
                "models": [
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-35-turbo"
                ],
                "default_model": "gpt-4",
                "api_key_var": "AZURE_OPENAI_API_KEY",
                "description": "Enterprise OpenAI with Azure integration",
                "setup_url": "https://portal.azure.com/"
            }
        }

    def display_providers(self):
        """Display available LLM providers"""
        print_header("🤖 Available LLM Providers")
        
        for key, provider in self.providers.items():
            print(f"{Colors.OKBLUE}{key}.{Colors.ENDC} {Colors.BOLD}{provider['name']}{Colors.ENDC}")
            print(f"   {provider['description']}")
            print(f"   Default model: {Colors.OKCYAN}{provider['default_model']}{Colors.ENDC}")
            print()

    def get_user_choice(self) -> str:
        """Get provider choice from user"""
        while True:
            try:
                choice = input(f"{Colors.BOLD}Choose LLM provider (1-5): {Colors.ENDC}").strip()
                if choice in self.providers:
                    return choice
                else:
                    print_error("Invalid choice. Please enter 1-5.")
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                sys.exit(0)

    def get_model_choice(self, provider_config: Dict) -> str:
        """Get model choice from user"""
        print(f"\n{Colors.OKBLUE}Available models for {provider_config['name']}:{Colors.ENDC}")
        for i, model in enumerate(provider_config['models'], 1):
            marker = " (recommended)" if model == provider_config['default_model'] else ""
            print(f"  {i}. {model}{Colors.OKGREEN}{marker}{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"\nChoose model (1-{len(provider_config['models'])}) or press Enter for default: ").strip()
                
                if not choice:
                    return provider_config['default_model']
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(provider_config['models']):
                    return provider_config['models'][choice_idx]
                else:
                    print_error(f"Invalid choice. Please enter 1-{len(provider_config['models'])}.")
            except ValueError:
                print_error("Please enter a valid number.")
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                sys.exit(0)

    def get_api_key(self, provider_config: Dict) -> Optional[str]:
        """Get API key from user"""
        if not provider_config['api_key_var']:
            return None
            
        print(f"\n{Colors.OKBLUE}API Key Setup for {provider_config['name']}:{Colors.ENDC}")
        print(f"Get your API key from: {Colors.UNDERLINE}{provider_config['setup_url']}{Colors.ENDC}")
        
        while True:
            try:
                api_key = input(f"Enter your {provider_config['api_key_var']}: ").strip()
                if api_key:
                    return api_key
                else:
                    print_error("API key cannot be empty.")
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                sys.exit(0)

    def get_azure_config(self) -> Dict[str, str]:
        """Get Azure-specific configuration"""
        print(f"\n{Colors.OKBLUE}Azure OpenAI Configuration:{Colors.ENDC}")
        
        config = {}
        required_fields = [
            ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI Endpoint (e.g., https://your-resource.openai.azure.com/)"),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", "Deployment Name"),
        ]
        
        for env_var, description in required_fields:
            while True:
                try:
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        config[env_var] = value
                        break
                    else:
                        print_error("This field cannot be empty.")
                except KeyboardInterrupt:
                    print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                    sys.exit(0)
        
        # Optional API version
        api_version = input("Enter API Version (press Enter for default '2024-02-01'): ").strip()
        config["AZURE_OPENAI_API_VERSION"] = api_version if api_version else "2024-02-01"
        
        return config

    def get_ollama_config(self) -> Dict[str, str]:
        """Get Ollama configuration"""
        print(f"\n{Colors.OKBLUE}Ollama Configuration:{Colors.ENDC}")
        print_info("Make sure Ollama is installed and running: ollama serve")
        
        base_url = input("Enter Ollama base URL (press Enter for default 'http://localhost:11434'): ").strip()
        return {
            "OLLAMA_BASE_URL": base_url if base_url else "http://localhost:11434"
        }

    def get_spring_boot_project_path(self) -> str:
        """Get Spring Boot project path from user"""
        print_header("📁 Spring Boot Project Configuration")
        print("Please provide the absolute path to your Spring Boot project.")
        print(f"{Colors.OKCYAN}Example: /Users/username/projects/my-spring-app{Colors.ENDC}")
        
        # Check if there's an existing path
        existing_env = self.load_existing_env()
        current_path = existing_env.get('SPRING_BOOT_PROJECT_PATH', '')
        
        if current_path and current_path != "/absolute/path/to/your/springboot/project":
            print(f"Current path: {Colors.OKGREEN}{current_path}{Colors.ENDC}")
            use_current = input("Use current path? (Y/n): ").strip().lower()
            if use_current in ['', 'y', 'yes']:
                return current_path
        
        while True:
            try:
                project_path = input("Enter Spring Boot project path: ").strip()
                
                if not project_path:
                    print_error("Project path cannot be empty.")
                    continue
                
                # Expand user path (~)
                project_path = os.path.expanduser(project_path)
                project_path_obj = Path(project_path)
                
                if not project_path_obj.exists():
                    print_error(f"Path does not exist: {project_path}")
                    continue
                
                if not project_path_obj.is_dir():
                    print_error(f"Path is not a directory: {project_path}")
                    continue
                
                # Check if it looks like a Spring Boot project
                pom_xml = project_path_obj / "pom.xml"
                build_gradle = project_path_obj / "build.gradle"
                src_main_java = project_path_obj / "src" / "main" / "java"
                
                if not (pom_xml.exists() or build_gradle.exists()):
                    print_warning("No pom.xml or build.gradle found. Are you sure this is a Maven/Gradle project?")
                    proceed = input("Proceed anyway? (y/N): ").strip().lower()
                    if proceed not in ['y', 'yes']:
                        continue
                
                if not src_main_java.exists():
                    print_warning("No src/main/java directory found. Are you sure this is a Spring Boot project?")
                    proceed = input("Proceed anyway? (y/N): ").strip().lower()
                    if proceed not in ['y', 'yes']:
                        continue
                
                return str(project_path_obj.absolute())
                
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                sys.exit(0)

    def get_build_tool(self) -> str:
        """Get build tool preference"""
        print(f"\n{Colors.OKBLUE}Build Tool Configuration:{Colors.ENDC}")
        
        # Try to auto-detect
        existing_env = self.load_existing_env()
        spring_boot_path = existing_env.get('SPRING_BOOT_PROJECT_PATH', '')
        
        detected_tool = None
        if spring_boot_path and spring_boot_path != "/absolute/path/to/your/springboot/project":
            project_path = Path(spring_boot_path)
            if (project_path / "pom.xml").exists():
                detected_tool = "maven"
            elif (project_path / "build.gradle").exists():
                detected_tool = "gradle"
        
        if detected_tool:
            print(f"Detected build tool: {Colors.OKGREEN}{detected_tool}{Colors.ENDC}")
            use_detected = input("Use detected build tool? (Y/n): ").strip().lower()
            if use_detected in ['', 'y', 'yes']:
                return detected_tool
        
        print("1. Maven (pom.xml)")
        print("2. Gradle (build.gradle)")
        
        while True:
            try:
                choice = input("Choose build tool (1-2): ").strip()
                if choice == "1":
                    return "maven"
                elif choice == "2":
                    return "gradle"
                else:
                    print_error("Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
                sys.exit(0)

    def get_advanced_settings(self) -> Dict[str, str]:
        """Get advanced LLM settings"""
        print(f"\n{Colors.OKBLUE}Advanced LLM Settings (Optional):{Colors.ENDC}")
        
        # Temperature
        while True:
            temp_input = input("Temperature (0.0-1.0, press Enter for default 0.1): ").strip()
            if not temp_input:
                temperature = "0.1"
                break
            try:
                temp_val = float(temp_input)
                if 0.0 <= temp_val <= 1.0:
                    temperature = temp_input
                    break
                else:
                    print_error("Temperature must be between 0.0 and 1.0.")
            except ValueError:
                print_error("Please enter a valid number.")
        
        # Max tokens
        max_tokens_input = input("Max tokens (press Enter for unlimited): ").strip()
        max_tokens = max_tokens_input if max_tokens_input else None
        
        config = {"LLM_TEMPERATURE": temperature}
        if max_tokens:
            config["LLM_MAX_TOKENS"] = max_tokens
            
        return config

    def create_env_config(self, provider_key: str) -> Dict[str, str]:
        """Create complete environment configuration"""
        provider_config = self.providers[provider_key]
        
        print(f"\n{Colors.OKGREEN}Configuring {provider_config['name']}...{Colors.ENDC}")
        
        # Get Spring Boot project path first
        spring_boot_path = self.get_spring_boot_project_path()
        build_tool = self.get_build_tool()
        
        # Basic configuration
        config = {
            "SPRING_BOOT_PROJECT_PATH": spring_boot_path,
            "BUILD_TOOL": build_tool,
            "ENABLE_INCREMENTAL_MERGE": "true",
            "LLM_PROVIDER": provider_config['provider'],
            "LLM_MODEL_NAME": self.get_model_choice(provider_config)
        }
        
        # API Key
        api_key = self.get_api_key(provider_config)
        if api_key and provider_config['api_key_var']:
            config[provider_config['api_key_var']] = api_key
        
        # Provider-specific configuration
        if provider_config['provider'] == 'azure':
            config.update(self.get_azure_config())
        elif provider_config['provider'] == 'ollama':
            config.update(self.get_ollama_config())
        
        # Advanced settings
        config.update(self.get_advanced_settings())
        
        return config

    def load_existing_env(self) -> Dict[str, str]:
        """Load existing environment variables from .env file"""
        env_vars = {}
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        
        return env_vars

    def write_env_file(self, config: Dict[str, str]):
        """Write configuration to .env file"""
        # Load existing configuration
        existing_env = self.load_existing_env()
        
        # Remove old LLM configuration but keep other settings
        llm_keys = [
            'LLM_PROVIDER', 'LLM_MODEL_NAME', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS',
            'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
            'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME',
            'AZURE_OPENAI_API_VERSION', 'OLLAMA_BASE_URL'
        ]
        
        for key in llm_keys:
            existing_env.pop(key, None)
        
        # Merge with new configuration (this will override existing core config too)
        existing_env.update(config)
        
        # Write to file with organized structure
        with open(self.env_file, 'w') as f:
            f.write("# =============================================================================\n")
            f.write("# Java Test Generation Suite Configuration\n")
            f.write("# =============================================================================\n\n")
            
            # Core configuration
            f.write("# Core Configuration (Required)\n")
            f.write(f"SPRING_BOOT_PROJECT_PATH={config['SPRING_BOOT_PROJECT_PATH']}\n")
            f.write(f"BUILD_TOOL={config['BUILD_TOOL']}\n")
            f.write(f"ENABLE_INCREMENTAL_MERGE={config.get('ENABLE_INCREMENTAL_MERGE', 'true')}\n")
            
            f.write("\n# =============================================================================\n")
            f.write("# LLM Configuration\n")
            f.write("# =============================================================================\n\n")
            
            # LLM basic configuration
            f.write(f"LLM_PROVIDER={config['LLM_PROVIDER']}\n")
            f.write(f"LLM_MODEL_NAME={config['LLM_MODEL_NAME']}\n")
            f.write(f"LLM_TEMPERATURE={config.get('LLM_TEMPERATURE', '0.1')}\n")
            
            if 'LLM_MAX_TOKENS' in config:
                f.write(f"LLM_MAX_TOKENS={config['LLM_MAX_TOKENS']}\n")
            
            f.write("\n# API Keys and Provider Settings\n")
            
            # API keys and provider-specific settings
            for key in llm_keys:
                if key in config and key not in ['LLM_PROVIDER', 'LLM_MODEL_NAME', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS']:
                    f.write(f"{key}={config[key]}\n")
            
            # Other existing configuration (if any)
            core_keys = ['SPRING_BOOT_PROJECT_PATH', 'BUILD_TOOL', 'ENABLE_INCREMENTAL_MERGE', 'LLM_PROVIDER', 'LLM_MODEL_NAME', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS']
            other_configs = {k: v for k, v in existing_env.items() 
                           if k not in core_keys and k not in llm_keys}
            
            if other_configs:
                f.write("\n# Additional Configuration\n")
                for key, value in other_configs.items():
                    f.write(f"{key}={value}\n")

    def test_configuration(self, config: Dict[str, str]):
        """Test the LLM configuration"""
        print(f"\n{Colors.OKBLUE}Testing LLM configuration...{Colors.ENDC}")
        
        try:
            # Set environment variables temporarily
            original_env = dict(os.environ)
            os.environ.update(config)
            
            # Test LLM creation
            test_script = '''
import sys
sys.path.insert(0, "src")
from llm.llm_factory import create_llm
try:
    llm = create_llm()
    print(f"SUCCESS: {type(llm).__name__}")
except Exception as e:
    print(f"ERROR: {e}")
'''
            
            result = subprocess.run([sys.executable, '-c', test_script], 
                                  capture_output=True, text=True, timeout=30)
            
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
            
            if "SUCCESS:" in result.stdout:
                print_success("LLM configuration test passed!")
                return True
            else:
                print_error(f"LLM configuration test failed: {result.stdout + result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print_error("LLM configuration test timed out.")
            return False
        except Exception as e:
            print_error(f"Error testing configuration: {e}")
            return False

    def run(self):
        """Run the interactive configuration"""
        print_header("🚀 Java Test Generation Suite - Complete Setup")
        
        print("This script will help you configure:")
        print("• Spring Boot project path")
        print("• Build tool (Maven/Gradle)")  
        print("• LLM provider for automated test generation")
        print(f"\nConfiguration will be saved to: {Colors.OKCYAN}{self.env_file}{Colors.ENDC}")
        
        # Display providers
        self.display_providers()
        
        # Get user choice
        provider_key = self.get_user_choice()
        
        # Create configuration (this now includes Spring Boot path)
        config = self.create_env_config(provider_key)
        
        # Display configuration summary
        print_header("📋 Configuration Summary")
        print(f"Spring Boot Project: {Colors.OKGREEN}{config['SPRING_BOOT_PROJECT_PATH']}{Colors.ENDC}")
        print(f"Build Tool: {Colors.OKGREEN}{config['BUILD_TOOL']}{Colors.ENDC}")
        print(f"LLM Provider: {Colors.OKGREEN}{config['LLM_PROVIDER']}{Colors.ENDC}")
        print(f"LLM Model: {Colors.OKGREEN}{config['LLM_MODEL_NAME']}{Colors.ENDC}")
        print(f"Temperature: {Colors.OKGREEN}{config.get('LLM_TEMPERATURE', '0.1')}{Colors.ENDC}")
        
        # Confirm before saving
        confirm = input(f"\n{Colors.BOLD}Save this configuration? (Y/n): {Colors.ENDC}").strip().lower()
        if confirm in ['n', 'no']:
            print(f"{Colors.WARNING}Configuration cancelled.{Colors.ENDC}")
            return
        
        # Write configuration
        print(f"\n{Colors.OKBLUE}Saving configuration...{Colors.ENDC}")
        self.write_env_file(config)
        print_success(f"Configuration saved to {self.env_file}")
        
        # Test configuration
        if input(f"\nTest the LLM configuration now? (Y/n): ").strip().lower() not in ['n', 'no']:
            if self.test_configuration(config):
                print_success("🎉 Setup completed successfully!")
                print(f"\n{Colors.BOLD}🚀 Ready to generate tests!{Colors.ENDC}")
                print(f"{Colors.OKGREEN}Run the full pipeline: bash run.sh{Colors.ENDC}")
                print(f"{Colors.OKCYAN}Or generate tests directly: python3 src/llm/test_case_generator.py{Colors.ENDC}")
            else:
                print_warning("Configuration saved but LLM test failed. Please check your settings.")
                print(f"You can re-run this script to fix the configuration: {Colors.OKCYAN}python3 configure_llm.py{Colors.ENDC}")
        else:
            print_success("🎉 Configuration completed!")
            print(f"\n{Colors.BOLD}🚀 Next steps:{Colors.ENDC}")
            print(f"{Colors.OKGREEN}1. Run the full pipeline: bash run.sh{Colors.ENDC}")
            print(f"{Colors.OKCYAN}2. Or generate tests directly: python3 src/llm/test_case_generator.py{Colors.ENDC}")
        
        print(f"\n{Colors.OKCYAN}💡 Tip: You can re-run this script anytime to change your configuration.{Colors.ENDC}")

if __name__ == "__main__":
    configurator = LLMConfigurator()
    configurator.run()
