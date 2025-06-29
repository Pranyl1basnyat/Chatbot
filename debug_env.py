import os
from pathlib import Path

print("=== Environment Debug Checker ===\n")

# Check current working directory
current_dir = Path.cwd()
print(f"Current working directory: {current_dir}")

# Look for .env file
env_file = current_dir / ".env"
print(f"Looking for .env file at: {env_file}")
print(f".env file exists: {env_file.exists()}")

# If .env exists, show its contents (safely)
if env_file.exists():
    print(f"\n.env file contents:")
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if 'TOKEN' in key.upper():
                            print(f"Line {i}: {key}=***[HIDDEN - {len(value)} chars]***")
                        else:
                            print(f"Line {i}: {line}")
                    else:
                        print(f"Line {i}: {line} (WARNING: No '=' found)")
                else:
                    print(f"Line {i}: {line}")
    except Exception as e:
        print(f"Error reading .env file: {e}")
else:
    print("\n.env file NOT FOUND!")
    print("You need to create a .env file with your HuggingFace token.")

# Check if HF_TOKEN is already in environment
print(f"\nCurrent HF_TOKEN in environment: {os.environ.get('HF_TOKEN', 'NOT SET')}")

# Try to load with dotenv
print("\n--- Attempting to load with dotenv ---")
try:
    from dotenv import load_dotenv, find_dotenv
    
    # Find .env file
    found_env = find_dotenv()
    print(f"find_dotenv() found: {found_env}")
    
    # Load it
    result = load_dotenv()
    print(f"load_dotenv() result: {result}")
    
    # Check again
    token_after = os.environ.get('HF_TOKEN')
    if token_after:
        print(f"✅ HF_TOKEN loaded successfully! (length: {len(token_after)} chars)")
    else:
        print("❌ HF_TOKEN still not loaded")
        
except ImportError:
    print("❌ python-dotenv not installed!")
    print("Run: pip install python-dotenv")
except Exception as e:
    print(f"Error with dotenv: {e}")

print("\n=== Debug Complete ===")