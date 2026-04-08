import os
from huggingface_hub import HfApi

def deploy():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: Please set your HF_TOKEN environment variable first.")
        print("You can get one from: https://huggingface.co/settings/tokens")
        print('\nOn Windows PowerShell, run: $env:HF_TOKEN="your_token_here"')
        print('On Linux/macOS, run: export HF_TOKEN="your_token_here"\n')
        return

    api = HfApi()
    
    try:
        username = api.whoami(token=token)["name"]
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    repo_name = "crypto-trading-ai-agent"
    repo_id = f"{username}/{repo_name}"

    print(f"🚀 Initializing Hugging Face Space: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token
        )
    except Exception as e:
        print(f"❌ Failed to create space: {e}")
        return

    print("📤 Uploading project files to Hugging Face...")
    
    # We exclude large/unnecessary folders
    ignore_patterns = [".venv/*", "__pycache__/*", ".git/*", "HF_README.md", "deploy_space.py"]
    
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        token=token
    )
    
    print("📝 Applying Space metadata to README...")
    # HF Spaces requires the configuration frontmatter in the uploaded README.md
    with open("HF_README.md", "r", encoding="utf-8") as f:
        # Extract everything before the placeholder line
        hf_readme = f.read().split("See README.md")[0]
        
    with open("README.md", "r", encoding="utf-8") as f:
        main_readme = f.read()
        
    combined = hf_readme + "\n\n" + main_readme
    
    with open("_temp_readme.md", "w", encoding="utf-8") as f:
        f.write(combined)
        
    api.upload_file(
        path_or_fileobj="_temp_readme.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token
    )
    os.remove("_temp_readme.md")
    
    print()
    print("✅ Deployment completed successfully!")
    print(f"🔗 Your Space URL: https://huggingface.co/spaces/{repo_id}")
    print("\n⚠️  IMPORTANT NEXT STEP:")
    print(f"Go to your Space Settings at https://huggingface.co/spaces/{repo_id}/settings")
    print("and set your API Keys as 'Space Secrets' (e.g., BINANCE_API_KEY, BINANCE_API_SECRET, API_BASE_URL, MODEL_NAME, etc.)")

if __name__ == "__main__":
    deploy()
