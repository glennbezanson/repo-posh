# repo-posh

GitHub repository beautifier using Claude via Azure AI Foundry.

## What This Does

Crawls your GitHub repos, analyzes code/READMEs, and uses Claude to generate **visually stunning** documentation with:
- 🎯 Emoji-prefixed descriptions
- 🛡️ Shields.io badges (language, license, tech stack)
- ✨ Formatted feature lists with emoji bullets
- 📦 Clean code blocks with syntax highlighting
- 📊 Tables for configuration options

Tracks state so subsequent runs only process new or changed repos.

## Project Structure

```
repo-posh/
├── repo_posh.py      # Main CLI tool (single file, ~500 lines)
├── requirements.txt  # requests, rich
├── .env.example      # Template for environment variables
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python repo_posh.py --setup
```

Required configuration:
- `GITHUB_TOKEN` - Personal access token with `repo`, `read:user` scopes
- `AZURE_AI_ENDPOINT` - Azure AI Foundry endpoint (e.g., `https://your-resource.openai.azure.com`)
- `AZURE_AI_API_KEY` - API key from Azure
- `AZURE_AI_DEPLOYMENT` - Deployment name for Claude model (default: `claude-opus-4-5`)

Config stored in `~/.repo-posh/config.json`. State stored in `~/.repo-posh/state.json`.

## Usage

```bash
python repo_posh.py              # Process all new/changed repos
python repo_posh.py --dry-run    # Preview without applying
python repo_posh.py --force      # Reprocess everything
python repo_posh.py --repos X Y  # Process specific repos only
python repo_posh.py --status     # Show what's been processed
python repo_posh.py --reset      # Clear state file
```

## How It Works

1. Fetches all repos via GitHub API (filters to non-forks with push access)
2. For each repo: gets README, languages, existing topics
3. Computes hash of repo state - skips if unchanged since last run
4. Calls Claude via Azure Foundry with repo context, asks for JSON with:
   - `description`: One-liner, max 160 chars
   - `topics`: 5-10 relevant tags
   - `readme`: Polished markdown documentation
5. Updates GitHub via API (description, topics, README.md)
6. Saves state with hash so next run skips processed repos

## Azure Foundry Integration

Uses Azure OpenAI-compatible endpoint pattern:
```
POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-06-01
Headers: api-key: {key}
```

The prompt asks Claude to return pure JSON (no markdown wrapping). Parsing handles both clean JSON and markdown-wrapped responses.

## Key Functions

- `get_github_repos()` - Fetches all repos for authenticated user
- `get_repo_readme()` / `get_repo_languages()` / `get_repo_topics()` - Repo metadata
- `call_azure_claude()` - Makes API call to Azure Foundry
- `generate_repo_content()` - Builds prompt, calls Claude, parses JSON response
- `update_repo_description()` / `update_repo_topics()` / `update_repo_readme()` - GitHub updates
- `compute_repo_hash()` - Creates hash for change detection
- `run_posh()` - Main processing loop

## State Management

State file tracks:
```json
{
  "repos": {
    "repo-name": {
      "hash": "md5-of-state",
      "processed_at": "2025-01-01T00:00:00",
      "description": "generated description"
    }
  },
  "last_run": "2025-01-01T00:00:00"
}
```

Hash is computed from: current description, first 1000 chars of README, updated_at timestamp.

## Common Modifications

**Add interactive confirmation before applying:**
Add a prompt in `run_posh()` after displaying proposed changes, before the "Apply changes" section.

**Change Claude model:**
Update `azure_deployment` in config or pass different `AZURE_AI_DEPLOYMENT` env var.

**Customize generated content:**
Modify the system prompt and user prompt in `generate_repo_content()`. The prompt specifies:
- Emoji prefix for descriptions
- Shields.io badge format and placement
- Section headers with emojis (✨ Features, 🚀 Quick Start, etc.)
- Emoji bullets for feature lists
- Table formatting for config options
- Horizontal rules between sections

**Add more metadata to generation:**
Fetch additional repo info (issues, contributors, etc.) and include in the prompt context.

**Change what triggers reprocessing:**
Modify `compute_repo_hash()` to include/exclude different fields.
