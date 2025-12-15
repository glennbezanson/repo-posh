#!/usr/bin/env python3
"""
repo-posh: Beautify your GitHub repos with AI-generated descriptions and READMEs.
Uses Claude via Azure AI Foundry to polish your public presence.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from anthropic import AnthropicFoundry
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

console = Console(force_terminal=True)

# Configuration
STATE_FILE = Path.home() / ".repo-posh" / "state.json"
CONFIG_FILE = Path.home() / ".repo-posh" / "config.json"


def load_config() -> dict:
    """Load configuration from file or environment variables."""
    config = {
        "github_token": os.environ.get("GITHUB_TOKEN"),
        "azure_resource": os.environ.get("AZURE_AI_RESOURCE"),
        "azure_api_key": os.environ.get("AZURE_AI_API_KEY"),
        "azure_model": os.environ.get("AZURE_AI_MODEL", "claude-sonnet-4-20250514"),
    }

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            file_config = json.load(f)
            config.update({k: v for k, v in file_config.items() if v})

    return config


def save_config(config: dict):
    """Save configuration to file."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"[green]Config saved to {CONFIG_FILE}[/green]")


def load_state() -> dict:
    """Load processing state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"repos": {}, "last_run": None}


def save_state(state: dict):
    """Save processing state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_github_repos(token: str, username: Optional[str] = None) -> list[dict]:
    """Fetch all repos for the authenticated user or specified username."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    if username:
        url = f"https://api.github.com/users/{username}/repos"
    else:
        url = "https://api.github.com/user/repos"

    repos = []
    page = 1

    while True:
        response = requests.get(url, headers=headers, params={"page": page, "per_page": 100, "type": "owner"})
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1

    return repos


def get_repo_readme(token: str, owner: str, repo: str) -> Optional[str]:
    """Fetch the README content for a repo."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    content = response.json()
    return base64.b64decode(content["content"]).decode("utf-8")


def get_repo_languages(token: str, owner: str, repo: str) -> dict:
    """Fetch language breakdown for a repo."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_repo_topics(token: str, owner: str, repo: str) -> list[str]:
    """Fetch topics/tags for a repo."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/topics"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("names", [])


def update_repo_description(token: str, owner: str, repo: str, description: str) -> bool:
    """Update a repo's description."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.patch(url, headers=headers, json={"description": description})
    return response.status_code == 200


def update_repo_topics(token: str, owner: str, repo: str, topics: list[str]) -> bool:
    """Update a repo's topics/tags."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/topics"

    # GitHub topics must be lowercase, max 50 chars, alphanumeric with hyphens
    clean_topics = [t.lower().replace(" ", "-")[:50] for t in topics[:20]]

    response = requests.put(url, headers=headers, json={"names": clean_topics})
    return response.status_code == 200


def update_repo_readme(token: str, owner: str, repo: str, content: str, sha: Optional[str]) -> bool:
    """Update or create a README file."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/README.md"

    data = {
        "message": "docs: beautify README via repo-posh",
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        data["sha"] = sha

    response = requests.put(url, headers=headers, json=data)
    return response.status_code in (200, 201)


def get_readme_sha(token: str, owner: str, repo: str) -> Optional[str]:
    """Get the SHA of the current README file."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/README.md"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("sha")
    return None


def call_azure_claude(config: dict, system: str, user: str) -> str:
    """Call Claude via Azure AI Foundry using Anthropic SDK."""
    resource = config["azure_resource"]
    model = config.get("azure_model", "claude-sonnet-4-20250514")
    api_key = config["azure_api_key"]

    client = AnthropicFoundry(
        api_key=api_key,
        resource=resource,
    )

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    return response.content[0].text


def parse_json_response(text: str) -> dict:
    """Parse JSON from response, handling markdown-wrapped responses."""
    text = text.strip()

    # Extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        nl = text.find("\n", start)
        if nl > start:
            start = nl + 1
        end = text.rfind("```")
        if end > start:
            text = text[start:end].strip()

    # Fix newlines inside JSON string values
    # Join lines that are inside a string (odd quote count)
    lines = text.split("\n")
    fixed_lines = []
    in_string = False

    for line in lines:
        quote_count = line.count('"') - line.count('\\"')

        if in_string:
            # Continuing a string - append to previous line with space
            if fixed_lines:
                fixed_lines[-1] = fixed_lines[-1].rstrip() + " " + line.lstrip()
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        if quote_count % 2 == 1:
            in_string = not in_string

    text = "\n".join(fixed_lines)

    # Try parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: find JSON object boundaries
    if "{" in text and "}" in text:
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        try:
            return json.loads(text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def generate_repo_content(config: dict, repo: dict, readme: Optional[str],
                          languages: dict, topics: list[str], include_readme: bool = False) -> dict:
    """Use Claude to generate polished description, topics, and optionally README."""

    primary_lang = list(languages.keys())[0] if languages else None
    lang_list = ", ".join(languages.keys()) if languages else "Unknown"
    license_id = repo.get("license", {}).get("spdx_id") if repo.get("license") else None

    system_prompt = """You are a GitHub README designer who makes repositories look VISUALLY STUNNING.
Your job is to create eye-catching, professional content that stands out.
You love emojis, badges, clean formatting, and visual hierarchy.
Make repos look like they belong to a serious developer who cares about presentation.

CRITICAL: Return ONLY valid JSON. No markdown code blocks around it."""

    if include_readme:
        prompt = f"""Analyze this GitHub repository and generate VISUALLY APPEALING content:

Repository: {repo['name']}
Owner: {repo['owner']['login']}
Current Description: {repo.get('description') or 'None'}
Languages: {lang_list}
Primary Language: {primary_lang or 'Unknown'}
Stars: {repo.get('stargazers_count', 0)}
Forks: {repo.get('forks_count', 0)}
Has Issues: {repo.get('has_issues', False)}
License: {license_id or 'None'}
Current Topics: {', '.join(topics) if topics else 'None'}

Current README:
{readme[:3000] if readme else 'No README exists'}

Generate the following as JSON with these exact keys:

1. "description": A compelling one-line description (max 160 chars). Start with an emoji that represents the project. Focus on what problem it solves.

2. "topics": An array of 5-10 relevant GitHub topics/tags. Include language names, frameworks, and problem domains. Lowercase, hyphenated.

3. "readme": A VISUALLY STUNNING README.md with:

   HEADER SECTION:
   - Project name as H1 with a relevant emoji
   - Shields.io badges on ONE line right under the title, including:
     * Language badge: ![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
     * License badge if known: ![License](https://img.shields.io/badge/license-{license_id or 'MIT'}-green)

   OVERVIEW:
   - One SHORT paragraph (2-3 sentences max) explaining what it does
   - Use **bold** for the key value proposition

   FEATURES SECTION:
   - Use "## ✨ Features" as header
   - Bullet list with emoji prefix for each feature
   - Keep each bullet to ONE line

   QUICK START:
   - Use "## 🚀 Quick Start" as header
   - Minimal steps to get running
   - Code blocks with language hints

   USAGE:
   - Use "## 📖 Usage" as header
   - Show 2-3 common commands or code examples

   STYLE RULES:
   - Use horizontal rules (---) to separate major sections
   - Keep it CONCISE - no walls of text
   - NO table of contents

Return ONLY the JSON object."""
    else:
        prompt = f"""Analyze this GitHub repository and generate improved metadata:

Repository: {repo['name']}
Owner: {repo['owner']['login']}
Current Description: {repo.get('description') or 'None'}
Languages: {lang_list}
Primary Language: {primary_lang or 'Unknown'}
Stars: {repo.get('stargazers_count', 0)}
License: {license_id or 'None'}
Current Topics: {', '.join(topics) if topics else 'None'}

Current README (first 2000 chars):
{readme[:2000] if readme else 'No README exists'}

Generate JSON with these exact keys:

1. "description": A compelling one-line description (max 160 chars). Start with an emoji that represents the project. Focus on what problem it solves.

2. "topics": An array of 5-10 relevant GitHub topics/tags. Include language names, frameworks, and problem domains. Lowercase, hyphenated.

Return ONLY the JSON object, no markdown wrapping."""

    response = call_azure_claude(config, system_prompt, prompt)
    return parse_json_response(response)


def compute_repo_hash(repo: dict, readme: Optional[str]) -> str:
    """Compute a hash of repo state to detect changes."""
    state_str = json.dumps({
        "description": repo.get("description"),
        "readme": readme[:1000] if readme else None,
        "updated_at": repo.get("updated_at"),
    }, sort_keys=True)
    return hashlib.md5(state_str.encode()).hexdigest()


def display_repos_table(repos: list[dict], state: dict):
    """Display a table of repos and their status."""
    table = Table(title="GitHub Repositories")
    table.add_column("Repository", style="cyan")
    table.add_column("Description", style="dim", max_width=40)
    table.add_column("Stars", justify="right")
    table.add_column("Status", style="green")

    for repo in repos:
        name = repo["name"]
        desc = repo.get("description") or "[dim]No description[/dim]"
        stars = str(repo.get("stargazers_count", 0))

        if name in state.get("repos", {}):
            status = "✓ Processed"
        else:
            status = "○ New"

        table.add_row(name, desc[:40], stars, status)

    console.print(table)


def run_posh(config: dict, dry_run: bool = False, force: bool = False,
             repos_filter: list[str] = None, include_readme: bool = False,
             output_file: Optional[str] = None):
    """Main processing loop."""
    state = load_state()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching repositories...", total=None)
        repos = get_github_repos(config["github_token"])
        progress.remove_task(task)

    # Filter to non-forks with push access
    repos = [r for r in repos if not r.get("fork") and r.get("permissions", {}).get("push", False)]

    # Apply name filter if specified
    if repos_filter:
        repos = [r for r in repos if r["name"] in repos_filter]

    console.print(f"\n[bold]Found {len(repos)} repositories[/bold]\n")
    display_repos_table(repos, state)

    # For markdown output
    md_output = "# repo-posh Dry Run Report\n\n"
    md_output += f"**Repositories found:** {len(repos)}\n\n"
    md_output += f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    md_output += "---\n\n"

    updated = []
    skipped = []

    for repo in repos:
        name = repo["name"]
        owner = repo["owner"]["login"]

        console.print(f"\n[bold cyan]Processing: {name}[/bold cyan]")

        # Get current content
        readme = get_repo_readme(config["github_token"], owner, name)
        languages = get_repo_languages(config["github_token"], owner, name)
        topics = get_repo_topics(config["github_token"], owner, name)
        current_hash = compute_repo_hash(repo, readme)

        # Check if already processed and unchanged
        if not force and name in state.get("repos", {}):
            if state["repos"][name].get("hash") == current_hash:
                console.print("  [dim]No changes detected, skipping[/dim]")
                skipped.append(name)
                continue

        # Generate new content
        with console.status("  Generating polished content..."):
            try:
                new_content = generate_repo_content(config, repo, readme, languages, topics, include_readme)
            except Exception as e:
                console.print(f"  [red]Error generating content: {e}[/red]")
                md_output += f"## {name}\n\n**Error:** {e}\n\n---\n\n"
                continue

        # Display what would change
        console.print("\n  [bold]Proposed changes:[/bold]")
        console.print(f"  [yellow]Description:[/yellow] {new_content.get('description', 'N/A')}")
        console.print(f"  [yellow]Topics:[/yellow] {', '.join(new_content.get('topics', []))}")
        if new_content.get("readme"):
            console.print(f"  [yellow]README:[/yellow] {len(new_content['readme'])} characters")

        # Add to markdown output
        md_output += f"## {name}\n\n"
        md_output += f"**URL:** {repo['html_url']}\n\n"
        md_output += f"### Current\n"
        md_output += f"- **Description:** {repo.get('description') or '(none)'}\n"
        md_output += f"- **Topics:** {', '.join(topics) if topics else '(none)'}\n\n"
        md_output += f"### Proposed\n"
        md_output += f"- **Description:** {new_content.get('description', '(none)')}\n"
        md_output += f"- **Topics:** {', '.join(new_content.get('topics', []))}\n"
        if new_content.get("readme"):
            md_output += f"- **README:** {len(new_content['readme'])} characters\n"
        md_output += "\n---\n\n"

        if dry_run:
            console.print("  [dim](dry run - no changes made)[/dim]")
            updated.append(name)  # Count as "would update"
            continue

        # Apply changes
        success = True

        if new_content.get("description"):
            if update_repo_description(config["github_token"], owner, name, new_content["description"]):
                console.print("  [green]✓ Updated description[/green]")
            else:
                console.print("  [red]✗ Failed to update description[/red]")
                success = False

        if new_content.get("topics"):
            if update_repo_topics(config["github_token"], owner, name, new_content["topics"]):
                console.print("  [green]✓ Updated topics[/green]")
            else:
                console.print("  [red]✗ Failed to update topics[/red]")
                success = False

        if new_content.get("readme") and include_readme:
            sha = get_readme_sha(config["github_token"], owner, name)
            if update_repo_readme(config["github_token"], owner, name, new_content["readme"], sha):
                console.print("  [green]✓ Updated README[/green]")
            else:
                console.print("  [red]✗ Failed to update README[/red]")
                success = False

        if success:
            updated.append(name)
            state.setdefault("repos", {})[name] = {
                "hash": current_hash,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "description": new_content.get("description"),
            }

    # Save state
    if not dry_run:
        save_state(state)

    # Write markdown output if requested
    if output_file:
        Path(output_file).write_text(md_output, encoding='utf-8')
        console.print(f"\n[green]Report written to {output_file}[/green]")

    # Summary
    console.print("\n" + "=" * 50)
    console.print(Panel(
        f"[green]{'Would update' if dry_run else 'Updated'}:[/green] {len(updated)}\n"
        f"[yellow]Skipped:[/yellow] {len(skipped)}\n"
        f"[dim]State file: {STATE_FILE}[/dim]",
        title="Summary"
    ))


def setup_wizard():
    """Interactive setup wizard."""
    console.print(Panel("[bold]repo-posh Setup Wizard[/bold]", style="cyan"))

    config = load_config()

    console.print("\n[bold]GitHub Configuration[/bold]")
    console.print("Create a token at: https://github.com/settings/tokens")
    console.print("Required scopes: repo, read:user\n")

    if config.get("github_token"):
        console.print(f"Current token: [dim]{config['github_token'][:8]}...configured[/dim]")
        if console.input("Update? (y/N): ").lower() != "y":
            pass
        else:
            config["github_token"] = console.input("GitHub Token: ").strip()
    else:
        config["github_token"] = console.input("GitHub Token: ").strip()

    console.print("\n[bold]Azure AI Foundry Configuration[/bold]")
    console.print("Get these from your Azure AI Foundry deployment\n")

    config["azure_resource"] = console.input(
        f"Azure Resource Name [{config.get('azure_resource', 'edgesol-ai')}]: "
    ).strip() or config.get("azure_resource", "edgesol-ai")

    current_key = config.get('azure_api_key', '')
    key_display = current_key[:8] + '...' if current_key else ''
    config["azure_api_key"] = console.input(
        f"Azure API Key [{key_display}]: "
    ).strip() or config.get("azure_api_key", "")

    config["azure_model"] = console.input(
        f"Model Name [{config.get('azure_model', 'claude-sonnet-4-20250514')}]: "
    ).strip() or config.get("azure_model", "claude-sonnet-4-20250514")

    save_config(config)
    console.print("\n[green]Setup complete![/green]")


def show_status():
    """Show processing status."""
    state = load_state()

    console.print(f"\n[bold]Last run:[/bold] {state.get('last_run', 'Never')}")
    console.print(f"[bold]Processed repos:[/bold] {len(state.get('repos', {}))}")

    if state.get("repos"):
        table = Table(title="Processed Repositories")
        table.add_column("Repository", style="cyan")
        table.add_column("Processed At", style="green")
        table.add_column("Description", style="dim", max_width=50)

        for name, info in sorted(state["repos"].items()):
            processed = info.get("processed_at", "unknown")
            desc = info.get("description", "")[:50]
            if len(info.get("description", "")) > 50:
                desc += "..."
            table.add_row(name, processed, desc)

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Beautify your GitHub repos with AI-generated descriptions and READMEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  repo-posh --setup              # Run setup wizard
  repo-posh --dry-run            # Preview changes without applying
  repo-posh --dry-run -o report.md  # Preview and save to markdown
  repo-posh --force              # Re-process all repos
  repo-posh --repos foo bar      # Only process specific repos
  repo-posh --with-readme        # Also generate/update READMEs
  repo-posh --status             # Show current state
        """
    )

    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all repos")
    parser.add_argument("--repos", nargs="+", help="Only process specific repos by name")
    parser.add_argument("--with-readme", action="store_true", help="Also generate/update README files")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--reset", action="store_true", help="Reset processing state")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write report to markdown file")

    args = parser.parse_args()

    console.print(Panel("[bold cyan]repo-posh[/bold cyan] - GitHub Repository Beautifier", style="cyan"))

    if args.setup:
        setup_wizard()
        return

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            console.print("[green]State reset successfully[/green]")
        else:
            console.print("[dim]No state file to reset[/dim]")
        return

    if args.status:
        show_status()
        return

    # Load and validate config
    config = load_config()

    missing = []
    if not config.get("github_token"):
        missing.append("github_token")
    if not config.get("azure_resource"):
        missing.append("azure_resource")
    if not config.get("azure_api_key"):
        missing.append("azure_api_key")

    if missing:
        console.print(f"[red]Missing configuration: {', '.join(missing)}[/red]")
        console.print("Run [bold]python repo_posh.py --setup[/bold] or set environment variables")
        return

    run_posh(
        config,
        dry_run=args.dry_run,
        force=args.force,
        repos_filter=args.repos,
        include_readme=args.with_readme,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
