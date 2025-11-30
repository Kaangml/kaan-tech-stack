#!/usr/bin/env python3
"""
Knowledge Base Directory Structure Generator
Creates nested folder structure with README.md files, .gitignore, and main README.
"""

import os
from pathlib import Path

# Define the folder structure as a nested dictionary
FOLDER_STRUCTURE = {
    "0-foundations": {},
    "1-software-architecture": {},
    "2-data-engineering": {
        "geospatial-analysis": {},
        "etl-pipelines": {
            "scraping-tools": {
                "beautifulsoup": {},
                "lxml": {},
                "requests-httpx": {},
            },
            "pipeline-patterns": {},
        },
        "db-optimization": {},
        "architecture": {},
    },
    "3-ai-and-agents": {
        "llm-agents": {},
        "rag-systems": {},
        "computer-vision": {},
        "ml-ops": {},
    },
    "4-automation-workflows": {
        "n8n-workflows": {},
        "event-driven-systems": {},
        "browser-automation": {
            "selenium": {},
            "playwright": {},
        },
    },
    "5-python-ecosystem": {
        "web-frameworks": {},
        "data-science-stack": {},
        "visualization": {},
        "pdf-processing": {},
        "testing": {},
    },
    "6-databases": {
        "relational": {},
        "graph": {},
        "vector": {},
    },
    "7-devops-cloud": {
        "containerization": {},
        "aws-serverless": {},
    },
    "99-blueprints-and-recipes": {
        "autonomous-browser-agent-architecture": {},
        "legal-rag-system": {},
        "scalable-scraping-architecture": {},
        "scalable-data-pipelines": {},
    },
}

# Professional .gitignore content
GITIGNORE_CONTENT = """# =============================================================================
# Python
# =============================================================================

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# =============================================================================
# OS Generated Files
# =============================================================================

# macOS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
.AppleDouble
.LSOverride
Icon?
.DocumentRevisions-V100
.fseventsd
.TemporaryItems
.VolumeIcon.icns
.com.apple.timemachine.donotpresent

# Windows
Desktop.ini
$RECYCLE.BIN/
*.lnk

# Linux
*~
.fuse_hidden*
.directory
.Trash-*
.nfs*

# =============================================================================
# IDE / Editor Files
# =============================================================================

# VS Code
.vscode/
*.code-workspace

# JetBrains IDEs (PyCharm, IntelliJ, etc.)
.idea/
*.iml
*.iws
*.ipr
out/
.idea_modules/

# Sublime Text
*.sublime-project
*.sublime-workspace

# Vim
*.swp
*.swo
*.swn
*~
Session.vim
.netrwhist

# Emacs
*~
\\#*\\#
/.emacs.desktop
/.emacs.desktop.lock
*.elc
auto-save-list
tramp
.\\#*

# =============================================================================
# Project Specific
# =============================================================================

# Local configuration files
*.local
.env.local
.env.*.local
config.local.py
secrets.py
secrets/

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Database files
*.db
*.sqlite
*.sqlite3

# Temporary files
tmp/
temp/
*.tmp
*.temp
*.bak

# Data files (add exceptions as needed)
data/
!data/.gitkeep

# Model files
*.pkl
*.pickle
*.h5
*.hdf5
*.pt
*.pth
*.onnx
models/
!models/.gitkeep

# Credentials and secrets
*.pem
*.key
*.crt
credentials/
"""

# Main README content
MAIN_README_CONTENT = """# 📚 Knowledge Base

> Personal knowledge repository for software engineering, DevOps, AI/ML, and data engineering practices.

## 📁 Repository Structure

```
.
├── 0-foundations/              # Core programming concepts & fundamentals
├── 1-software-architecture/    # Design patterns, principles, system design
├── 2-data-engineering/         # ETL, pipelines, geospatial, optimization
│   ├── geospatial-analysis/
│   ├── etl-pipelines/
│   │   ├── scraping-tools/
│   │   └── pipeline-patterns/
│   ├── db-optimization/
│   └── architecture/
├── 3-ai-and-agents/            # LLMs, RAG, CV, MLOps
│   ├── llm-agents/
│   ├── rag-systems/
│   ├── computer-vision/
│   └── ml-ops/
├── 4-automation-workflows/     # n8n, event-driven, browser automation
│   ├── n8n-workflows/
│   ├── event-driven-systems/
│   └── browser-automation/
├── 5-python-ecosystem/         # Frameworks, data science, visualization
│   ├── web-frameworks/
│   ├── data-science-stack/
│   ├── visualization/
│   ├── pdf-processing/
│   └── testing/
├── 6-databases/                # SQL, Graph, Vector databases
│   ├── relational/
│   ├── graph/
│   └── vector/
├── 7-devops-cloud/             # Containers, AWS, serverless
│   ├── containerization/
│   └── aws-serverless/
└── 99-blueprints-and-recipes/  # Production-ready architectures
    ├── autonomous-browser-agent-architecture/
    ├── legal-rag-system/
    ├── scalable-scraping-architecture/
    └── scalable-data-pipelines/
```

## 🚀 Quick Navigation

| Category | Description |
|----------|-------------|
| **Foundations** | Core CS concepts, algorithms, data structures |
| **Architecture** | Design patterns, microservices, clean architecture |
| **Data Engineering** | ETL pipelines, scraping, geospatial analysis |
| **AI & Agents** | LLM agents, RAG systems, computer vision, MLOps |
| **Automation** | Workflow automation, event-driven systems |
| **Python Ecosystem** | Web frameworks, data science, testing |
| **Databases** | Relational, graph, and vector databases |
| **DevOps & Cloud** | Docker, Kubernetes, AWS serverless |
| **Blueprints** | Production-ready architecture templates |

## 📝 Contributing Guidelines

1. Each topic should have its own README.md with clear documentation
2. Include code examples where applicable
3. Add references and resources at the end of each document
4. Use consistent naming conventions (kebab-case for folders)

## 📜 License

This is a personal knowledge base. All content is for educational purposes.

---

*Last updated: {date}*
"""


def create_folder_structure(base_path: Path, structure: dict) -> int:
    """
    Recursively create folder structure with README.md files.
    Returns the count of created folders.
    """
    count = 0
    for folder_name, subfolders in structure.items():
        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        count += 1
        
        # Create README.md in each folder
        readme_path = folder_path / "README.md"
        if not readme_path.exists():
            readme_path.write_text(f"# {folder_name.replace('-', ' ').title()}\n\n> Documentation for {folder_name}\n")
        
        # Recursively create subfolders
        if subfolders:
            count += create_folder_structure(folder_path, subfolders)
    
    return count


def create_gitignore(base_path: Path) -> None:
    """Create .gitignore file in the base directory."""
    gitignore_path = base_path / ".gitignore"
    gitignore_path.write_text(GITIGNORE_CONTENT)
    print(f"✅ Created: {gitignore_path}")


def create_main_readme(base_path: Path) -> None:
    """Create main README.md in the base directory."""
    from datetime import datetime
    readme_path = base_path / "README.md"
    content = MAIN_README_CONTENT.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
    readme_path.write_text(content)
    print(f"✅ Created: {readme_path}")


def main():
    """Main entry point."""
    # Get the directory where the script is located
    base_path = Path(__file__).parent.resolve()
    
    print("=" * 60)
    print("🚀 Knowledge Base Structure Generator")
    print("=" * 60)
    print(f"\n📁 Base directory: {base_path}\n")
    
    # Create folder structure
    print("📂 Creating folder structure...")
    folder_count = create_folder_structure(base_path, FOLDER_STRUCTURE)
    print(f"✅ Created {folder_count} folders with README.md files\n")
    
    # Create .gitignore
    print("📄 Creating .gitignore...")
    create_gitignore(base_path)
    
    # Create main README
    print("\n📄 Creating main README.md...")
    create_main_readme(base_path)
    
    print("\n" + "=" * 60)
    print("✨ Structure created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
