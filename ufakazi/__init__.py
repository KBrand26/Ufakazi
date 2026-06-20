"""Ufakazi: investigating language-conditioned truthiness bias in LLMs.

See DESIGN.md for the experimental design and CLAUDE.md for architecture.
"""

from dotenv import load_dotenv

# Load .env (gitignored) at import so every entrypoint picks up provider keys, e.g.
# OPENROUTER_API_KEY. Inspect only auto-loads .env via its own CLI, not the Python API.
load_dotenv()
