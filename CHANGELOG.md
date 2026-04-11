# Changelog

## [0.3.0] - 2026-04-10

### Added
- Ollama provider support — use local models with the same unified interface
- `estimate_cost()` now includes Gemma 4 / Ollama models (returns $0.00 for local inference)
- Token counting for Gemma 4 tokenizer (falls back to tiktoken cl100k for unsupported models)

### Changed
- `retry_with_backoff()` now respects `Retry-After` headers from Anthropic API
- Default streaming chunk size reduced from 1024 to 512 bytes for smoother UX on local models

### Fixed
- RAG pipeline: embedding cache now invalidated when source documents change (content hash check)
- Cost estimation for Claude Sonnet 4 pricing updated

## [0.2.0] - 2026-04-07

### Added
- Initial public release
- Prompt chaining with `Chain` class
- Token counting (tiktoken-based)
- Cost estimation for OpenAI and Anthropic models
- Retry handler with exponential backoff
- Streaming response utilities
- Basic RAG pipeline (embed + retrieve + generate)
