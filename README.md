<p align="center">
  <h1>SolidCue - Config-Driven AI Agent Orchestration CLI</h1>
</p>

![License](https://img.shields.io/badge/License-MIT-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-0EA5E9)
![Arize Phoenix](https://img.shields.io/badge/Observability-Arize%20Phoenix-16A34A)
![MCP](https://img.shields.io/badge/Tools-MCP-111827)
![Python](https://img.shields.io/badge/Language-Python%203.12%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-2E3440)
![Typer](https://img.shields.io/badge/CLI-Typer-0A9396)
![Rich](https://img.shields.io/badge/Terminal-Rich-6C63FF)
![Pydantic](https://img.shields.io/badge/Validation-Pydantic-E92063)
![YAML](https://img.shields.io/badge/Config-YAML-CB171E?logo=yaml&logoColor=white)

SolidCue is a Python CLI for building and running config-driven AI agents with
LangGraph. It uses YAML configuration to define agents, tools, MCP servers,
provider settings, and user context, then executes a multi-step workflow from
the terminal. Arize Phoenix tracing is supported for observability.

## Keywords

AI agents, agent orchestration, LangGraph, Arize Phoenix, Model Context Protocol,
MCP, AI CLI, LLM tools, OpenAI-compatible API, Anthropic Claude, OpenRouter,
YAML agent configuration, Python agent framework.

## Why SolidCue

- Build agents without hardcoding orchestration logic per use case
- Keep agent behavior inspectable through YAML and deterministic workflow stages
- Connect external tools through MCP or direct API tool configuration
- Trace full LangGraph runs in Arize Phoenix when debugging or evaluating behavior

## Architecture Overview

SolidCue runs a LangGraph-based workflow with explicit stages:

1. Decision: decide whether to answer directly or call tools
2. Execution: invoke selected tools (MCP/API/RAG placeholder)
3. Reflection: analyze tool results and identify gaps
4. Validation: check response quality and completeness
5. Synthesis: assemble final response
6. Final Output: return the response to the CLI

This structure keeps orchestration explicit, testable, and maintainable.

## Core Features

- Interactive terminal UX using Typer, Rich, and InquirerPy
- Config-driven agent setup via YAML files
- MCP server registration and MCP tool discovery
- Direct HTTP API tool configuration
- Placeholder RAG tool configuration for retrieval workflows
- Provider support:
  - OpenAI-compatible APIs
  - Anthropic
  - OpenRouter
- User profile management (location, timezone, display name, preferences)
- Debug mode for introspecting prompts, decisions, and validation flow
- Optional Arize Phoenix tracing for LangGraph runs

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

Install `uv` if needed:

```bash
pip install uv
```

## Installation

From project root:

```bash
uv sync
```

## Quick Start

1. Initialize setup:

```bash
uv run cli setup-init
```

2. Add tools:

```bash
uv run cli create-mcp-server
uv run cli create-tool
```

3. Create an agent:

```bash
uv run cli create-agent
```

4. Run the agent:

```bash
uv run cli run-agent
```

## CLI Usage

Show help:

```bash
uv run cli --help
```

### Setup Commands

```bash
uv run cli setup-init
uv run cli setup-view
uv run cli setup-update
```

User profile config path:

- `solidcue/user/configs/user_profile.yaml`

### Tooling Commands

```bash
uv run cli create-mcp-server
uv run cli list-mcp-servers
uv run cli create-tool
uv run cli list-tools
```

Config paths:

- MCP servers: `solidcue/tools/configs/mcp_servers/`
- Tools: `solidcue/tools/configs/tools/`

Supported tool types:

- `mcp`: discovered from a configured MCP server
- `api`: direct HTTP API tool
- `rag`: placeholder retrieval tool config

### Agent Commands

```bash
uv run cli create-agent
uv run cli list-agents
uv run cli run-agent
uv run cli run-agent --debug
```

Agent config path:

- `solidcue/agents/configs/`

### Environment File Behavior

When creating an agent, provider API keys are written to `.env` by default.
Override with:

```bash
SOLIDCUE_ENV_PATH=.env.local uv run cli create-agent
```

## Tracing (LangSmith + Arize Phoenix)

Start Phoenix locally:

```bash
uv run --with arize-phoenix phoenix serve
```

Enable Phoenix export:

```bash
export PHOENIX_ENABLED=true
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
export PHOENIX_SERVICE_NAME=solidcue
```

Control providers with SolidCue switches:

```bash
# Phoenix on/off
export SOLIDCUE_PHOENIX_ENABLED=true

# LangSmith on/off
export SOLIDCUE_LANGSMITH_ENABLED=false
```

Common setups:

```bash
# Phoenix only
export SOLIDCUE_PHOENIX_ENABLED=true
export SOLIDCUE_LANGSMITH_ENABLED=false

# LangSmith only
export SOLIDCUE_PHOENIX_ENABLED=false
export SOLIDCUE_LANGSMITH_ENABLED=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_PROJECT=solidcue

# Both enabled
export SOLIDCUE_PHOENIX_ENABLED=true
export SOLIDCUE_LANGSMITH_ENABLED=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_PROJECT=solidcue

# Disable all tracing
export SOLIDCUE_PHOENIX_ENABLED=false
export SOLIDCUE_LANGSMITH_ENABLED=false
```

Run as usual:

```bash
uv run cli run-agent
```

For a custom collector or hosted Phoenix endpoint, set:

```bash
export PHOENIX_COLLECTOR_ENDPOINT=https://your-phoenix-endpoint/v1/traces
```

## Optional Global `solidcue` Command (macOS/zsh)

```bash
./scripts/install-solidcue-cli.sh
source ~/.zshrc
source ~/.zprofile
solidcue --help
```

This creates `bin/solidcue` and adds the project `bin/` directory to your shell
path.

After install, use:

```bash
solidcue setup-init
solidcue create-agent
solidcue run-agent
```

## Project Structure

```text
solidcue/
  app/                 Typer CLI commands and CLI helpers
  agents/              Agent schemas, registry, loader, and YAML configs
  core/                LangGraph orchestration, state, nodes, and execution
  memory/              Memory package placeholder
  prompts/             Decision, reflection, and synthesis prompts
  providers/           Provider adapters and provider resolution
  services/            Application services used by CLI commands
  storage/             Storage package placeholder
  tasks/               Task package placeholder
  tools/               Tool schemas, loader, registry, MCP client, and configs
  user/                User profile schema, loader, and config
  utils/               Shared utilities
tests/                 Pytest test suite
scripts/               Local setup/install scripts
bin/                   Optional CLI wrapper
```

## Development

Run all tests:

```bash
uv run pytest
```

Run a specific file:

```bash
uv run pytest tests/test_orchestrator.py
```

## Notes

- Agent, MCP server, and tool keys are generated from display names
- Existing agent API key entries in `.env` are not overwritten
- RAG tools currently generate a basic placeholder config
- Use `--debug` to inspect decision, tool, validation, and prompt payloads
