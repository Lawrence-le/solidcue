# Skill

# Create Agent Skill

## Purpose

Guide the creation of a new runnable agent from a blank workspace or an
existing template.

## Source Materials

This workflow should rely on:

- `solidcue/skills/create-agent.md`
- `solidcue/skills/create-skill.md`
- `solidcue/tools/functions/create-agent.md`

## When To Use

Use this skill when the user wants to:

- create a new agent
- define a new agent key and name
- choose providers, models, and tools
- generate the default persona, skill, and tools files
- save the final agent under `solidcue/agents/<agent_key>/`

Do not use this skill for:

- executing an existing agent
- router-only clarification
- workspace repair that does not create a new agent

## Workflow

1. Confirm the target agent name, agent key, description, and intended use case.
2. Confirm provider selections for the brain, lite, reviewer, and writer roles.
3. Confirm the tool list and whether the agent needs optional writer support.
4. Validate the agent key is filesystem-safe and not already taken.
5. Build the agent configuration and persist it to the agent folder.
6. Generate or copy `PERSONA.md`, `SKILL.md`, and `TOOLS.md`.
7. Verify the final agent folder is complete and runnable.

## Required Fields

- agent name
- agent key
- description
- decision provider
- lite provider
- reviewer provider
- selected tools

## Validation Rules

- Never overwrite an existing agent.
- Never invent tools or capabilities that were not confirmed by the user.
- Keep the final agent key lowercase and filesystem-safe.
- If a provider or model is missing, ask before continuing.

## Output Contract

The completed agent must include:

- `solidcue/agents/<agent_key>/<agent_key>.yaml`
- `solidcue/agents/<agent_key>/PERSONA.md`
- `solidcue/agents/<agent_key>/SKILL.md`
- `solidcue/agents/<agent_key>/TOOLS.md`
