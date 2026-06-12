# Skill

# User Profile Skill

## Purpose

Guide the creation and update of the workspace user profile before any agent
is selected or executed.

## When To Use

Use this skill when the user wants to:

- create a new workspace profile
- update location or timezone
- set a display name
- define personality or preferences
- configure the router provider

Do not use this skill for:

- creating a runnable agent
- executing an existing agent
- task-specific resume or job workflows

## Workflow

1. Confirm whether this is a new profile or an update to an existing profile.
2. Collect the profile fields that matter for the workspace setup.
3. Validate provider details if router configuration is being changed.
4. Persist the profile data.
5. Verify the saved profile can be loaded successfully.

## Required / Optional Fields

Typical profile fields include:

- location
- timezone
- display_name
- personality
- preferences
- router_provider

## Validation Rules

- Never invent a provider type, API key, or model.
- If router_provider is provided, make sure the provider type is supported.
- Ask for missing fields only when they are needed for the requested update.
- Keep the profile concise and workspace-focused.

## Output Contract

The completed profile should be stored under the workspace profile storage
used by the app and remain compatible with:

- `solidcue.user.loader.load_user_profile()`
- `solidcue.user.loader.save_user_profile()`
