import typer
from rich import print

from solidcue.user.loader import get_user_profile_path, load_user_profile, save_user_profile
from solidcue.user.schema import UserProfileConfig


def register(app: typer.Typer) -> None:
    app.command("setup-init", rich_help_panel="Setup")(setup_init)
    app.command("setup-view", rich_help_panel="Setup")(setup_view)
    app.command("setup-update", rich_help_panel="Setup")(setup_update)


def _prompt_profile(default: UserProfileConfig) -> UserProfileConfig:
    location = typer.prompt("Location", default=default.location)
    timezone = typer.prompt("Timezone (optional)", default=default.timezone or "", show_default=False)
    display_name = typer.prompt("Display name (optional)", default=default.display_name or "", show_default=False)
    personality = typer.prompt("Personality (optional)", default=default.personality or "", show_default=False)
    job_title = typer.prompt("Job title (optional)", default=default.job_title or "", show_default=False)

    return UserProfileConfig(
        location=location,
        timezone=timezone or None,
        display_name=display_name or None,
        personality=personality or None,
        job_title=job_title or None,
        preferences=default.preferences,
    )


def setup_init() -> None:
    """Initialize user profile setup."""
    current = load_user_profile()
    profile = _prompt_profile(current)
    path = save_user_profile(profile)
    print(f"[green]Saved user profile:[/green] {path}")


def setup_view() -> None:
    """View current user profile setup."""
    profile = load_user_profile()
    path = get_user_profile_path()
    print(f"[bold]User profile[/bold] [dim]({path})[/dim]")
    print(f"[dim]Location:[/dim] {profile.location or '-'}")
    print(f"[dim]Timezone:[/dim] {profile.timezone or '-'}")
    print(f"[dim]Display Name:[/dim] {profile.display_name or '-'}")
    print(f"[dim]Personality:[/dim] {profile.personality or '-'}")
    print(f"[dim]Job Title:[/dim] {profile.job_title or '-'}")
    print(f"[dim]Preferences:[/dim] {profile.preferences if profile.preferences else '-'}")


def setup_update() -> None:
    """Update user profile setup."""
    current = load_user_profile()
    profile = _prompt_profile(current)
    path = save_user_profile(profile)
    print(f"[green]Updated user profile:[/green] {path}")
