"""Click-based CLI for MemCtrl."""

import json
import click
from memctrl import MemoryController


@click.group()
@click.option("--user", "-u", default="default", help="User ID")
@click.option(
    "--provider", "-p", default="auto",
    help="LLM provider: auto, anthropic, huggingface, echo",
)
@click.pass_context
def cli(ctx, user, provider):
    """MemCtrl - Task-Aware Memory Management for LLMs"""
    ctx.ensure_object(dict)
    ctx.obj["user"] = user
    ctx.obj["provider"] = provider


def _get_controller(ctx) -> MemoryController:
    from memctrl.llm.backend import create_llm_backend
    llm = create_llm_backend(ctx.obj["provider"])
    return MemoryController(user_id=ctx.obj["user"], llm=llm)


@cli.command()
@click.argument("message")
@click.pass_context
def chat(ctx, message):
    """Send a message and get a response."""
    controller = _get_controller(ctx)
    response = controller.chat(message)
    click.echo(response)
    controller.close_session()


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start an interactive chat session."""
    controller = _get_controller(ctx)
    click.echo("MemCtrl interactive mode. Type 'quit' to exit.")

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.startswith("/pin "):
            result = controller.pin(user_input[5:])
            click.echo(f"Pinned: {result['chunk_id']}")
            continue

        if user_input.startswith("/forget "):
            result = controller.forget(user_input[8:], confirm=False)
            click.echo(f"Forgot {result.get('num_deleted', 0)} chunks")
            continue

        if user_input == "/memory":
            memory = controller.show_memory()
            click.echo(json.dumps(memory, indent=2))
            continue

        if user_input == "/stats":
            stats = controller.get_stats()
            click.echo(json.dumps(stats, indent=2))
            continue

        response = controller.chat(user_input)
        click.echo(f"Assistant> {response}")

    controller.close_session()
    click.echo("Session closed.")


@cli.command()
@click.argument("content")
@click.option("--note", "-n", default=None, help="Optional note")
@click.pass_context
def pin(ctx, content, note):
    """Pin content to permanent memory."""
    controller = _get_controller(ctx)
    result = controller.pin(content, note=note)
    click.echo(f"Pinned: {result['chunk_id']}")
    controller.close_session()


@cli.command()
@click.argument("query")
@click.pass_context
def forget(ctx, query):
    """Forget matching content."""
    controller = _get_controller(ctx)
    result = controller.forget(query, confirm=False)
    click.echo(f"Forgot {result.get('num_deleted', 0)} chunks")
    controller.close_session()


@cli.command()
@click.option("--category", "-c", default="all", help="Category: all, pinned, session, ai_managed")
@click.pass_context
def show(ctx, category):
    """Show stored memory."""
    controller = _get_controller(ctx)
    memory = controller.show_memory(category=category)
    click.echo(json.dumps(memory, indent=2))
    controller.close_session()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory statistics."""
    controller = _get_controller(ctx)
    stats_data = controller.get_stats()
    click.echo(json.dumps(stats_data, indent=2))
    controller.close_session()


@cli.command(name="export")
@click.option("--format", "-f", "fmt", default="json", help="Export format: json, text")
@click.pass_context
def export_cmd(ctx, fmt):
    """Export user data."""
    controller = _get_controller(ctx)
    output = controller.export_data(format=fmt)
    click.echo(output)
    controller.close_session()


@cli.command()
@click.option("--port", default=7860, help="Port for Gradio server")
@click.pass_context
def start(ctx, port):
    """Launch the Gradio web interface."""
    from memctrl.interfaces.web import create_app
    app = create_app(user_id=ctx.obj["user"], provider=ctx.obj["provider"])
    app.launch(server_port=port)


if __name__ == "__main__":
    cli()
