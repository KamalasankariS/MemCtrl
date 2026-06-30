"""Gradio web interface for MemCtrl with BYOK support."""

import json

import gradio as gr
from memctrl import MemoryController
from memctrl.llm.backend import create_llm_backend


def create_app(
    user_id: str = "default",
    provider: str = "auto",
    api_key: str = None,
) -> gr.Blocks:
    state = {
        "controller": None,
        "provider": provider,
        "api_key": api_key,
        "user_id": user_id,
    }

    def _ensure_controller():
        if state["controller"] is None:
            llm = create_llm_backend(
                state["provider"],
                api_key=state["api_key"],
            )
            state["controller"] = MemoryController(
                user_id=state["user_id"], llm=llm,
            )
        return state["controller"]

    def connect_fn(new_provider, new_key):
        state["provider"] = new_provider or "auto"
        state["api_key"] = new_key or None
        state["controller"] = None
        try:
            ctrl = _ensure_controller()
            backend_name = ctrl.llm.provider_name
            return (
                f"Connected to {backend_name} backend. "
                "Ready to chat."
            )
        except Exception as e:
            state["controller"] = None
            return f"Connection failed: {e}"

    def chat_fn(message, history):
        ctrl = _ensure_controller()
        return ctrl.chat(message)

    def pin_fn(content, note):
        ctrl = _ensure_controller()
        result = ctrl.pin(content, note=note if note else None)
        return json.dumps(result, indent=2)

    def forget_fn(query):
        ctrl = _ensure_controller()
        result = ctrl.forget(query, confirm=False)
        return json.dumps(result, indent=2)

    def temporary_fn(content):
        ctrl = _ensure_controller()
        result = ctrl.temporary(content)
        return json.dumps(result, indent=2)

    def show_memory_fn(category):
        ctrl = _ensure_controller()
        memory = ctrl.show_memory(category=category)
        return json.dumps(memory, indent=2)

    def stats_fn():
        ctrl = _ensure_controller()
        stats = ctrl.get_stats()
        return json.dumps(stats, indent=2)

    with gr.Blocks(title="MemCtrl") as app:
        gr.Markdown("# MemCtrl - Task-Aware Memory Management")

        with gr.Tab("Settings"):
            gr.Markdown(
                "### LLM Configuration\n"
                "Enter your API key below. "
                "It is stored in memory only — never saved to disk."
            )
            with gr.Row():
                provider_select = gr.Dropdown(
                    choices=[
                        "auto", "anthropic", "openai",
                        "ollama", "echo",
                    ],
                    value=provider,
                    label="Provider",
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="sk-... or leave empty for Ollama/Echo",
                )
            connect_btn = gr.Button("Connect", variant="primary")
            connect_status = gr.Textbox(
                label="Status", interactive=False,
            )
            connect_btn.click(
                connect_fn,
                inputs=[provider_select, api_key_input],
                outputs=connect_status,
            )

        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat_fn)

        with gr.Tab("Memory Control"):
            with gr.Row():
                with gr.Column():
                    pin_input = gr.Textbox(label="Content to pin")
                    pin_note = gr.Textbox(
                        label="Note (optional)",
                    )
                    pin_btn = gr.Button("Pin")
                    pin_output = gr.JSON(label="Result")
                    pin_btn.click(
                        pin_fn,
                        inputs=[pin_input, pin_note],
                        outputs=pin_output,
                    )

                with gr.Column():
                    forget_input = gr.Textbox(
                        label="Forget query",
                    )
                    forget_btn = gr.Button("Forget")
                    forget_output = gr.JSON(label="Result")
                    forget_btn.click(
                        forget_fn,
                        inputs=[forget_input],
                        outputs=forget_output,
                    )

                with gr.Column():
                    temp_input = gr.Textbox(
                        label="Temporary content",
                    )
                    temp_btn = gr.Button("Add Temporary")
                    temp_output = gr.JSON(label="Result")
                    temp_btn.click(
                        temporary_fn,
                        inputs=[temp_input],
                        outputs=temp_output,
                    )

        with gr.Tab("Inspect"):
            with gr.Row():
                category_select = gr.Dropdown(
                    choices=[
                        "all", "pinned", "session", "ai_managed",
                    ],
                    value="all",
                    label="Category",
                )
                memory_btn = gr.Button("Show Memory")
            memory_output = gr.JSON(label="Memory")
            memory_btn.click(
                show_memory_fn,
                inputs=[category_select],
                outputs=memory_output,
            )

            stats_btn = gr.Button("Show Stats")
            stats_output = gr.JSON(label="Stats")
            stats_btn.click(stats_fn, outputs=stats_output)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
