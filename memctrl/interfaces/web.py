"""Gradio web interface for MemCtrl."""

import json

import gradio as gr
from memctrl import MemoryController
from memctrl.llm.backend import create_llm_backend


def create_app(user_id: str = "default", provider: str = "auto") -> gr.Blocks:
    llm = create_llm_backend(provider)
    controller = MemoryController(user_id=user_id, llm=llm)

    def chat_fn(message, history):
        response = controller.chat(message)
        return response

    def pin_fn(content, note):
        result = controller.pin(content, note=note if note else None)
        return json.dumps(result, indent=2)

    def forget_fn(query):
        result = controller.forget(query, confirm=False)
        return json.dumps(result, indent=2)

    def temporary_fn(content):
        result = controller.temporary(content)
        return json.dumps(result, indent=2)

    def show_memory_fn(category):
        memory = controller.show_memory(category=category)
        return json.dumps(memory, indent=2)

    def stats_fn():
        stats = controller.get_stats()
        return json.dumps(stats, indent=2)

    with gr.Blocks(title="MemCtrl") as app:
        gr.Markdown("# MemCtrl - Task-Aware Memory Management")

        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat_fn)

        with gr.Tab("Memory Control"):
            with gr.Row():
                with gr.Column():
                    pin_input = gr.Textbox(label="Content to pin")
                    pin_note = gr.Textbox(label="Note (optional)")
                    pin_btn = gr.Button("Pin")
                    pin_output = gr.JSON(label="Result")
                    pin_btn.click(pin_fn, inputs=[pin_input, pin_note], outputs=pin_output)

                with gr.Column():
                    forget_input = gr.Textbox(label="Forget query")
                    forget_btn = gr.Button("Forget")
                    forget_output = gr.JSON(label="Result")
                    forget_btn.click(forget_fn, inputs=[forget_input], outputs=forget_output)

                with gr.Column():
                    temp_input = gr.Textbox(label="Temporary content")
                    temp_btn = gr.Button("Add Temporary")
                    temp_output = gr.JSON(label="Result")
                    temp_btn.click(temporary_fn, inputs=[temp_input], outputs=temp_output)

        with gr.Tab("Inspect"):
            with gr.Row():
                category_select = gr.Dropdown(
                    choices=["all", "pinned", "session", "ai_managed"],
                    value="all",
                    label="Category",
                )
                memory_btn = gr.Button("Show Memory")
            memory_output = gr.JSON(label="Memory")
            memory_btn.click(show_memory_fn, inputs=[category_select], outputs=memory_output)

            stats_btn = gr.Button("Show Stats")
            stats_output = gr.JSON(label="Stats")
            stats_btn.click(stats_fn, outputs=stats_output)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
