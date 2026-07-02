"""Gradio web interface for MemCtrl with memory dashboard."""

import json

import gradio as gr
from memctrl import MemoryController
from memctrl.llm.backend import create_llm_backend


TASK_BADGES = {
    "medical": "🏥 Medical",
    "code": "💻 Code",
    "tutoring": "📚 Tutoring",
    "writing": "✍️ Writing",
    "general": "💬 General",
    "unclassified": "❓ Unclassified",
}


def _format_chunk_row(c):
    badge = TASK_BADGES.get(c["task_type"], c["task_type"])
    pin = "📌 " if c.get("is_pinned") else ""
    priority = c.get("priority", "?")
    tokens = c.get("tokens", "?")
    content = c.get("summary") or c.get("content", "")
    return (
        f"{pin}{badge} | Priority: {priority} | "
        f"{tokens} tok | {content}"
    )


def _format_trash_row(item):
    task = TASK_BADGES.get(item.get("task_type", ""), "")
    content = item["content"][:80] + "..." if len(item["content"]) > 80 else item["content"]
    return f"🗑️ {task} | Deleted: {item['deleted_at'][:16]} | {content}"


def _pressure_bar(pct):
    if pct < 50:
        color = "green"
    elif pct < 80:
        color = "orange"
    else:
        color = "red"
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return f"`[{bar}]` **{pct}%** (_{color}_)"


def _render_dashboard(ctrl):
    dash = ctrl.get_dashboard()
    mp = dash["memory_pressure"]

    lines = ["## Memory Pressure"]
    lines.append(
        f"**Active (Tier 0):** {_pressure_bar(mp['tier0_pct'])} "
        f"— {mp['tier0_tokens']}/{mp['tier0_max']} tokens"
    )
    lines.append(
        f"**Compressed (Tier 1):** {_pressure_bar(mp['tier1_pct'])} "
        f"— {mp['tier1_tokens']}/{mp['tier1_max']} tokens"
    )
    lines.append("")

    lines.append(f"## 📌 Pinned ({len(dash['pinned'])})")
    if dash["pinned"]:
        for c in dash["pinned"]:
            lines.append(f"- {_format_chunk_row(c)}")
    else:
        lines.append("_No pinned memories._")
    lines.append("")

    lines.append(f"## ⚡ Active ({len(dash['active'])})")
    if dash["active"]:
        for c in dash["active"]:
            lines.append(f"- {_format_chunk_row(c)}")
    else:
        lines.append("_No active chunks._")
    lines.append("")

    lines.append(f"## 📦 Compressed ({len(dash['compressed'])})")
    if dash["compressed"]:
        for c in dash["compressed"]:
            lines.append(f"- {_format_chunk_row(c)}")
    else:
        lines.append("_No compressed chunks._")
    lines.append("")

    lines.append(f"## 🗑️ Trash ({len(dash['trash'])})")
    if dash["trash"]:
        for item in dash["trash"][-5:]:
            lines.append(f"- {_format_trash_row(item)}")
        if len(dash["trash"]) > 5:
            lines.append(f"_...and {len(dash['trash']) - 5} more_")
    else:
        lines.append("_Trash is empty._")

    return "\n".join(lines)


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
            return f"Connected to **{backend_name}** backend. Ready to chat."
        except Exception as e:
            state["controller"] = None
            return f"Connection failed: {e}"

    def chat_fn(message, history):
        ctrl = _ensure_controller()
        response = ctrl.chat(message)
        return response

    def refresh_dashboard():
        ctrl = _ensure_controller()
        return _render_dashboard(ctrl)

    def chat_and_refresh(message, history):
        response = chat_fn(message, history)
        dashboard = refresh_dashboard()
        return response, dashboard

    def pin_fn(content, note):
        ctrl = _ensure_controller()
        result = ctrl.pin(content, note=note if note else None)
        dashboard = _render_dashboard(ctrl)
        return json.dumps(result, indent=2), dashboard

    def forget_fn(query):
        ctrl = _ensure_controller()
        result = ctrl.forget(query, confirm=False)
        dashboard = _render_dashboard(ctrl)
        return json.dumps(result, indent=2), dashboard

    def unpin_fn(chunk_id):
        ctrl = _ensure_controller()
        result = ctrl.unpin(chunk_id.strip())
        dashboard = _render_dashboard(ctrl)
        return json.dumps(result, indent=2), dashboard

    def restore_fn(chunk_id):
        ctrl = _ensure_controller()
        result = ctrl.restore_from_trash(chunk_id.strip())
        dashboard = _render_dashboard(ctrl)
        return json.dumps(result, indent=2), dashboard

    def temporary_fn(content):
        ctrl = _ensure_controller()
        result = ctrl.temporary(content)
        dashboard = _render_dashboard(ctrl)
        return json.dumps(result, indent=2), dashboard

    def export_fn():
        ctrl = _ensure_controller()
        return ctrl.export_data(format="json")

    with gr.Blocks(
        title="MemCtrl",
    ) as app:
        gr.Markdown(
            "# MemCtrl — Task-Aware Memory Management\n"
            "Control what your LLM remembers. Pin, forget, compress, restore."
        )

        with gr.Row():
            # Left: main interaction area
            with gr.Column(scale=3):
                with gr.Tab("Settings"):
                    gr.Markdown(
                        "### LLM Configuration\n"
                        "Your API key is stored in memory only — "
                        "never saved to disk."
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
                            placeholder="sk-... (leave empty for Ollama/Echo)",
                        )
                    connect_btn = gr.Button("Connect", variant="primary")
                    connect_status = gr.Markdown("")
                    connect_btn.click(
                        connect_fn,
                        inputs=[provider_select, api_key_input],
                        outputs=connect_status,
                    )

                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(height=400)
                    msg_input = gr.Textbox(
                        placeholder="Type a message...",
                        show_label=False,
                    )

                    def user_msg(message, history):
                        history = history or []
                        history.append({"role": "user", "content": message})
                        return "", history

                    def bot_respond(history):
                        ctrl = _ensure_controller()
                        user_message = history[-1]["content"]
                        response = ctrl.chat(user_message)
                        history.append({"role": "assistant", "content": response})
                        dashboard = _render_dashboard(ctrl)
                        return history, dashboard

                    msg_input.submit(
                        user_msg, [msg_input, chatbot], [msg_input, chatbot],
                    ).then(
                        bot_respond, [chatbot], [chatbot, gr.Markdown()],
                    )

                with gr.Tab("Memory Control"):
                    gr.Markdown("### Pin, Forget, Restore")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Pin to permanent memory**")
                            pin_input = gr.Textbox(
                                label="Content",
                                placeholder="Something to always remember...",
                            )
                            pin_note = gr.Textbox(
                                label="Note (optional)",
                                placeholder="Why is this important?",
                            )
                            pin_btn = gr.Button("📌 Pin", variant="primary")
                            pin_output = gr.JSON(label="Result")

                        with gr.Column():
                            gr.Markdown("**Forget memories**")
                            forget_input = gr.Textbox(
                                label="Search query",
                                placeholder="What to forget...",
                            )
                            forget_btn = gr.Button("🗑️ Forget")
                            forget_output = gr.JSON(label="Result")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Unpin a chunk**")
                            unpin_input = gr.Textbox(
                                label="Chunk ID",
                                placeholder="Paste chunk ID...",
                            )
                            unpin_btn = gr.Button("📌 Unpin")
                            unpin_output = gr.JSON(label="Result")

                        with gr.Column():
                            gr.Markdown("**Restore from trash**")
                            restore_input = gr.Textbox(
                                label="Chunk ID",
                                placeholder="Paste chunk ID from trash...",
                            )
                            restore_btn = gr.Button("♻️ Restore")
                            restore_output = gr.JSON(label="Result")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Add temporary memory**")
                            temp_input = gr.Textbox(
                                label="Content",
                                placeholder="Session-only memory...",
                            )
                            temp_btn = gr.Button("⏳ Add Temporary")
                            temp_output = gr.JSON(label="Result")

                        with gr.Column():
                            gr.Markdown("**Export all data**")
                            export_btn = gr.Button("📥 Export JSON")
                            export_output = gr.Code(
                                label="Export", language="json",
                            )

            # Right: memory dashboard sidebar
            with gr.Column(scale=2):
                gr.Markdown("## Memory Dashboard")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                dashboard_md = gr.Markdown(
                    "_Connect to a provider to see memory state._",
                    elem_classes=["dashboard"],
                )
                refresh_btn.click(refresh_dashboard, outputs=dashboard_md)

        # Wire memory control buttons to update dashboard
        pin_btn.click(
            pin_fn, [pin_input, pin_note], [pin_output, dashboard_md],
        )
        forget_btn.click(
            forget_fn, [forget_input], [forget_output, dashboard_md],
        )
        unpin_btn.click(
            unpin_fn, [unpin_input], [unpin_output, dashboard_md],
        )
        restore_btn.click(
            restore_fn, [restore_input], [restore_output, dashboard_md],
        )
        temp_btn.click(
            temporary_fn, [temp_input], [temp_output, dashboard_md],
        )
        export_btn.click(export_fn, outputs=export_output)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
