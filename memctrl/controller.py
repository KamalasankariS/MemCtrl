import json
import logging
import re
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime, timedelta

from .models import Chunk, Session, User, ChunkType, ChunkPriority
from .core.tiers import TierManager, compute_task_aware_priority
from .config import get_config
from .tokenizer import count_tokens
from .llm.backend import LLMBackend, create_llm_backend

logger = logging.getLogger(__name__)


class MemoryController:
    """Primary API for MemCtrl memory management."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        control_mode: str = "hybrid",
        config_path: Optional[str] = None,
        llm: Optional[LLMBackend] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if config_path:
            from .config import MemCtrlConfig, set_config

            config = MemCtrlConfig.from_yaml(config_path)
            set_config(config)

        self.config = get_config()
        self.control_mode = control_mode

        if llm:
            self.llm = llm
        else:
            p = provider or self.config.llm_provider
            self.llm = create_llm_backend(p, api_key=api_key)

        self.tier_manager = TierManager(llm=self.llm)
        self.user_id = user_id or str(uuid4())
        self.user = self._load_or_create_user(self.user_id)
        self.current_session: Optional[Session] = None
        # Load persisted trash and audit log from SQLite
        self.trash: List[Dict[str, Any]] = self.tier_manager.tier2.store.get_trash(self.user_id)
        self.audit_log: List[Dict[str, Any]] = self.tier_manager.tier2.store.get_audit_log(self.user_id)

    def _load_or_create_user(self, user_id: str) -> User:
        user = self.tier_manager.tier2.store.retrieve_user(user_id)
        if not user:
            user = User(id=user_id)
            self.tier_manager.tier2.store.store_user(user)
        return user

    def _get_or_create_session(self) -> Session:
        if self.current_session and self.current_session.is_active:
            return self.current_session

        session = Session(id=str(uuid4()), user_id=self.user_id)
        self.current_session = session
        self.tier_manager.tier2.store.store_session(session)
        return session

    def _create_chunk(self, content: str, chunk_type: ChunkType = ChunkType.CONVERSATION) -> Chunk:
        tokens = count_tokens(content, self.config.tokenizer_model)
        return Chunk(
            id=str(uuid4()),
            content=content,
            tokens=tokens,
            chunk_type=chunk_type,
            timestamp=datetime.now(),
        )

    def _log_action(self, action: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": self.user_id,
            "details": details,
        }
        self.audit_log.append(entry)
        self.tier_manager.tier2.store.store_audit_entry(entry)

    def _build_context_messages(self, query: str) -> List[Dict[str, str]]:
        """Build LLM messages from pinned memory, recent context, and the current query."""
        messages: List[Dict[str, str]] = []

        # System prompt with pinned memories
        pinned = self.tier_manager.tier2.get_pinned(self.user_id)
        if pinned:
            pinned_text = "\n".join(f"- {c.content}" for c in pinned)
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "The user has pinned the following "
                    f"information:\n{pinned_text}"
                ),
            })
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})

        # Recent conversation from current session
        if self.current_session:
            for chunk in self.current_session.get_recent_chunks(20):
                if chunk.content.startswith("User: "):
                    messages.append({"role": "user", "content": chunk.content[6:]})
                elif chunk.content.startswith("Assistant: "):
                    messages.append({"role": "assistant", "content": chunk.content[11:]})

        # Relevant past context via semantic search
        relevant = self.tier_manager.tier2.search(query, user_id=self.user_id, limit=5)
        if relevant:
            context_text = "\n".join(
                f"- {c.content}" for c in relevant
                if not c.content.startswith("User: ")
            )
            if context_text.strip():
                system_msg = messages[0]
                system_msg["content"] += f"\n\nRelevant past context:\n{context_text}"

        messages.append({"role": "user", "content": query})
        return messages

    # -- SDK API (Option 3: Context Budget Optimizer) --

    def add_message(self, role: str, content: str) -> Dict[str, Any]:
        """Record a message into the memory system.

        Call this for every user message and every assistant response.
        MemCtrl will chunk it, classify its task type, and manage its
        lifecycle across tiers automatically.

        Args:
            role: "user", "assistant", or "system"
            content: The message text

        Returns:
            Dict with chunk_id, task_type, tokens, and tier placement.
        """
        session = self._get_or_create_session()
        prefix = {"user": "User: ", "assistant": "Assistant: ", "system": "System: "}
        chunk = self._create_chunk(f"{prefix.get(role, '')}{content}")
        self.tier_manager.add_chunk(chunk, user_id=self.user_id, session_id=session.id)
        session.add_chunk(chunk)

        if self.tier_manager.tier0.is_full():
            self._handle_memory_pressure()

        self._log_action("add_message", {
            "role": role, "chunk_id": chunk.id, "session_id": session.id,
        })

        return {
            "chunk_id": chunk.id,
            "task_type": chunk.task_type,
            "tokens": chunk.tokens,
            "tier": "active" if self.tier_manager.tier0.get(chunk.id) else "compressed",
        }

    def optimize(self, max_tokens: int = 4096) -> List[Dict[str, str]]:
        """Build an optimized message list that fits within a token budget.

        This is the core SDK method. Call it before every LLM API request.
        It returns a standard messages list (system/user/assistant dicts)
        that fits within max_tokens by:
          1. Always including pinned memories in the system prompt
          2. Including recent messages in full (newest first)
          3. Replacing older messages with compressed summaries
          4. Dropping lowest-priority messages when budget is exceeded

        Args:
            max_tokens: Maximum total tokens for the returned messages.

        Returns:
            List of {"role": ..., "content": ...} dicts ready for any LLM API.
        """
        messages: List[Dict[str, str]] = []
        budget = max_tokens

        # 1. System prompt with pinned memories (always included)
        pinned = self.tier_manager.tier2.get_pinned(self.user_id)
        system_parts = ["You are a helpful assistant."]
        if pinned:
            pinned_text = "\n".join(f"- {c.content}" for c in pinned)
            system_parts.append(f"The user has pinned the following information:\n{pinned_text}")

        system_content = " ".join(system_parts)
        system_tokens = count_tokens(system_content, self.config.tokenizer_model)
        budget -= system_tokens

        # 2. Gather all conversation chunks, scored by priority
        tier0_chunks = self.tier_manager.tier0.get_all()
        tier1_chunks = self.tier_manager.tier1.get_all()

        conversation_chunks = []
        for c in tier0_chunks:
            if not c.is_pinned:
                conversation_chunks.append(("full", c))
        for c in tier1_chunks:
            conversation_chunks.append(("compressed", c))

        # Sort by timestamp (oldest first for final output)
        conversation_chunks.sort(key=lambda x: x[1].timestamp)

        # 3. Fill budget: newest messages get full content, older get summaries
        selected: List[Dict[str, str]] = []
        tokens_used = 0

        # Process newest first to prioritize recent context
        for source, chunk in reversed(conversation_chunks):
            if source == "full":
                text = chunk.content
                tok = chunk.tokens
            else:
                text = chunk.summary or chunk.content
                tok = count_tokens(text, self.config.tokenizer_model)

            if tokens_used + tok > budget:
                # Try compressed version of full chunks
                if source == "full" and chunk.summary:
                    text = chunk.summary
                    tok = count_tokens(text, self.config.tokenizer_model)
                    if tokens_used + tok > budget:
                        continue
                else:
                    continue

            # Parse role from content prefix
            if text.startswith("User: "):
                selected.append({"role": "user", "content": text[6:], "_ts": chunk.timestamp})
            elif text.startswith("Assistant: "):
                selected.append({"role": "assistant", "content": text[11:], "_ts": chunk.timestamp})
            elif text.startswith("System: "):
                selected.append({"role": "system", "content": text[8:], "_ts": chunk.timestamp})
            else:
                selected.append({"role": "user", "content": text, "_ts": chunk.timestamp})

            tokens_used += tok

        # 4. Relevant past context via semantic search (if budget allows)
        remaining = budget - tokens_used
        if remaining > 100 and self.current_session:
            recent_chunks = self.current_session.get_recent_chunks(3)
            if recent_chunks:
                latest_content = recent_chunks[0].content
                if latest_content.startswith("User: "):
                    latest_content = latest_content[6:]
                relevant = self.tier_manager.tier2.search(
                    latest_content, user_id=self.user_id, limit=3,
                )
                context_parts = []
                context_tokens = 0
                for c in relevant:
                    if not c.content.startswith("User: ") and not c.content.startswith("Assistant: "):
                        text = c.summary or c.content
                        tok = count_tokens(text, self.config.tokenizer_model)
                        if context_tokens + tok <= remaining - 20:
                            context_parts.append(f"- {text}")
                            context_tokens += tok

                if context_parts:
                    system_content += "\n\nRelevant past context:\n" + "\n".join(context_parts)

        # 5. Assemble final messages
        messages.append({"role": "system", "content": system_content})

        # Sort selected back to chronological order and strip internal _ts
        selected.sort(key=lambda m: m.get("_ts", datetime.min))
        for msg in selected:
            msg.pop("_ts", None)
            messages.append(msg)

        self._log_action("optimize", {
            "max_tokens": max_tokens,
            "messages_returned": len(messages),
            "tokens_used": system_tokens + tokens_used,
        })

        return messages

    # -- Chat API (used by Gradio UI) --

    def chat(self, query: str) -> str:
        session = self._get_or_create_session()

        query_chunk = self._create_chunk(f"User: {query}")
        self.tier_manager.add_chunk(query_chunk, user_id=self.user_id, session_id=session.id)
        session.add_chunk(query_chunk)

        messages = self._build_context_messages(query)
        response = self.llm.generate(messages, max_tokens=self.config.max_tokens_per_chunk)

        response_chunk = self._create_chunk(f"Assistant: {response}")
        self.tier_manager.add_chunk(response_chunk, user_id=self.user_id, session_id=session.id)
        session.add_chunk(response_chunk)

        self.tier_manager.tier2.store.store_session(session)

        if self.tier_manager.tier0.is_full():
            self._handle_memory_pressure()

        self._log_action("chat", {"query": query, "session_id": session.id})
        return response

    def pin(self, content: str, note: Optional[str] = None) -> Dict[str, Any]:
        session = self._get_or_create_session()

        chunk = self._create_chunk(content)
        chunk.is_pinned = True
        chunk.priority = ChunkPriority.USER_PINNED

        if note:
            chunk.metadata["user_note"] = note

        self.tier_manager.add_chunk(chunk, user_id=self.user_id, session_id=session.id)
        self.user.pin_chunk(chunk.id)
        self.tier_manager.tier2.store.store_user(self.user)

        self._log_action("pin", {"chunk_id": chunk.id, "content": content, "note": note})

        return {"success": True, "chunk_id": chunk.id, "message": "Pinned to permanent memory"}

    def forget(self, query: str, confirm: bool = True) -> Dict[str, Any]:
        matches = self.tier_manager.tier2.search(query, user_id=self.user_id)

        if not matches:
            return {"success": False, "message": "No matching chunks found", "matches": []}

        if confirm:
            return {
                "success": True,
                "confirm_required": True,
                "matches": [
                    {
                        "chunk_id": c.id,
                        "content": c.content[:100] + "..." if len(c.content) > 100 else c.content,
                        "timestamp": c.timestamp.isoformat(),
                    }
                    for c in matches
                ],
                "message": f"Found {len(matches)} chunks. Call forget_confirmed() to delete.",
            }

        for chunk in matches:
            trash_item = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "task_type": chunk.task_type,
                "timestamp": chunk.timestamp.isoformat(),
                "deleted_at": datetime.now().isoformat(),
            }
            self.trash.append(trash_item)
            self.tier_manager.tier2.store.store_trash_item(self.user_id, trash_item)
            self.tier_manager.remove_chunk(chunk.id)
            self.user.forget_chunk(chunk.id)

        self.tier_manager.tier2.store.store_user(self.user)
        self._log_action("forget", {"query": query, "num_deleted": len(matches)})

        n = len(matches)
        return {
            "success": True, "num_deleted": n,
            "message": f"Forgot {n} chunks (moved to trash)",
        }

    def forget_confirmed(self, chunk_ids: List[str]) -> Dict[str, Any]:
        for chunk_id in chunk_ids:
            self.tier_manager.remove_chunk(chunk_id)
            self.user.forget_chunk(chunk_id)

        self.tier_manager.tier2.store.store_user(self.user)
        self._log_action(
            "forget_confirmed",
            {"chunk_ids": chunk_ids, "num_deleted": len(chunk_ids)},
        )

        n = len(chunk_ids)
        return {
            "success": True, "num_deleted": n,
            "message": f"Forgot {n} chunks",
        }

    def temporary(self, content: str) -> Dict[str, Any]:
        session = self._get_or_create_session()

        chunk = self._create_chunk(content)
        chunk.is_temporary = True

        self.tier_manager.add_chunk(chunk, user_id=self.user_id, session_id=session.id)
        session.add_chunk(chunk)

        self._log_action("temporary", {"chunk_id": chunk.id, "content": content})

        return {
            "success": True, "chunk_id": chunk.id,
            "message": "Added to session memory (temporary)",
        }

    def show_memory(self, category: str = "all") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat(),
        }

        if category in ("all", "pinned"):
            pinned = self.tier_manager.tier2.get_pinned(self.user_id)
            result["pinned"] = [
                {
                    "chunk_id": c.id,
                    "content": c.content,
                    "timestamp": c.timestamp.isoformat(),
                    "note": c.metadata.get("user_note"),
                }
                for c in pinned
            ]

        if category in ("all", "session"):
            if self.current_session:
                session_chunks = self.current_session.get_recent_chunks(10)
                result["session"] = [
                    {
                        "chunk_id": c.id,
                        "content": c.content[:100] + "..." if len(c.content) > 100 else c.content,
                        "timestamp": c.timestamp.isoformat(),
                    }
                    for c in session_chunks
                ]

        if category in ("all", "ai_managed"):
            ai_chunks = self.tier_manager.tier1.get_all()
            result["ai_managed"] = [
                {
                    "chunk_id": c.id,
                    "importance": c.importance_score,
                    "task_type": c.task_type,
                    "summary": c.summary,
                }
                for c in ai_chunks[:10]
            ]

        return result

    def get_stats(self) -> Dict[str, Any]:
        tier_stats = self.tier_manager.get_all_stats()
        user_stats = self.tier_manager.tier2.get_stats(self.user_id)

        return {
            "user_id": self.user_id,
            "control_mode": self.control_mode,
            "tiers": tier_stats,
            "user": user_stats,
            "current_session_active": (
                self.current_session is not None
                and self.current_session.is_active
            ),
        }

    def export_data(self, format: str = "json") -> str:
        data = {
            "user_id": self.user_id,
            "export_timestamp": datetime.now().isoformat(),
            "pinned_chunks": [],
            "sessions": [],
            "audit_log": self.audit_log,
        }

        pinned = self.tier_manager.tier2.get_pinned(self.user_id)
        data["pinned_chunks"] = [c.to_dict() for c in pinned]

        sessions = self.tier_manager.tier2.store.get_user_sessions(self.user_id)
        data["sessions"] = [s.to_dict() for s in sessions]

        if format == "json":
            return json.dumps(data, indent=2)

        lines = [
            f"MemCtrl Data Export - User: {self.user_id}",
            f"Exported: {data['export_timestamp']}",
            "",
            f"Pinned Chunks: {len(data['pinned_chunks'])}",
            f"Sessions: {len(data['sessions'])}",
            f"Audit Log Entries: {len(data['audit_log'])}",
        ]
        return "\n".join(lines)

    def get_dashboard(self) -> Dict[str, Any]:
        """Full memory state for the UI dashboard."""
        tier0_chunks = self.tier_manager.tier0.get_all()
        tier1_chunks = self.tier_manager.tier1.get_all()
        tier0_usage = self.tier_manager.tier0.get_usage()
        tier1_usage = self.tier_manager.tier1.get_usage()

        def _chunk_info(c: Chunk, tier: str) -> Dict[str, Any]:
            info: Dict[str, Any] = {
                "chunk_id": c.id,
                "content": c.content[:150] + "..." if len(c.content) > 150 else c.content,
                "task_type": c.task_type or "unclassified",
                "tier": tier,
                "is_pinned": c.is_pinned,
                "timestamp": c.timestamp.isoformat(),
                "access_count": c.access_count,
                "tokens": c.tokens,
                "priority": round(compute_task_aware_priority(c), 1),
            }
            if c.summary:
                info["summary"] = c.summary[:100] + "..." if len(c.summary) > 100 else c.summary
            return info

        pinned = [_chunk_info(c, "active") for c in tier0_chunks if c.is_pinned]
        active = [_chunk_info(c, "active") for c in tier0_chunks if not c.is_pinned]
        compressed = [_chunk_info(c, "compressed") for c in tier1_chunks]

        return {
            "pinned": pinned,
            "active": active,
            "compressed": compressed,
            "trash": self.trash[-20:],
            "memory_pressure": {
                "tier0_pct": round(tier0_usage["utilization"] * 100, 1),
                "tier0_tokens": tier0_usage["current_tokens"],
                "tier0_max": tier0_usage["max_tokens"],
                "tier1_pct": round(tier1_usage["utilization"] * 100, 1),
                "tier1_tokens": tier1_usage["current_tokens"],
                "tier1_max": tier1_usage["max_tokens"],
            },
        }

    def restore_from_trash(self, chunk_id: str) -> Dict[str, Any]:
        """Restore a chunk from trash back into memory."""
        for i, item in enumerate(self.trash):
            if item["chunk_id"] == chunk_id:
                session = self._get_or_create_session()
                chunk = self._create_chunk(item["content"])
                chunk.id = item["chunk_id"]
                chunk.task_type = item.get("task_type")
                self.tier_manager.add_chunk(
                    chunk, user_id=self.user_id, session_id=session.id,
                )
                self.trash.pop(i)
                self.tier_manager.tier2.store.delete_trash_item(chunk_id)
                self._log_action("restore", {"chunk_id": chunk_id})
                return {"success": True, "message": f"Restored chunk {chunk_id[:8]}..."}
        return {"success": False, "message": "Chunk not found in trash"}

    def unpin(self, chunk_id: str) -> Dict[str, Any]:
        """Unpin a pinned chunk."""
        chunk = self.tier_manager.tier0.get(chunk_id)
        if chunk and chunk.is_pinned:
            chunk.is_pinned = False
            chunk.priority = ChunkPriority.NORMAL
            self.user.forget_chunk(chunk_id)
            # Update the chunk in Tier2 so get_pinned() reflects the change
            self.tier_manager.tier2.store.store_chunk(chunk)
            self.tier_manager.tier2.store.store_user(self.user)
            self._log_action("unpin", {"chunk_id": chunk_id})
            return {"success": True, "message": f"Unpinned {chunk_id[:8]}..."}
        return {"success": False, "message": "Chunk not found or not pinned"}

    # -- Smart Suggestions (suggest, never decide) --

    # Patterns that indicate critical information worth pinning
    _CRITICAL_PATTERNS = [
        # Credentials and connection strings
        (r'(?:password|passwd|pwd|secret|token|api.?key)\s*(?:is|=|:)\s*\S+', 'credential'),
        (r'(?:postgresql|mysql|mongodb|redis|sqlite)://\S+', 'connection_string'),
        (r'sk-[a-zA-Z0-9]{20,}', 'api_key'),
        # Medical
        (r'\b(?:allergic|allergy)\s+(?:to\s+)?\w+', 'allergy'),
        (r'\b\d+\s*(?:mg|mcg|ml|units?)(?:/(?:day|daily|kg|dose))?\b', 'dosage'),
        (r'\b(?:diagnosed|diagnosis)\s+(?:with\s+)?\w+', 'diagnosis'),
        (r'\bHbA1c\s*(?:is|was|=|:)\s*[\d.]+', 'lab_result'),
        (r'\b(?:BP|blood pressure)\s*(?:is|was|=|:)?\s*\d+/\d+', 'vital_sign'),
        # Technical specifics
        (r'(?:port|PORT)\s*(?:is|=|:)\s*\d{2,5}', 'port'),
        (r'https?://\S+(?:webhook|notify|callback)\S*', 'webhook_url'),
    ]

    def suggest_pins(self) -> List[Dict[str, Any]]:
        """Detect critical information in recent messages and suggest pinning.

        Returns a list of suggestions. The user decides whether to accept each one.
        MemCtrl never auto-pins — it only suggests.
        """
        suggestions = []
        seen_contents = set()

        # Check recent chunks in Tier0 and Tier1
        all_chunks = self.tier_manager.tier0.get_all() + self.tier_manager.tier1.get_all()
        for chunk in all_chunks:
            if chunk.is_pinned:
                continue

            content = chunk.content
            for pattern, category in self._CRITICAL_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    if matched_text in seen_contents:
                        continue
                    seen_contents.add(matched_text)

                    suggestions.append({
                        "chunk_id": chunk.id,
                        "category": category,
                        "matched_text": matched_text,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content,
                        "reason": f"Detected {category.replace('_', ' ')} — this looks important. Pin it?",
                    })

        return suggestions

    def accept_pin_suggestion(self, chunk_id: str) -> Dict[str, Any]:
        """Accept a pin suggestion. Pins the chunk."""
        chunk = self.tier_manager.tier0.get(chunk_id) or self.tier_manager.tier1.get(chunk_id)
        if not chunk:
            return {"success": False, "message": "Chunk not found"}

        chunk.is_pinned = True
        chunk.priority = ChunkPriority.USER_PINNED
        self.user.pin_chunk(chunk.id)
        self.tier_manager.tier2.store.store_chunk(chunk)
        self.tier_manager.tier2.store.store_user(self.user)
        self._log_action("accept_pin_suggestion", {"chunk_id": chunk_id})
        return {"success": True, "message": f"Pinned {chunk_id[:8]}..."}

    def suggest_cleanup(self, stale_hours: float = 24.0) -> List[Dict[str, Any]]:
        """Find stale chunks and suggest deletion with token savings estimate.

        Returns suggestions for chunks not accessed since stale_hours ago.
        The user decides whether to keep or delete each one.
        """
        cutoff = datetime.now() - timedelta(hours=stale_hours)
        suggestions = []

        all_chunks = self.tier_manager.tier0.get_all() + self.tier_manager.tier1.get_all()
        for chunk in all_chunks:
            if chunk.is_pinned:
                continue
            if chunk.last_accessed < cutoff:
                age_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
                suggestions.append({
                    "chunk_id": chunk.id,
                    "content_preview": chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content,
                    "task_type": chunk.task_type or "unclassified",
                    "tokens": chunk.tokens,
                    "last_accessed": chunk.last_accessed.isoformat(),
                    "hours_stale": round(age_hours, 1),
                    "reason": (
                        f"Not accessed in {round(age_hours, 1)} hours. "
                        f"Deleting saves {chunk.tokens} tokens."
                    ),
                })

        # Sort by staleness (most stale first)
        suggestions.sort(key=lambda s: s["hours_stale"], reverse=True)

        if suggestions:
            total_tokens = sum(s["tokens"] for s in suggestions)
            for s in suggestions:
                s["total_recoverable_tokens"] = total_tokens

        return suggestions

    def accept_cleanup(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Accept cleanup suggestions. Moves chunks to trash."""
        deleted = 0
        tokens_saved = 0
        for chunk_id in chunk_ids:
            chunk = self.tier_manager.tier0.get(chunk_id) or self.tier_manager.tier1.get(chunk_id)
            if chunk and not chunk.is_pinned:
                trash_item = {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "task_type": chunk.task_type,
                    "timestamp": chunk.timestamp.isoformat(),
                    "deleted_at": datetime.now().isoformat(),
                }
                self.trash.append(trash_item)
                self.tier_manager.tier2.store.store_trash_item(self.user_id, trash_item)
                tokens_saved += chunk.tokens
                self.tier_manager.remove_chunk(chunk.id)
                deleted += 1

        self._log_action("accept_cleanup", {"num_deleted": deleted, "tokens_saved": tokens_saved})
        return {
            "success": True,
            "num_deleted": deleted,
            "tokens_saved": tokens_saved,
            "message": f"Cleaned up {deleted} stale chunks, saved {tokens_saved} tokens",
        }

    # -- Context Budget Debugger --

    def budget_report(self, max_tokens: int = 4096) -> Dict[str, Any]:
        """Show exactly where tokens are being spent in the context window.

        Returns a breakdown of token usage by category so users can see
        what's consuming their budget and make informed decisions.
        """
        # System prompt
        pinned = self.tier_manager.tier2.get_pinned(self.user_id)
        system_parts = ["You are a helpful assistant."]
        if pinned:
            pinned_text = "\n".join(f"- {c.content}" for c in pinned)
            system_parts.append(f"The user has pinned the following information:\n{pinned_text}")
        system_content = " ".join(system_parts)
        system_tokens = count_tokens(system_content, self.config.tokenizer_model)

        # Pinned tokens (subset of system)
        pinned_tokens = sum(
            count_tokens(c.content, self.config.tokenizer_model) for c in pinned
        )

        # Active (Tier0) tokens
        tier0_chunks = self.tier_manager.tier0.get_all()
        active_tokens = sum(c.tokens for c in tier0_chunks if not c.is_pinned)
        active_count = sum(1 for c in tier0_chunks if not c.is_pinned)

        # Compressed (Tier1) tokens
        tier1_chunks = self.tier_manager.tier1.get_all()
        compressed_tokens = sum(
            count_tokens(c.summary or c.content, self.config.tokenizer_model)
            for c in tier1_chunks
        )
        compressed_count = len(tier1_chunks)

        total_used = system_tokens + active_tokens + compressed_tokens
        remaining = max(0, max_tokens - total_used)
        usage_pct = round(total_used / max_tokens * 100, 1) if max_tokens > 0 else 0

        return {
            "max_tokens": max_tokens,
            "total_used": total_used,
            "remaining": remaining,
            "usage_pct": usage_pct,
            "breakdown": {
                "system_prompt": {"tokens": system_tokens - pinned_tokens, "label": "System prompt"},
                "pinned": {"tokens": pinned_tokens, "count": len(pinned), "label": "Pinned memories"},
                "active": {"tokens": active_tokens, "count": active_count, "label": "Active messages"},
                "compressed": {"tokens": compressed_tokens, "count": compressed_count, "label": "Compressed messages"},
            },
            "recommendations": self._budget_recommendations(
                usage_pct, pinned_tokens, active_tokens, compressed_tokens, remaining,
            ),
        }

    def _budget_recommendations(
        self, usage_pct, pinned_tokens, active_tokens, compressed_tokens, remaining,
    ) -> List[str]:
        recs = []
        if usage_pct > 90:
            recs.append("Context is over 90% full. Consider forgetting old messages or running suggest_cleanup().")
        if pinned_tokens > active_tokens and pinned_tokens > 500:
            recs.append(f"Pinned memories use {pinned_tokens} tokens. Review pins — unpin anything no longer needed.")
        if remaining < 200:
            recs.append(f"Only {remaining} tokens remaining. New messages may trigger auto-eviction.")
        if usage_pct < 30:
            recs.append("Plenty of budget remaining. No action needed.")
        return recs

    def close_session(self):
        if self.current_session:
            self.current_session.close()
            self.tier_manager.tier2.store.store_session(self.current_session)
            self.current_session = None

    # -- Internal --

    def _handle_memory_pressure(self):
        if self.control_mode in ("automatic", "hybrid"):
            self._auto_evict()

    def _auto_evict(self):
        evicted = self.tier_manager.task_aware_evict()
        self._log_action("auto_evict", {"num_evicted": evicted, "reason": "memory_pressure"})


def wrap(client, max_tokens: int = 4096, user_id: Optional[str] = None, **kwargs):
    """Wrap an OpenAI or Anthropic client with automatic memory management.

    Usage:
        import openai
        client = openai.OpenAI(api_key="sk-...")
        wrapped = memctrl.wrap(client, max_tokens=4096)

        # Use normally — memctrl optimizes context automatically
        response = wrapped.chat("Hello, help me debug my Flask app")
        response = wrapped.chat("The error is on line 42")

    The wrapper intercepts messages, manages memory tiers, and
    ensures the conversation fits within the token budget.
    """
    return WrappedClient(client, max_tokens=max_tokens, user_id=user_id, **kwargs)


class WrappedClient:
    """Transparent wrapper around an LLM client with automatic memory management."""

    def __init__(self, client, max_tokens: int = 4096, user_id: Optional[str] = None, **kwargs):
        self._client = client
        self._max_tokens = max_tokens
        self._provider = self._detect_provider(client)

        # Create a MemoryController with echo LLM (we use the user's client for generation)
        self.mc = MemoryController(
            user_id=user_id,
            provider="echo",
            **kwargs,
        )

    @staticmethod
    def _detect_provider(client) -> str:
        module = type(client).__module__ or ""
        if "anthropic" in module:
            return "anthropic"
        if "openai" in module:
            return "openai"
        return "openai"  # default to OpenAI-compatible

    def chat(self, message: str, **kwargs) -> str:
        """Send a message and get a response with automatic memory management.

        Args:
            message: The user's message
            **kwargs: Extra args passed to the underlying API call
                      (model, temperature, etc.)

        Returns:
            The assistant's response text.
        """
        # Record the user message in memory
        self.mc.add_message("user", message)

        # Build optimized context
        messages = self.mc.optimize(max_tokens=self._max_tokens)

        # Call the underlying client
        response_text = self._call_client(messages, **kwargs)

        # Record the assistant response in memory
        self.mc.add_message("assistant", response_text)

        return response_text

    def _call_client(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if self._provider == "anthropic":
            return self._call_anthropic(messages, **kwargs)
        return self._call_openai(messages, **kwargs)

    def _call_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        call_kwargs = {"messages": chat_messages, "max_tokens": kwargs.pop("max_tokens", 1024)}
        if system_msg:
            call_kwargs["system"] = system_msg
        if "model" not in kwargs:
            kwargs["model"] = "claude-sonnet-4-20250514"
        call_kwargs.update(kwargs)

        response = self._client.messages.create(**call_kwargs)
        return response.content[0].text

    def _call_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if "model" not in kwargs:
            kwargs["model"] = "gpt-4o-mini"
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        response = self._client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    # Expose MemoryController methods for advanced usage
    def pin(self, content: str, **kwargs):
        return self.mc.pin(content, **kwargs)

    def forget(self, query: str, **kwargs):
        return self.mc.forget(query, **kwargs)

    def suggest_pins(self):
        return self.mc.suggest_pins()

    def suggest_cleanup(self, **kwargs):
        return self.mc.suggest_cleanup(**kwargs)

    def budget_report(self):
        return self.mc.budget_report(max_tokens=self._max_tokens)

    def show_memory(self, **kwargs):
        return self.mc.show_memory(**kwargs)

    def get_stats(self):
        return self.mc.get_stats()
