import json
import logging
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime

from .models import Chunk, Session, User, ChunkType, ChunkPriority
from .core.tiers import TierManager
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
        self.audit_log: List[Dict[str, Any]] = []

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
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": self.user_id,
            "details": details,
        })

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

    # -- Public API --

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
            self.tier_manager.remove_chunk(chunk.id)
            self.user.forget_chunk(chunk.id)

        self.tier_manager.tier2.store.store_user(self.user)
        self._log_action("forget", {"query": query, "num_deleted": len(matches)})

        n = len(matches)
        return {
            "success": True, "num_deleted": n,
            "message": f"Forgot {n} chunks",
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
        tier0_chunks = self.tier_manager.tier0.get_all()
        sorted_chunks = sorted(tier0_chunks, key=lambda c: c.get_priority_value())
        num_to_evict = max(1, len(sorted_chunks) // 5)

        evicted = 0
        for chunk in sorted_chunks:
            if chunk.is_pinned:
                continue
            if self.tier_manager.demote_to_tier1(chunk.id):
                evicted += 1
            if evicted >= num_to_evict:
                break

        self._log_action("auto_evict", {"num_evicted": evicted, "reason": "memory_pressure"})
