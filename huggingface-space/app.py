"""MemCtrl — Interactive Demo on HuggingFace Spaces.

Watch your tokens being managed in real time. See exactly where every token goes,
what gets compressed, what gets pinned, and how much you save.
"""

import json
import re
from datetime import datetime
from uuid import uuid4

import gradio as gr

# ---------------------------------------------------------------------------
# Custom SVG Icon System — rounded, soft, unique to MemCtrl
# ---------------------------------------------------------------------------

def _svg(paths, color="#0284c7", size=20, viewbox="0 0 24 24", stroke_width=2):
    """Build an inline SVG icon."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{viewbox}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;margin-right:6px;">'
        f'{paths}</svg>'
    )

ICONS = {
    # Task types
    "medical": _svg(
        '<path d="M12 3v6m0 0v6m0-6h6m-6 0H6"/>'  # plus/cross
        '<circle cx="12" cy="12" r="10"/>',
        color="#ef4444"
    ),
    "code": _svg(
        '<polyline points="16 18 22 12 16 6"/>'
        '<polyline points="8 6 2 12 8 18"/>',
        color="#3b82f6"
    ),
    "tutoring": _svg(
        '<circle cx="12" cy="8" r="5"/>'
        '<path d="M12 13v4"/>'
        '<path d="M9 21h6"/>'
        '<path d="M10 5.5c0-1 .5-2.5 2-2.5s2 1.5 2 2.5"/>',
        color="#8b5cf6"
    ),
    "writing": _svg(
        '<path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>',
        color="#f59e0b"
    ),
    "general": _svg(
        '<path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 '
        '8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 '
        '8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>',
        color="#6b7280"
    ),

    # Actions
    "pin": _svg(
        '<path d="M12 2v8"/>'
        '<circle cx="12" cy="14" r="4"/>'
        '<path d="M12 18v4"/>',
        color="#b45309"
    ),
    "compress": _svg(
        '<polyline points="4 14 10 14 10 20"/>'
        '<polyline points="20 10 14 10 14 4"/>'
        '<line x1="14" y1="10" x2="21" y2="3"/>'
        '<line x1="3" y1="21" x2="10" y2="14"/>',
        color="#d97706"
    ),
    "evict": _svg(
        '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>',
        color="#dc2626"
    ),
    "add": _svg(
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="16"/>'
        '<line x1="8" y1="12" x2="16" y2="12"/>',
        color="#0284c7"
    ),
    "save": _svg(
        '<path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/>'
        '<polyline points="17 21 17 13 7 13 7 21"/>'
        '<polyline points="7 3 7 8 15 8"/>',
        color="#64748b"
    ),

    # UI elements
    "tokens": _svg(
        '<circle cx="12" cy="12" r="8"/>'
        '<path d="M12 8v8"/>'
        '<path d="M9 11h6"/>'
        '<path d="M9 13h6"/>',
        color="#0284c7"
    ),
    "savings": _svg(
        '<path d="M12 2L2 7l10 5 10-5-10-5z"/>'
        '<path d="M2 17l10 5 10-5"/>'
        '<path d="M2 12l10 5 10-5"/>',
        color="#16a34a"
    ),
    "money": _svg(
        '<line x1="12" y1="1" x2="12" y2="23"/>'
        '<path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/>',
        color="#16a34a"
    ),
    "xray": _svg(
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '<line x1="8" y1="11" x2="14" y2="11"/>'
        '<line x1="11" y1="8" x2="11" y2="14"/>',
        color="#7c3aed"
    ),
    "tiers": _svg(
        '<rect x="3" y="3" width="18" height="18" rx="6" stroke-dasharray="4 3" opacity="0.4"/>'
        '<rect x="6.5" y="6.5" width="11" height="11" rx="4" opacity="0.7"/>'
        '<rect x="9.5" y="9.5" width="5" height="5" rx="2" fill="#0284c7" stroke="none"/>',
        color="#0284c7"
    ),
    "reset": _svg(
        '<polyline points="1 4 1 10 7 10"/>'
        '<path d="M3.51 15a9 9 0 102.13-9.36L1 10"/>',
        color="#64748b"
    ),
    "send": _svg(
        '<line x1="22" y1="2" x2="11" y2="13"/>'
        '<polygon points="22 2 15 22 11 13 2 9 22 2"/>',
        color="#0284c7"
    ),
    "chart": _svg(
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>',
        color="#0284c7"
    ),
    "shield": _svg(
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        color="#16a34a"
    ),
    "entity": _svg(
        '<path d="M4 7V4h16v3"/>'
        '<path d="M9 20h6"/>'
        '<path d="M12 4v16"/>',
        color="#6d28d9"
    ),
}

def icon(name, size=18):
    """Get an icon by name, optionally at a different size."""
    base = ICONS.get(name, ICONS["general"])
    if size != 20:
        return base.replace('width="20"', f'width="{size}"').replace('height="20"', f'height="{size}"')
    return base


# ---------------------------------------------------------------------------
# Lightweight simulation of MemCtrl tiers
# ---------------------------------------------------------------------------

TASK_RETENTION = {
    "medical": {"weight": 1.5, "decay_hours": 168, "color": "#ef4444", "label": "Medical"},
    "code": {"weight": 1.3, "decay_hours": 72, "color": "#3b82f6", "label": "Code"},
    "tutoring": {"weight": 1.2, "decay_hours": 48, "color": "#8b5cf6", "label": "Tutoring"},
    "writing": {"weight": 1.0, "decay_hours": 24, "color": "#f59e0b", "label": "Writing"},
    "general": {"weight": 0.8, "decay_hours": 12, "color": "#6b7280", "label": "General"},
}

ENTITY_PATTERNS = [
    (r'\b\d+\.\d+\.\d+(?:\.\d+)?\b', 'version'),
    (r'https?://\S+', 'url'),
    (r'(?:postgresql|mysql|mongodb|redis|sqlite)://\S+', 'connection_string'),
    (r'sk-[a-zA-Z0-9]{20,}', 'api_key'),
    (r'\b\d+\s*(?:mg|mcg|ml|units?|kg|lbs?|[°][CF])\b', 'measurement'),
    (r'\b\d+/\d+\b', 'ratio'),
    (r'(?:port|PORT)\s*(?:is|=|:)\s*\d{2,5}', 'port'),
    (r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', 'identifier'),
    (r'\b[a-z_]+\.[a-z_]+\(\)', 'function_call'),
    (r'\bHbA1c\s*(?:is|was|=|:)?\s*[\d.]+', 'lab_result'),
    (r'\b\d+(?:\.\d+)?%\b', 'percentage'),
]

AUTO_PIN_PATTERNS = [
    (r'(?:password|passwd|pwd|secret|token|api.?key)\s*(?:is|=|:)\s*\S+', 'credential'),
    (r'(?:postgresql|mysql|mongodb|redis|sqlite)://\S+', 'connection_string'),
    (r'\b(?:allergic|allergy)\s+(?:to\s+)?\w+', 'allergy'),
    (r'\b\d+\s*(?:mg|mcg|ml|units?)(?:/(?:day|daily|kg|dose))?\b', 'dosage'),
    (r'\b(?:diagnosed|diagnosis)\s+(?:with\s+)?\w+', 'diagnosis'),
]

KEYWORD_TASK_MAP = {
    "medical": {"patient", "diagnosis", "symptoms", "medication", "dosage", "allergy", "blood", "pressure", "mg", "prescription", "vitals", "allergic"},
    "code": {"function", "class", "import", "def", "return", "error", "debug", "api", "database", "sql", "flask", "server", "endpoint", "bug", "exception", "port"},
    "tutoring": {"explain", "learn", "understand", "example", "concept", "homework", "study", "tutorial", "lesson", "practice"},
    "writing": {"essay", "paragraph", "draft", "rewrite", "tone", "proofread", "grammar", "outline", "article"},
}

PRICE_PER_1K_INPUT = 0.0025


def classify_task(text):
    words = set(text.lower().split())
    best, best_score = "general", 0
    for task, keywords in KEYWORD_TASK_MAP.items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best = task
    return best if best_score >= 2 else "general"


def count_tokens(text):
    return int(len(text.split()) * 1.3)


def extract_entities(text):
    entities = []
    seen = set()
    for pattern, label in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text):
            value = match.group(0).strip()
            if value not in seen and len(value) > 2:
                seen.add(value)
                entities.append(f"{label}: {value}")
    return entities


def extractive_summarize(text, max_words=30):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1 or len(text.split()) <= max_words:
        return text
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "it", "i", "you", "he",
            "she", "we", "they", "this", "that", "with", "from", "by", "as"}
    word_freq = {}
    for word in text.lower().split():
        w = re.sub(r'[^a-z0-9]', '', word)
        if w and w not in stop:
            word_freq[w] = word_freq.get(w, 0) + 1
    scored = []
    for i, sent in enumerate(sentences):
        words = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in sent.split()]
        score = sum(word_freq.get(w, 0) for w in words if w)
        scored.append((score, i, sent))
    scored.sort(reverse=True)
    selected = []
    total = 0
    for _, idx, sent in scored:
        sw = len(sent.split())
        if total + sw > max_words:
            if not selected:
                selected.append(idx)
            break
        selected.append(idx)
        total += sw
    selected.sort()
    return " ".join(sentences[i] for i in selected)


class MemCtrlSimulator:
    def __init__(self, max_budget=4096):
        self.max_budget = max_budget
        self.messages = []
        self.tier0 = []
        self.tier1 = []
        self.pinned = []
        self.event_log = []
        self.total_tokens_raw = 0
        self.tier0_max = int(max_budget * 0.6)
        self.tier1_max = int(max_budget * 0.3)

    def add_message(self, role, content):
        tokens = count_tokens(content)
        task = classify_task(content)
        entities = extract_entities(content)
        msg = {
            "id": str(uuid4())[:8],
            "role": role,
            "content": content,
            "tokens": tokens,
            "task": task,
            "entities": entities,
            "timestamp": datetime.now(),
            "tier": "active",
        }
        self.messages.append(msg)
        self.total_tokens_raw += tokens

        # Check for auto-pin
        for pattern, category in AUTO_PIN_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                pin_text = match.group(0)
                if not any(pin_text in p["content"] for p in self.pinned):
                    pin = {
                        "id": str(uuid4())[:8],
                        "content": pin_text,
                        "category": category,
                        "tokens": count_tokens(pin_text),
                        "source_msg": msg["id"],
                    }
                    self.pinned.append(pin)
                    self.event_log.append({
                        "action": "auto_pin",
                        "detail": f"Auto-pinned {category}: '{pin_text[:50]}'",
                        "tokens_protected": pin["tokens"],
                        "icon": "pin",
                    })

        self.tier0.append(msg)
        self.event_log.append({
            "action": "add",
            "detail": f"Added to Active tier ({tokens} tokens, {task})",
            "tokens_used": tokens,
            "icon": "add",
        })

        # Compress if tier0 is over budget
        tier0_tokens = sum(m["tokens"] for m in self.tier0)
        while tier0_tokens > self.tier0_max and len(self.tier0) > 2:
            oldest = self.tier0.pop(0)
            if any(p["source_msg"] == oldest["id"] for p in self.pinned):
                self.tier0.insert(0, oldest)
                if len(self.tier0) > 2:
                    oldest = self.tier0.pop(1)
                else:
                    break

            summary = extractive_summarize(oldest["content"])
            entities = extract_entities(oldest["content"])
            if entities:
                summary += " [" + "; ".join(entities[:3]) + "]"
            original_tokens = oldest["tokens"]
            compressed_tokens = count_tokens(summary)
            savings = original_tokens - compressed_tokens

            compressed = {
                **oldest,
                "original_content": oldest["content"],
                "content": summary,
                "original_tokens": original_tokens,
                "tokens": compressed_tokens,
                "tier": "compressed",
            }
            self.tier1.append(compressed)
            tier0_tokens -= original_tokens

            self.event_log.append({
                "action": "compress",
                "detail": (
                    f"Compressed: {original_tokens} > {compressed_tokens} tokens "
                    f"(saved {savings}). Entities preserved: {len(entities)}"
                ),
                "tokens_saved": savings,
                "icon": "compress",
            })

        # Evict from tier1 if over budget
        tier1_tokens = sum(m["tokens"] for m in self.tier1)
        while tier1_tokens > self.tier1_max and self.tier1:
            evicted = self.tier1.pop(0)
            tier1_tokens -= evicted["tokens"]
            self.event_log.append({
                "action": "evict",
                "detail": f"Evicted to disk: '{evicted['content'][:40]}...' ({evicted['tokens']} tokens freed)",
                "tokens_freed": evicted["tokens"],
                "icon": "evict",
            })

        return msg

    def get_stats(self):
        tier0_tokens = sum(m["tokens"] for m in self.tier0)
        tier1_tokens = sum(m["tokens"] for m in self.tier1)
        pinned_tokens = sum(p["tokens"] for p in self.pinned)
        total_managed = tier0_tokens + tier1_tokens + pinned_tokens
        savings_tokens = self.total_tokens_raw - total_managed
        savings_pct = (savings_tokens / self.total_tokens_raw * 100) if self.total_tokens_raw > 0 else 0
        money_saved = savings_tokens / 1000 * PRICE_PER_1K_INPUT

        return {
            "tier0_tokens": tier0_tokens,
            "tier0_count": len(self.tier0),
            "tier0_max": self.tier0_max,
            "tier0_pct": tier0_tokens / self.tier0_max * 100 if self.tier0_max > 0 else 0,
            "tier1_tokens": tier1_tokens,
            "tier1_count": len(self.tier1),
            "tier1_max": self.tier1_max,
            "tier1_pct": tier1_tokens / self.tier1_max * 100 if self.tier1_max > 0 else 0,
            "pinned_tokens": pinned_tokens,
            "pinned_count": len(self.pinned),
            "total_raw": self.total_tokens_raw,
            "total_managed": total_managed,
            "savings_tokens": max(0, savings_tokens),
            "savings_pct": max(0, savings_pct),
            "money_saved": max(0, money_saved),
            "messages_total": len(self.messages),
            "budget_used_pct": total_managed / self.max_budget * 100 if self.max_budget > 0 else 0,
            "budget_remaining": max(0, self.max_budget - total_managed),
        }


# ---------------------------------------------------------------------------
# Pre-built scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Patient Consult": [
        ("user", "I'm Dr. Sarah. New patient: Maria, 45F, presenting with chest pain and shortness of breath."),
        ("assistant", "I'll help track Maria's case. What are her vitals and relevant history?"),
        ("user", "BP is 145/92, heart rate 88. She's allergic to penicillin. Currently on 50mg metoprolol daily."),
        ("assistant", "Noted: elevated BP 145/92, HR 88. Penicillin allergy flagged. Metoprolol 50mg daily. Any lab results?"),
        ("user", "HbA1c is 7.2, indicating pre-diabetic range. Cholesterol is 240 mg/dL total."),
        ("assistant", "HbA1c 7.2 (pre-diabetic) and elevated cholesterol 240 mg/dL noted. I'd recommend considering statin therapy given the cardiovascular risk profile."),
        ("user", "Good point. Let's also order a stress test and echocardiogram. Her family history includes MI in father at age 52."),
        ("assistant", "Ordering stress test and echo. Family history of early MI is significant. Combined with her current profile she's moderate-to-high cardiovascular risk."),
        ("user", "What was her blood pressure again? And the allergy?"),
        ("assistant", "BP was 145/92 and she's allergic to penicillin. Both were captured at intake."),
    ],
    "Debug Session": [
        ("user", "My Flask app is crashing. The DB connection is postgresql://admin:secret@db.prod.internal:5432/myapp"),
        ("assistant", "I'll help debug. What error are you seeing? I've noted your connection string."),
        ("user", "Getting ConnectionRefusedError on line 42 of app.py when calling db.session.query()"),
        ("assistant", "ConnectionRefusedError usually means PostgreSQL isn't accepting connections. Let's check: Is the DB server running? Is port 5432 open?"),
        ("user", "The server is running. I checked with pg_isready and it says accepting connections on port 5432."),
        ("assistant", "If pg_isready passes, the issue might be network or firewall between your app and db.prod.internal. Try: telnet db.prod.internal 5432"),
        ("user", "telnet works fine. I think the issue is SQLAlchemy pool exhaustion. My pool_size is 5 but I have 20 workers."),
        ("assistant", "That's your bottleneck. With 20 workers and pool_size=5, connections get exhausted. Fix: set pool_size=20 and max_overflow=10 in create_engine()."),
        ("user", "That fixed it! Now I'm getting a different error: ImportError for flask_cors on the deploy server."),
        ("assistant", "flask_cors isn't in your requirements.txt. Add it: pip install flask-cors, then add flask-cors to requirements.txt and redeploy."),
        ("user", "What was my database connection string? I need it for the deploy config."),
        ("assistant", "Your connection string is postgresql://admin:secret@db.prod.internal:5432/myapp"),
    ],
    "ML Tutoring": [
        ("user", "I'm learning machine learning. Can you explain what gradient descent is?"),
        ("assistant", "Gradient descent is an optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent. Think of it like a ball rolling downhill."),
        ("user", "What's the difference between batch, mini-batch, and stochastic gradient descent?"),
        ("assistant", "Batch GD uses ALL training data per update. Stochastic GD uses ONE sample. Mini-batch is the sweet spot using a small batch of 32-256 samples for each update. Most deep learning uses mini-batch."),
        ("user", "What learning rate should I use? My model loss is oscillating wildly."),
        ("assistant", "Oscillating loss means your learning rate is too high. Start with 0.001 for Adam optimizer. If still oscillating, try 0.0001. You can also use learning rate schedulers like ReduceLROnPlateau."),
        ("user", "Thanks! Now explain backpropagation in simple terms."),
        ("assistant", "Backpropagation is how neural networks learn from mistakes. It works backward: make a prediction, calculate error, send error signal backward through each layer, each layer adjusts its weights to reduce the error."),
        ("user", "What was the recommended learning rate you mentioned?"),
        ("assistant", "I recommended starting with 0.001 for Adam optimizer, and reducing to 0.0001 if the loss still oscillates."),
    ],
}


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def build_stats_html(stats):
    tier0_color = "#16a34a" if stats["tier0_pct"] < 60 else "#f59e0b" if stats["tier0_pct"] < 85 else "#ef4444"
    tier1_color = "#16a34a" if stats["tier1_pct"] < 60 else "#f59e0b" if stats["tier1_pct"] < 85 else "#ef4444"
    budget_color = "#16a34a" if stats["budget_used_pct"] < 60 else "#f59e0b" if stats["budget_used_pct"] < 85 else "#ef4444"

    def progress_bar(pct, color, icon_name, label, detail):
        pct = min(pct, 100)
        return f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 600; color: var(--mc-text);">{icon(icon_name, 16)} {label}</span>
                <span style="color: var(--mc-muted); font-size: 0.9em;">{detail}</span>
            </div>
            <div style="background: var(--mc-border); border-radius: 8px; height: 10px; overflow: hidden;">
                <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 8px;
                            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        """

    # Compute per-message average
    avg_tokens = round(stats['total_raw'] / max(stats['messages_total'], 1), 1)
    budget_total = stats['total_managed'] + stats['budget_remaining']
    budget_remaining = stats['budget_remaining']

    # Soft pink blob decorator
    def pink_blob(top, left, size, opacity="0.12"):
        return (f'<div style="position:absolute; top:{top}; left:{left}; '
                f'width:{size}px; height:{size}px; border-radius:50%; '
                f'background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%); '
                f'opacity:{opacity}; pointer-events:none;"></div>')

    blobs = "".join([
        pink_blob("5%", "80%", 120, "0.15"),
        pink_blob("60%", "5%", 90, "0.1"),
        pink_blob("85%", "70%", 100, "0.08"),
    ])

    # Minimal flat stat card
    def flat_card(value, label, detail_text):
        return f"""
        <div class="stat-card" onclick="this.classList.toggle('expanded')"
             style="background: var(--mc-card); border-radius: 16px; padding: 24px 18px 18px 18px;
                    text-align: center; border: 1px solid var(--mc-border);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                    position: relative; transition: all 0.2s ease;">
            <div style="font-size: 2.2em; font-weight: 800; color: var(--mc-text);
                        letter-spacing: -0.02em; line-height: 1;">{value}</div>
            <div style="font-size: 0.75em; color: var(--mc-muted); margin-top: 8px;
                        font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">{label}</div>
            <div class="stat-detail" style="color: var(--mc-muted);">{detail_text}</div>
        </div>
        """

    return f"""
    <div style="padding: 16px; position: relative;">
        {blobs}

        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;
                    margin-bottom: 16px;">
            {flat_card(stats['messages_total'], "Messages",
                       f"User: {stats.get('user_count', stats['messages_total']//2)} / "
                       f"Asst: {stats.get('assistant_count', stats['messages_total'] - stats['messages_total']//2)}"
                       f"<br>Avg {avg_tokens} tok/msg")}
            {flat_card(stats['total_raw'], "Raw Tokens",
                       f"Tier 0: {stats['tier0_tokens']} active<br>"
                       f"Tier 1: {stats['tier1_tokens']} compressed")}
            {flat_card(stats['savings_tokens'], "Saved",
                       f"{stats['savings_pct']:.1f}% of context freed up for new conversation.")}
            {flat_card(f"${stats['money_saved']:.4f}", "Money Saved",
                       f"At GPT-4 rate. 100 sessions = ~${stats['money_saved'] * 100:.2f} saved.")}
        </div>

        <!-- Token Budget card -->
        <div style="background: var(--mc-card); border-radius: 16px; padding: 22px;
                    border: 1px solid var(--mc-border); box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                    position: relative;">
            <div style="position:absolute; top:-10px; right:16px;">
                <span style="background: var(--mc-accent); color: white; padding: 4px 12px;
                             border-radius: 14px; font-size: 0.72em; font-weight: 700;
                             text-transform: uppercase; letter-spacing: 0.05em;">LIVE</span>
            </div>
            <h3 style="margin: 0 0 16px 0; color: var(--mc-text); font-size: 1em; font-weight: 800;
                       text-transform: uppercase; letter-spacing: 0.04em;">
                {icon("chart", 18)} Token Budget
            </h3>
            {progress_bar(stats['budget_used_pct'], budget_color, 'chart', 'Overall Budget',
                          f"{stats['total_managed']} / {budget_total} tokens")}
            {progress_bar(stats['tier0_pct'], tier0_color, 'tiers', 'Active (Tier 0)',
                          f"{stats['tier0_tokens']} tok, {stats['tier0_count']} msgs")}
            {progress_bar(stats['tier1_pct'], tier1_color, 'compress', 'Compressed (Tier 1)',
                          f"{stats['tier1_tokens']} tok, {stats['tier1_count']} msgs")}
            <div style="text-align: right; font-size: 0.78em; color: var(--mc-muted); margin-top: 6px;
                        font-weight: 600;">
                {budget_remaining} tokens remaining
            </div>
        </div>

        <!-- Pinned + Savings cards -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
            <div class="stat-card" onclick="this.classList.toggle('expanded')"
                 style="background: var(--mc-card); border-radius: 16px; padding: 18px 20px;
                        border: 1px solid var(--mc-border);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: var(--mc-accent); text-transform: uppercase;
                            font-size: 0.75em; letter-spacing: 0.06em;">{icon("pin", 16)} Pinned</div>
                <div style="font-size: 1.8em; font-weight: 800; color: var(--mc-text);
                            margin: 4px 0; line-height: 1;">{stats['pinned_count']}</div>
                <div style="color: var(--mc-muted); font-size: 0.78em; font-weight: 500;">
                    {stats['pinned_tokens']} tokens protected</div>
                <div class="stat-detail" style="color: var(--mc-muted);">
                    Critical values auto-detected. Never compressed or evicted.
                </div>
            </div>
            <div class="stat-card" onclick="this.classList.toggle('expanded')"
                 style="background: var(--mc-card); border-radius: 16px; padding: 18px 20px;
                        border: 1px solid var(--mc-border);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: var(--mc-accent); text-transform: uppercase;
                            font-size: 0.75em; letter-spacing: 0.06em;">{icon("savings", 16)} Savings</div>
                <div style="font-size: 1.8em; font-weight: 800; color: var(--mc-text);
                            margin: 4px 0; line-height: 1;">{stats['savings_pct']:.1f}%</div>
                <div style="color: var(--mc-muted); font-size: 0.78em; font-weight: 500;">
                    token reduction</div>
                <div class="stat-detail" style="color: var(--mc-muted);">
                    {stats['savings_pct']:.1f}% compressed with 88% recall. More % = more saved.
                </div>
            </div>
        </div>
    </div>
    """


def build_event_log_html(events):
    if not events:
        return (
            '<div style="color: #94a3b8; text-align: center; padding: 40px;">'
            'Send a message to see what MemCtrl does behind the scenes...</div>'
        )

    ACTION_STYLES = {
        "compress": {"bg": "#fffbeb", "border": "#fcd34d"},
        "auto_pin": {"bg": "#fdf2f8", "border": "#f9a8d4"},
        "evict": {"bg": "#fef2f2", "border": "#fca5a5"},
        "add": {"bg": "#f0f9ff", "border": "#bae6fd"},
    }

    rows = []
    for e in reversed(events[-20:]):
        icon_name = e.get("icon", "general")
        action = e["action"]
        detail = e["detail"]
        style = ACTION_STYLES.get(action, ACTION_STYLES["add"])

        rows.append(f"""
        <div style="background: {style['bg']}; border-left: 3px solid {style['border']};
                    padding: 10px 14px; margin-bottom: 6px; border-radius: 0 10px 10px 0;
                    font-size: 0.9em;">
            {icon(icon_name, 16)}
            <strong style="color: #334155;">{action.replace('_', ' ').upper()}</strong>
            <span style="color: #64748b; margin-left: 8px;">{detail}</span>
        </div>
        """)

    return f"""
    <div style="padding: 8px; max-height: 500px; overflow-y: auto;">
        {''.join(rows)}
    </div>
    """


def build_tier_view_html(sim):
    sections = []

    # Pinned
    if sim.pinned:
        pins = ""
        for p in sim.pinned:
            pins += f"""
            <div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px;
                        padding: 12px 16px; margin-bottom: 6px; display: flex; align-items: center; gap: 10px;">
                {icon("pin", 18)}
                <div>
                    <div style="font-weight: 600; color: #92400e; font-size: 0.82em;">
                        {p['category'].replace('_', ' ').upper()}
                    </div>
                    <div style="color: #78350f; font-size: 0.95em;">{p['content'][:80]}</div>
                    <div style="color: #a16207; font-size: 0.78em;">{p['tokens']} tokens &middot; always preserved</div>
                </div>
            </div>
            """
        sections.append(f"""
        <div style="margin-bottom: 18px;">
            <h4 style="color: #b45309; margin-bottom: 10px; font-size: 0.95em;">
                {icon("pin", 16)} Pinned ({len(sim.pinned)})
            </h4>
            {pins}
        </div>
        """)

    # Active (Tier 0)
    active = ""
    for m in sim.tier0:
        task_info = TASK_RETENTION.get(m["task"], TASK_RETENTION["general"])
        entities_str = ""
        if m["entities"]:
            ents = ", ".join(m["entities"][:3])
            entities_str = f'<div style="color: #6d28d9; font-size: 0.78em; margin-top: 4px;">{icon("entity", 14)} {ents}</div>'

        active += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                    padding: 12px 16px; margin-bottom: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background: {task_info['color']}18; color: {task_info['color']};
                             padding: 3px 10px; border-radius: 12px; font-size: 0.78em; font-weight: 600;">
                    {icon(m['task'], 14)} {task_info['label']}
                </span>
                <span style="color: #94a3b8; font-size: 0.78em;">{m['tokens']} tokens</span>
            </div>
            <div style="color: #334155; margin-top: 8px; font-size: 0.9em; line-height: 1.4;">
                <strong style="color: #64748b;">{m['role'].title()}:</strong>
                {m['content'][:120]}{'...' if len(m['content']) > 120 else ''}
            </div>
            {entities_str}
        </div>
        """

    sections.append(f"""
    <div style="margin-bottom: 18px;">
        <h4 style="color: #0284c7; margin-bottom: 10px; font-size: 0.95em;">
            {icon("tiers", 16)} Active — Tier 0 ({len(sim.tier0)})
        </h4>
        {active if active else '<div style="color: #94a3b8; padding: 12px;">Empty</div>'}
    </div>
    """)

    # Compressed (Tier 1)
    compressed = ""
    for m in sim.tier1:
        task_info = TASK_RETENTION.get(m["task"], TASK_RETENTION["general"])
        original = m.get("original_tokens", m["tokens"])
        compressed += f"""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px;
                    padding: 12px 16px; margin-bottom: 6px; opacity: 0.85;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background: {task_info['color']}18; color: {task_info['color']};
                             padding: 3px 10px; border-radius: 12px; font-size: 0.78em; font-weight: 600;">
                    {icon("compress", 14)} {task_info['label']}
                </span>
                <span style="color: #94a3b8; font-size: 0.78em;">
                    {original} &rarr; {m['tokens']} tokens
                </span>
            </div>
            <div style="color: #64748b; margin-top: 8px; font-size: 0.85em; font-style: italic; line-height: 1.4;">
                {m['content'][:120]}{'...' if len(m['content']) > 120 else ''}
            </div>
        </div>
        """

    sections.append(f"""
    <div>
        <h4 style="color: #64748b; margin-bottom: 10px; font-size: 0.95em;">
            {icon("compress", 16)} Compressed — Tier 1 ({len(sim.tier1)})
        </h4>
        {compressed if compressed else '<div style="color: #94a3b8; padding: 12px;">Empty</div>'}
    </div>
    """)

    return f"""
    <div style="padding: 8px; max-height: 600px; overflow-y: auto;">
        {''.join(sections)}
    </div>
    """


# ---------------------------------------------------------------------------
# Global state + handlers
# ---------------------------------------------------------------------------

SIM = {"instance": None}


def get_sim():
    if SIM["instance"] is None:
        SIM["instance"] = MemCtrlSimulator(max_budget=4096)
    return SIM["instance"]


def reset_sim():
    SIM["instance"] = MemCtrlSimulator(max_budget=4096)
    sim = SIM["instance"]
    return (
        [],
        build_stats_html(sim.get_stats()),
        build_event_log_html(sim.event_log),
        build_tier_view_html(sim),
    )


def _generate_response(message, task, sim):
    """Generate a contextual simulated response based on task type and message."""
    msg = message.lower().strip()

    # Check if user is asking about something from earlier in the conversation
    recall_keywords = [
        "what was", "what were", "you mentioned", "earlier", "remind me",
        "again", "what did", "do you remember", "recall",
    ]
    if any(k in msg for k in recall_keywords):
        # Look through previous messages for relevant content
        prev_msgs = [m for m in sim.tiers["active"] if m.get("role") == "assistant"]
        if prev_msgs:
            return (
                f"Looking back through our conversation, here's what I had noted: "
                f"\"{prev_msgs[-1]['content'][:120]}...\" "
                f"MemCtrl kept this in Tier 0 (active memory) so it was instantly retrievable."
            )
        return "I don't have any previous context to recall yet. Try asking me something first!"

    # Greetings
    if msg in ("hi", "hey", "hello", "sup", "yo", "hola"):
        return (
            "Hey! I'm a simulated assistant running through MemCtrl's memory pipeline. "
            "Try asking me something — a medical question, a coding problem, or an ML concept — "
            "and watch the dashboard track every token in real time."
        )

    # Medical context
    if task == "medical":
        if "allerg" in msg:
            return "Allergy noted and auto-pinned by MemCtrl. Allergies are classified as critical medical data and will never be compressed or evicted from memory."
        if "bp" in msg or "blood pressure" in msg or "vitals" in msg:
            return "Vitals recorded. MemCtrl assigns a 1.5x retention boost to medical data, meaning these values persist 50% longer than general context before compression."
        if any(w in msg for w in ("patient", "diagnosis", "symptom", "prescri", "dosage", "medication")):
            return "Medical information captured. In a real deployment, MemCtrl auto-pins patient identifiers, dosages, and allergies — ensuring critical data survives even aggressive compression."
        return "I've noted this medical context. MemCtrl applies the highest retention priority (1.5x, 7-day window) to medical conversations to ensure nothing critical is lost."

    # Code context
    if task == "code":
        if any(w in msg for w in ("error", "bug", "crash", "fail", "exception", "traceback")):
            return "Debugging context noted. MemCtrl auto-pins connection strings, API keys, and error codes it detects in your messages — so you never have to repeat credentials mid-session."
        if any(w in msg for w in ("python", "javascript", "function", "class", "import", "def ")):
            return "Code context tracked. MemCtrl uses entity-preserving compression for code discussions — variable names, function signatures, and file paths are protected even when surrounding text is summarized."
        if any(w in msg for w in ("factorial", "sort", "algorithm", "generate", "write", "create")):
            return (
                "In a real deployment, I'd generate that code for you. Here, MemCtrl is showing you "
                "how it tracks the token cost of your request and my response. Notice the dashboard — "
                "every token is accounted for, and code-related context gets a 1.3x retention boost."
            )
        return "Code context captured with 1.3x retention boost and 3-day window. Technical details like file paths, error messages, and connection strings are auto-pinned."

    # Tutoring context
    if task == "tutoring":
        if any(w in msg for w in ("explain", "what is", "what are", "how does", "how do", "tell me about")):
            return (
                "Great question! In a real session, I'd give you a full explanation. "
                "What MemCtrl demonstrates here is how educational Q&A accumulates tokens — "
                "and how it compresses earlier explanations while keeping key definitions intact, "
                "so you get 5.7x longer tutoring sessions within the same token budget."
            )
        if any(w in msg for w in ("gradient", "neural", "model", "train", "learning rate", "loss")):
            return "ML concept noted. MemCtrl applies 1.2x retention to tutoring content, keeping definitions and formulas pinned while compressing conversational filler."
        return "Tutoring context tracked with 1.2x retention. MemCtrl preserves key concepts and definitions while compressing back-and-forth filler, stretching your learning sessions further."

    # Writing context
    if task == "writing":
        return "Writing context captured. MemCtrl uses 1.0x standard retention for creative content, with entity-preserving compression that protects names, places, and key plot points."

    # General / fallback — give a helpful, varied response
    if "how" in msg and "work" in msg and "memctrl" in msg:
        return (
            "MemCtrl works in 3 steps: (1) Every message is tokenized and classified by task type. "
            "(2) Active tokens live in Tier 0 (fast LRU cache). When the budget fills, older messages are "
            "compressed with entity-preserving summarization into Tier 1. (3) Evicted data goes to Tier 2 "
            "(SQLite + FTS5) for retrieval if needed. Zero API calls — everything runs locally."
        )
    if any(w in msg for w in ("save", "money", "cost", "cheap", "expensive", "pricing", "token")):
        return (
            "MemCtrl saves money by compressing context locally instead of sending full history to the API. "
            "In benchmarks, it extends conversations 5.7x longer within the same token budget — "
            "that's real savings on every API call."
        )
    if "?" in msg:
        return (
            "Good question! This demo simulates how MemCtrl would manage tokens in a real LLM conversation. "
            "Watch the dashboard on the right — it shows exactly how many tokens your messages use, "
            "what gets pinned, and when compression kicks in. Try filling up the budget to see eviction in action!"
        )
    return (
        "Message tracked. MemCtrl classified this as general context. "
        "Try asking about a medical case, a coding bug, or an ML concept to see "
        "how different task types get different retention priorities. "
        "Or keep chatting to fill the budget and watch compression kick in!"
    )


def send_message(message, history):
    if not message.strip():
        sim = get_sim()
        return (
            "", history,
            build_stats_html(sim.get_stats()),
            build_event_log_html(sim.event_log),
            build_tier_view_html(sim),
        )

    sim = get_sim()
    sim.add_message("user", message)

    task = classify_task(message)
    response = _generate_response(message, task, sim)
    sim.add_message("assistant", response)

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    stats = sim.get_stats()
    return (
        "", history,
        build_stats_html(stats),
        build_event_log_html(sim.event_log),
        build_tier_view_html(sim),
    )


def run_scenario(scenario_name):
    SIM["instance"] = MemCtrlSimulator(max_budget=4096)
    sim = SIM["instance"]
    history = []

    if scenario_name not in SCENARIOS:
        return (
            history,
            build_stats_html(sim.get_stats()),
            build_event_log_html(sim.event_log),
            build_tier_view_html(sim),
        )

    for role, content in SCENARIOS[scenario_name]:
        sim.add_message(role, content)
        history.append({"role": role, "content": content})

    stats = sim.get_stats()
    return (
        history,
        build_stats_html(stats),
        build_event_log_html(sim.event_log),
        build_tier_view_html(sim),
    )


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

FORCE_LIGHT_CSS = """
/* Force light theme — colors from CSS custom properties */
.gradio-container, .main, .wrap, .contain,
div[class*="block"], div[class*="panel"],
.tabs, .tab-nav, .tabitem {
    background: var(--mc-bg) !important;
    color: var(--mc-text) !important;
}
body, .dark {
    background: var(--mc-bg) !important;
    --body-background-fill: var(--mc-bg) !important;
    --background-fill-primary: var(--mc-card) !important;
    --background-fill-secondary: var(--mc-highlight) !important;
    --block-background-fill: var(--mc-card) !important;
    --block-border-color: var(--mc-border) !important;
    --body-text-color: var(--mc-text) !important;
    --block-label-text-color: var(--mc-muted) !important;
    --input-background-fill: var(--mc-card) !important;
    --button-secondary-background-fill: var(--mc-highlight) !important;
    --button-secondary-text-color: var(--mc-text) !important;
    --button-primary-background-fill: var(--mc-send-bg) !important;
    --button-primary-text-color: #ffffff !important;
}
.tab-nav button {
    color: var(--mc-muted) !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: var(--mc-primary) !important;
    border-bottom-color: var(--mc-primary) !important;
}
footer { display: none !important; }

/* ---- Button effects ---- */
@keyframes btn-pop {
    0% { transform: scale(1); }
    40% { transform: scale(0.93); }
    70% { transform: scale(1.04); }
    100% { transform: scale(1); }
}
@keyframes btn-shine {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes confetti-burst {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3), 0 0 0 0 rgba(59,130,246,0.3), 0 0 0 0 rgba(16,185,129,0.3); }
    50% { box-shadow: -8px -10px 0 -2px rgba(239,68,68,0.6), 10px -8px 0 -2px rgba(59,130,246,0.6), 6px 10px 0 -2px rgba(16,185,129,0.6); }
    100% { box-shadow: -12px -16px 0 -2px rgba(239,68,68,0), 16px -14px 0 -2px rgba(59,130,246,0), 10px 16px 0 -2px rgba(16,185,129,0); }
}

/* All Gradio buttons get a tactile pop */
button.primary, button.secondary, button[class*="btn"] {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
button.primary:active, button.secondary:active, button[class*="btn"]:active {
    animation: btn-pop 0.35s ease !important;
}
button.primary:hover {
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    transform: translateY(-1px);
}
button.secondary:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    transform: translateY(-1px);
}

/* Stat card click expand */
.stat-card {
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.stat-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18) !important;
    z-index: 10 !important;
}
.stat-card:active {
    animation: btn-pop 0.3s ease;
}
.stat-detail {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.35s ease, padding 0.35s ease, opacity 0.25s ease;
    opacity: 0;
    padding: 0 12px;
    font-size: 0.78em;
    color: #475569;
    line-height: 1.5;
    border-top: 0px solid transparent;
}
.stat-card.expanded .stat-detail {
    max-height: 120px;
    opacity: 1;
    padding: 8px 12px 4px 12px;
    border-top: 1px solid rgba(0,0,0,0.06);
}

/* Reset button glow */
.reset-btn-wrap button {
    background: var(--mc-highlight) !important;
    color: var(--mc-text) !important;
    font-weight: 600 !important;
    border: 1px solid var(--mc-border) !important;
    border-radius: 12px !important;
    padding: 10px 0 !important;
    font-size: 0.95em !important;
    transition: all 0.2s ease !important;
}
.reset-btn-wrap button:hover {
    border-color: var(--mc-accent) !important;
    box-shadow: 0 2px 8px rgba(212, 130, 156, 0.2) !important;
    transform: translateY(-1px) !important;
}
.reset-btn-wrap button:active {
    animation: btn-pop 0.35s ease !important;
}
"""

# ---------------------------------------------------------------------------
# Retro decorative SVG elements (flowers, stars, checkers)
# ---------------------------------------------------------------------------

# Small daisy flower
DECO_FLOWER = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">'
    '<circle cx="14" cy="14" r="4" fill="var(--mc-accent)"/>'
    '<circle cx="14" cy="5" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="14" cy="23" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="5" cy="14" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="23" cy="14" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="7.6" cy="7.6" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="20.4" cy="7.6" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="7.6" cy="20.4" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="20.4" cy="20.4" r="4.5" fill="var(--mc-card)" stroke="var(--mc-accent)" stroke-width="1"/>'
    '<circle cx="14" cy="14" r="4" fill="var(--mc-accent)"/>'
    '</svg>'
)

# Small 4-point star
DECO_STAR = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
    '<path d="M10 0 L12 8 L20 10 L12 12 L10 20 L8 12 L0 10 L8 8 Z" fill="var(--mc-primary)" opacity="0.25"/>'
    '</svg>'
)

# Small clover/shamrock
DECO_CLOVER = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22">'
    '<circle cx="11" cy="6" r="5" fill="var(--mc-success)" opacity="0.35"/>'
    '<circle cx="6" cy="13" r="5" fill="var(--mc-success)" opacity="0.35"/>'
    '<circle cx="16" cy="13" r="5" fill="var(--mc-success)" opacity="0.35"/>'
    '<rect x="10" y="12" width="2" height="8" rx="1" fill="var(--mc-success)" opacity="0.35"/>'
    '</svg>'
)

# Checkered strip (horizontal divider)
def checkerboard_strip(cols=16, cell=18):
    rects = []
    for i in range(cols):
        color = "var(--mc-primary)" if i % 2 == 0 else "var(--mc-card)"
        opacity = "0.15" if i % 2 == 0 else "0.6"
        rects.append(f'<rect x="{i*cell}" y="0" width="{cell}" height="{cell}" '
                     f'fill="{color}" opacity="{opacity}"/>')
    w = cols * cell
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{cell}" '
            f'viewBox="0 0 {w} {cell}" preserveAspectRatio="none" '
            f'style="display:block;border-radius:4px;">{"".join(rects)}</svg>')

CHECKER_STRIP = checkerboard_strip()

# Rounded sticker badge
def sticker_badge(text, bg="var(--mc-accent)", fg="white", rotate="0"):
    return (
        f'<span style="display:inline-block; background:{bg}; color:{fg}; '
        f'padding:6px 16px; border-radius:20px; font-weight:800; font-size:0.8em; '
        f'letter-spacing:0.05em; '
        f'box-shadow: 2px 3px 0 rgba(0,0,0,0.1); text-transform:uppercase; '
        f'border: 2px solid rgba(255,255,255,0.3);">{text}</span>'
    )

MEMCTRL_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 48 48" '
    'fill="none" style="display:inline-block;vertical-align:middle;margin-right:14px;">'
    '<rect x="4" y="4" width="40" height="40" rx="12" '
    'stroke="var(--mc-primary)" stroke-width="2.5" stroke-dasharray="6 4" opacity="0.3"/>'
    '<rect x="11" y="11" width="26" height="26" rx="8" '
    'stroke="var(--mc-accent)" stroke-width="2.5" opacity="0.6"/>'
    '<rect x="18" y="18" width="12" height="12" rx="4" fill="var(--mc-primary)"/>'
    '<circle cx="24" cy="24" r="2" fill="var(--mc-card)"/>'
    '<path d="M8 24h6M34 24h6" stroke="var(--mc-primary)" stroke-width="1.5" '
    'stroke-linecap="round" opacity="0.25"/>'
    '<path d="M24 8v6M24 34v6" stroke="var(--mc-primary)" stroke-width="1.5" '
    'stroke-linecap="round" opacity="0.25"/>'
    '</svg>'
)

PALETTE_HEAD = """
<style>
  :root, body, .dark {
    /* Default = Light (white + pink) */
    --mc-bg: #ffffff;
    --mc-card: #ffffff;
    --mc-primary: #1a1a1a;
    --mc-accent: #d4829c;
    --mc-success: #e8a0b4;
    --mc-text: #1a1a1a;
    --mc-muted: #999999;
    --mc-border: #f0e4e8;
    --mc-highlight: #fdf2f5;
    --mc-stat1-from: #fdf0f4;
    --mc-stat1-to: #f8e0e8;
    --mc-stat1-border: #e8b8c8;
    --mc-stat2-from: #fdf0f4;
    --mc-stat2-to: #f5d8e2;
    --mc-stat2-border: #e8a0b4;
    --mc-step1-border: #d4829c;
    --mc-step2-border: #ecd5dd;
    --mc-step3-border: #e8a0b4;
    --mc-pin-bg: #fdf0f4;
    --mc-pin-border: #e8a0b4;
    --mc-pin-text: #8a2050;
    --mc-savings-bg: #fdf0f4;
    --mc-savings-border: #e8a0b4;
    --mc-savings-text: #8a2050;
    --mc-how-bg: #fdf0f4;
    --mc-how-border: #ecd5dd;
    --mc-code-bg: #fdf2f5;
    --mc-link: #c06080;
    --mc-send-bg: #1a1a1a;
    --mc-send-hover: #000000;
  }
</style>
<script>
  const PALETTES = {
    "light": {
      "--mc-bg":"#ffffff","--mc-card":"#ffffff","--mc-primary":"#1a1a1a",
      "--mc-accent":"#d4829c","--mc-success":"#e8a0b4","--mc-text":"#1a1a1a",
      "--mc-muted":"#999999","--mc-border":"#f0e4e8","--mc-highlight":"#fdf2f5",
      "--mc-stat1-from":"#fdf0f4","--mc-stat1-to":"#f8e0e8","--mc-stat1-border":"#e8b8c8",
      "--mc-stat2-from":"#fdf0f4","--mc-stat2-to":"#f5d8e2","--mc-stat2-border":"#e8a0b4",
      "--mc-step1-border":"#d4829c","--mc-step2-border":"#ecd5dd","--mc-step3-border":"#e8a0b4",
      "--mc-pin-bg":"#fdf0f4","--mc-pin-border":"#e8a0b4","--mc-pin-text":"#8a2050",
      "--mc-savings-bg":"#fdf0f4","--mc-savings-border":"#e8a0b4","--mc-savings-text":"#8a2050",
      "--mc-how-bg":"#fdf0f4","--mc-how-border":"#ecd5dd","--mc-code-bg":"#fdf2f5",
      "--mc-link":"#c06080","--mc-send-bg":"#1a1a1a","--mc-send-hover":"#000000"
    },
    "dark": {
      "--mc-bg":"#0a0a0a","--mc-card":"#141414","--mc-primary":"#e8a0b4",
      "--mc-accent":"#d4829c","--mc-success":"#e8a0b4","--mc-text":"#f0f0f0",
      "--mc-muted":"#777777","--mc-border":"#222222","--mc-highlight":"#1a1a1a",
      "--mc-stat1-from":"#1a1a1a","--mc-stat1-to":"#1e1e1e","--mc-stat1-border":"#333333",
      "--mc-stat2-from":"#1a1018","--mc-stat2-to":"#201520","--mc-stat2-border":"#3a2030",
      "--mc-step1-border":"#d4829c","--mc-step2-border":"#333333","--mc-step3-border":"#e8a0b4",
      "--mc-pin-bg":"#1a1018","--mc-pin-border":"#3a2030","--mc-pin-text":"#e8a0b4",
      "--mc-savings-bg":"#1a1018","--mc-savings-border":"#3a2030","--mc-savings-text":"#e8a0b4",
      "--mc-how-bg":"#1a1a1a","--mc-how-border":"#222222","--mc-code-bg":"#1a1a1a",
      "--mc-link":"#e8a0b4","--mc-send-bg":"#d4829c","--mc-send-hover":"#c06080"
    }
  };
  let mcDark = false;
  function toggleTheme() {
    mcDark = !mcDark;
    const p = PALETTES[mcDark ? "dark" : "light"];
    const root = document.documentElement;
    for (const [k, v] of Object.entries(p)) root.style.setProperty(k, v);
    // Also set on body for Gradio overrides
    for (const [k, v] of Object.entries(p)) document.body.style.setProperty(k, v);
    // Toggle sun/moon icons
    const suns = document.querySelectorAll(".mc-icon-sun");
    const moons = document.querySelectorAll(".mc-icon-moon");
    suns.forEach(el => el.style.display = mcDark ? "none" : "block");
    moons.forEach(el => el.style.display = mcDark ? "block" : "none");
  }
</script>
"""

with gr.Blocks(
    title="MemCtrl — See Your Tokens Work",
) as app:

    # ---- THEME TOGGLE (sun/moon) ----
    gr.HTML("""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 8px; padding: 0 4px;">
        <button onclick="toggleTheme()" title="Toggle theme"
                style="background: var(--mc-highlight); border: 1px solid var(--mc-border);
                       border-radius: 50%; width: 40px; height: 40px; cursor: pointer;
                       display: flex; align-items: center; justify-content: center;
                       transition: all 0.3s ease; padding: 0;"
                onmouseover="this.style.transform='scale(1.1)'; this.style.borderColor='var(--mc-accent)'"
                onmouseout="this.style.transform='scale(1)'; this.style.borderColor='var(--mc-border)'">
            <!-- Sun icon (shown in light mode) -->
            <svg class="mc-icon-sun" xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                 viewBox="0 0 24 24" fill="none" stroke="var(--mc-text)" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round" style="display:block;">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
            <!-- Moon icon (shown in dark mode) -->
            <svg class="mc-icon-moon" xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                 viewBox="0 0 24 24" fill="none" stroke="var(--mc-text)" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round" style="display:none;">
                <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
            </svg>
        </button>
    </div>
    """)

    # ---- HERO ----
    gr.HTML(f"""
    <div style="text-align: center; padding: 56px 24px 44px 24px; background: var(--mc-card);
                border-radius: 24px; margin-bottom: 0; border: 1px solid var(--mc-border);
                position: relative; overflow: hidden;">
        <!-- Pink blob decorations -->
        <div style="position:absolute; top:-20px; right:80px; width:180px; height:180px; border-radius:50%;
                    background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%);
                    opacity:0.12; pointer-events:none;"></div>
        <div style="position:absolute; bottom:-30px; left:60px; width:150px; height:150px; border-radius:50%;
                    background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%);
                    opacity:0.10; pointer-events:none;"></div>
        <div style="position:absolute; top:40%; left:-20px; width:100px; height:100px; border-radius:50%;
                    background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%);
                    opacity:0.06; pointer-events:none;"></div>

        <div style="display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px;">
            {MEMCTRL_LOGO}
        </div>
        <h1 style="font-size: 3.2em; font-weight: 800; color: var(--mc-text); margin: 0;
                   letter-spacing: -0.02em; text-transform: uppercase; line-height: 1.1;">
            EVERY TOKEN.<br>ACCOUNTED FOR.
        </h1>
        <p style="font-size: 1.1em; color: var(--mc-muted); margin: 16px auto 0 auto; line-height: 1.6;
                  max-width: 500px;">
            Watch your LLM memory being managed in real time.
            Save money. Keep context. Zero API calls.
        </p>
        <div style="display: inline-flex; gap: 10px; margin-top: 24px; flex-wrap: wrap; justify-content: center;">
            <span style="background: var(--mc-accent); color: white; padding: 6px 16px;
                         border-radius: 20px; font-size: 0.8em; font-weight: 600;">5.7X LONGER CHATS</span>
            <span style="background: var(--mc-primary); color: white; padding: 6px 16px;
                         border-radius: 20px; font-size: 0.8em; font-weight: 600;">88% RECALL</span>
            <span style="background: var(--mc-accent); color: white; padding: 6px 16px;
                         border-radius: 20px; font-size: 0.8em; font-weight: 600;">$0 COMPRESSION</span>
        </div>
    </div>
    """)

    # ---- HOW IT WORKS (minimal flat cards) ----
    gr.HTML(f"""
    <div style="padding: 40px 24px; position: relative; background: var(--mc-highlight);
                border-radius: 0 0 24px 24px; margin-bottom: 16px;">
        <!-- Pink blob decorators -->
        <div style="position:absolute; top:-30px; right:60px; width:140px; height:140px; border-radius:50%;
                    background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%);
                    opacity:0.12; pointer-events:none;"></div>
        <div style="position:absolute; bottom:20px; left:30px; width:100px; height:100px; border-radius:50%;
                    background: radial-gradient(circle, var(--mc-accent) 0%, transparent 70%);
                    opacity:0.08; pointer-events:none;"></div>

        <h2 style="text-align: center; color: var(--mc-text); font-size: 1.5em; font-weight: 800;
                   margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.04em;">
            How It Works
        </h2>
        <p style="text-align: center; color: var(--mc-muted); margin: 0 0 32px 0;
                  font-size: 1.05em;">
            Three steps. Zero complexity. All local.
        </p>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">

            <!-- Step 1: Record -->
            <div style="background: var(--mc-card); border-radius: 16px; padding: 28px 20px 24px 20px;
                        text-align: center; border: 1px solid var(--mc-border);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: var(--mc-highlight);
                            display: flex; align-items: center; justify-content: center;
                            margin: 0 auto 16px auto;">
                    {icon("add", 24)}
                </div>
                <div style="font-size: 1.1em; font-weight: 700; color: var(--mc-text);
                            margin-bottom: 8px;">Record</div>
                <div style="font-size: 0.9em; color: var(--mc-muted); line-height: 1.5;">
                    Every message gets tokenized, classified, and stored in active memory
                </div>
            </div>

            <!-- Step 2: Compress -->
            <div style="background: var(--mc-card); border-radius: 16px; padding: 28px 20px 24px 20px;
                        text-align: center; border: 1px solid var(--mc-border);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: var(--mc-highlight);
                            display: flex; align-items: center; justify-content: center;
                            margin: 0 auto 16px auto;">
                    {icon("compress", 24)}
                </div>
                <div style="font-size: 1.1em; font-weight: 700; color: var(--mc-text);
                            margin-bottom: 8px;">Compress</div>
                <div style="font-size: 0.9em; color: var(--mc-muted); line-height: 1.5;">
                    Older messages get summarized. Passwords, dosages & URLs stay pinned
                </div>
            </div>

            <!-- Step 3: Optimize -->
            <div style="background: var(--mc-card); border-radius: 16px; padding: 28px 20px 24px 20px;
                        text-align: center; border: 1px solid var(--mc-border);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background: var(--mc-highlight);
                            display: flex; align-items: center; justify-content: center;
                            margin: 0 auto 16px auto;">
                    {icon("money", 24)}
                </div>
                <div style="font-size: 1.1em; font-weight: 700; color: var(--mc-text);
                            margin-bottom: 8px;">Optimize</div>
                <div style="font-size: 0.9em; color: var(--mc-muted); line-height: 1.5;">
                    Get a perfect message list that fits your budget. Every API call costs less
                </div>
            </div>

        </div>
    </div>
    """)

    # ---- MAIN CONTENT ----
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Interactive Demo"):
                gr.HTML("""
                <p style="color: var(--mc-muted); font-weight: 500; margin: 4px 0 8px 0;">
                    Try a pre-built scenario or type your own messages:
                </p>
                """)
                with gr.Row():
                    scenario_medical = gr.Button("Patient Consult", size="sm", variant="secondary")
                    scenario_code = gr.Button("Debug Session", size="sm", variant="secondary")
                    scenario_tutor = gr.Button("ML Tutoring", size="sm", variant="secondary")

                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a message to see MemCtrl in action...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Tab("X-Ray Mode"):
                gr.HTML("""
                <p style="color: var(--mc-muted); font-weight: 500; margin: 4px 0 8px 0;">
                    See exactly what MemCtrl does with every message:
                </p>
                """)
                event_log_html = gr.HTML(
                    value=(
                        '<div style="color: #94a3b8; text-align: center; padding: 40px; '
                        'background: white; border-radius: 12px; border: 1px solid #e2e8f0;">'
                        'Send a message to see what MemCtrl does behind the scenes...</div>'
                    ),
                )

            with gr.Tab("Tier View"):
                gr.HTML("""
                <p style="color: var(--mc-muted); font-weight: 500; margin: 4px 0 8px 0;">
                    Watch messages flow through the memory tiers:
                </p>
                """)
                tier_view_html = gr.HTML(
                    value=(
                        '<div style="color: #94a3b8; text-align: center; padding: 40px; '
                        'background: white; border-radius: 12px; border: 1px solid #e2e8f0;">'
                        'Start a conversation to see the tier visualization...</div>'
                    ),
                )

        with gr.Column(scale=2):
            gr.HTML(f"""
            <h3 style="margin: 0 0 10px 0; color: var(--mc-text); font-size: 1.1em; font-weight: 700;">
                {icon("chart", 18)} Live Token Dashboard
            </h3>
            """)
            stats_html = gr.HTML(value=build_stats_html(MemCtrlSimulator().get_stats()))
            gr.HTML('<div style="height: 8px;"></div>')
            with gr.Column(elem_classes=["reset-btn-wrap"]):
                reset_btn = gr.Button(
                    "Start Fresh",
                    variant="secondary",
                    size="lg",
                )

    # ---- FOOTER ----
    gr.HTML(f"""
    <div style="text-align: center; color: var(--mc-muted); font-size: 0.88em;
                padding: 28px 20px; margin-top: 16px;">
        <p style="margin: 0 0 12px 0; font-weight: 700; font-size: 0.95em;
                  color: var(--mc-text);">
            MemCtrl is open source and free.
        </p>
        <div style="display: inline-flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; justify-content: center;">
            <a href="https://github.com/KamalasankariS/MemCtrl" target="_blank"
               style="color: white; text-decoration: none; background: var(--mc-primary);
                      padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85em;">
                GitHub</a>
            <a href="https://pypi.org/project/memctrl-llm/" target="_blank"
               style="color: white; text-decoration: none; background: var(--mc-accent);
                      padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85em;">
                PyPI</a>
            <code style="background: var(--mc-highlight); padding: 6px 14px; border-radius: 20px;
                         color: var(--mc-text); font-size: 0.85em; font-weight: 600;">pip install memctrl-llm</code>
        </div>
        <p style="margin: 0; color: var(--mc-muted); opacity: 0.6; font-size: 0.82em;">
            Your API keys are never sent to us. Runs locally. No hosted service. No fees.
        </p>
    </div>
    """)

    # Wire events
    all_outputs = [chatbot, stats_html, event_log_html, tier_view_html]
    msg_outputs = [msg_input, chatbot, stats_html, event_log_html, tier_view_html]

    msg_input.submit(send_message, [msg_input, chatbot], msg_outputs)
    send_btn.click(send_message, [msg_input, chatbot], msg_outputs)
    reset_btn.click(reset_sim, outputs=all_outputs)

    scenario_medical.click(
        lambda: run_scenario("Patient Consult"), outputs=all_outputs,
    )
    scenario_code.click(
        lambda: run_scenario("Debug Session"), outputs=all_outputs,
    )
    scenario_tutor.click(
        lambda: run_scenario("ML Tutoring"), outputs=all_outputs,
    )

app.launch(css=FORCE_LIGHT_CSS, head=PALETTE_HEAD)
