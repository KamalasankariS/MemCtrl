"""Real-world evaluation: Does MemCtrl actually save tokens and preserve quality?

Simulates realistic multi-turn conversations with planted recall facts,
then measures three things:

1. TOKEN SAVINGS — How many tokens does optimize() use vs sending everything raw?
2. RECALL ACCURACY — Can the LLM still answer questions about earlier context
   after compression/eviction?
3. ENDURANCE — At what message count does raw context fail but memctrl keeps working?

Usage:
    # Dry run (no LLM, measures token savings only):
    python scripts/eval_real_world.py

    # With a real LLM (measures quality too):
    python scripts/eval_real_world.py --provider anthropic --api-key sk-...
    python scripts/eval_real_world.py --provider openai --api-key sk-...
    python scripts/eval_real_world.py --provider ollama
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from memctrl import MemoryController
from memctrl.tokenizer import count_tokens


# ---------------------------------------------------------------------------
# Realistic conversation scenarios with planted recall facts
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Flask API Debugging Session",
        "task_type": "code",
        "description": "Developer building a patient records API, hits bugs, asks for help",
        "messages": [
            ("user", "I'm building a Flask REST API for a hospital patient records system."),
            ("assistant", "Great choice. Flask is lightweight and perfect for REST APIs. What database are you using?"),
            ("user", "PostgreSQL. The connection string is postgresql://admin:securepass123@db.hospital.internal:5432/patients"),
            ("assistant", "Good. Let me help you set up SQLAlchemy with that PostgreSQL connection. First, install flask-sqlalchemy."),
            ("user", "Done. Now I need a Patient model with fields: name, date_of_birth, diagnosis, blood_type, and allergies."),
            ("assistant", "Here's your Patient model with those fields. I've added proper column types — Text for diagnosis and allergies since those can be long."),
            ("user", "I'm getting a weird error: sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not translate host name 'db.hospital.internal'"),
            ("assistant", "That DNS resolution error means your app can't reach the database host. Are you running locally or in the hospital's network?"),
            ("user", "Locally. I need to use localhost for development. The dev database runs on port 5433."),
            ("assistant", "Change your connection string to postgresql://admin:securepass123@localhost:5433/patients_dev for local development."),
            ("user", "Works now. Next I need CRUD endpoints. Start with POST /patients to create a new record."),
            ("assistant", "Here's the POST endpoint. It validates required fields, creates the Patient object, and returns 201 with the new patient ID."),
            ("user", "I also need GET /patients/<id> with role-based access control. Only doctors and nurses should see diagnosis and allergies."),
            ("assistant", "I'll add a role check decorator. Doctors get full records, nurses see everything except detailed diagnosis notes, and admins see metadata only."),
            ("user", "The hospital requires all API responses to include a HIPAA compliance header: X-HIPAA-Audit: true"),
            ("assistant", "I'll add an after_request hook that sets that header on every response. This ensures HIPAA audit compliance across all endpoints."),
            ("user", "Now I need to add a search endpoint. GET /patients/search?blood_type=O-negative&allergy=penicillin"),
            ("assistant", "Here's the search endpoint with query parameter filtering. It supports filtering by blood_type, allergy substring match, and date_of_birth range."),
            ("user", "I want to add pagination too. Default 20 per page."),
            ("assistant", "Added page and per_page query params with defaults of page=1, per_page=20. The response includes total_count and next_page URL."),
            ("user", "Let me add a PUT endpoint for updating patient records. Only the fields that are sent should be updated."),
            ("assistant", "Here's the PATCH-style PUT endpoint. It only updates fields present in the request body, leaving others unchanged."),
            ("user", "I need to log every record access for HIPAA. Who accessed what record and when."),
            ("assistant", "I'll create an AuditLog model and a decorator that automatically logs every patient record access with user_id, patient_id, action, and timestamp."),
            ("user", "The hospital IT team says we need to encrypt the diagnosis and allergies fields at rest. They suggested using Fernet encryption."),
            ("assistant", "I'll add a Fernet encryption mixin. The diagnosis and allergies columns will store encrypted bytes, with transparent decrypt on read."),
            ("user", "Now I need to write tests. What's the test setup look like?"),
            ("assistant", "Use pytest with a test SQLite database. Create a fixture that sets up the app with testing config and yields a test client."),
            ("user", "Can you add rate limiting? Max 100 requests per minute per API key."),
            ("assistant", "I'll use flask-limiter with a Redis backend. Default limit is 100/minute, keyed by the API key in the Authorization header."),
            ("user", "One more thing — I need a DELETE endpoint but it should be soft delete. Set a deleted_at timestamp instead of actually removing the row."),
            ("assistant", "Added soft delete with a deleted_at column. The DELETE endpoint sets the timestamp, and all query endpoints filter out soft-deleted records by default."),
            ("user", "I'm getting a new error now: psycopg2.errors.UndefinedColumn: column 'deleted_at' does not exist"),
            ("assistant", "You need to run a migration to add the deleted_at column. Use flask-migrate: flask db migrate -m 'add soft delete' then flask db upgrade."),
            ("user", "The migration worked but now my search endpoint is returning deleted patients."),
            ("assistant", "Add .filter(Patient.deleted_at.is_(None)) to your base query in the search endpoint. This excludes soft-deleted records."),
            ("user", "Perfect. Let me add export functionality. GET /patients/export?format=csv should download all records as CSV."),
            ("assistant", "Here's the export endpoint. It streams the CSV using a generator to handle large datasets without loading everything into memory."),
            ("user", "The hospital wants a webhook that fires whenever a patient record is created or updated. They use a notification service at https://notify.hospital.internal/webhook"),
            ("assistant", "I'll add a post-commit hook that sends a POST request to the webhook URL with the patient ID and action type. It runs async to not block the API response."),
        ],
        "recall_questions": [
            {
                "question": "What is the PostgreSQL connection string I gave you at the start?",
                "answer_must_contain": ["securepass123", "db.hospital.internal", "5432"],
                "planted_at_message": 2,
            },
            {
                "question": "What fields does the Patient model have?",
                "answer_must_contain": ["blood_type", "allergies", "diagnosis"],
                "planted_at_message": 4,
            },
            {
                "question": "What HIPAA header does the hospital require on all API responses?",
                "answer_must_contain": ["X-HIPAA-Audit"],
                "planted_at_message": 14,
            },
            {
                "question": "What encryption method are we using for the diagnosis field?",
                "answer_must_contain": ["Fernet"],
                "planted_at_message": 22,
            },
            {
                "question": "What is the rate limit we set up?",
                "answer_must_contain": ["100"],
                "planted_at_message": 26,
            },
            {
                "question": "What is the webhook URL the hospital uses for notifications?",
                "answer_must_contain": ["notify.hospital.internal"],
                "planted_at_message": 35,
            },
        ],
    },
    {
        "name": "Machine Learning Tutoring Session",
        "task_type": "tutoring",
        "description": "Student learning gradient descent, step by step",
        "messages": [
            ("user", "I'm studying for my ML exam. Can you help me understand gradient descent?"),
            ("assistant", "Of course. Gradient descent is an optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent."),
            ("user", "What's the learning rate? My professor said it's usually set to 0.01."),
            ("assistant", "The learning rate (alpha) controls step size. Your professor's suggestion of 0.01 is a common starting point. Too large and you overshoot, too small and convergence is slow."),
            ("user", "Can you walk me through the math? Start with a simple function f(x) = x^2 + 4x + 4."),
            ("assistant", "Sure. The derivative is f'(x) = 2x + 4. The update rule is x_new = x_old - alpha * f'(x_old). The minimum is at x = -2 where f'(x) = 0."),
            ("user", "What about for neural networks? My exam covers backpropagation."),
            ("assistant", "Backpropagation applies the chain rule to compute gradients through the network. For each layer, the gradient of the loss with respect to the weights is computed by multiplying local gradients backward from the output."),
            ("user", "My professor gave us a specific formula for the exam. The weight update for layer l is: w_l = w_l - alpha * (dL/dw_l) where dL/dw_l = delta_l * a_{l-1}^T"),
            ("assistant", "That's the standard backprop update. delta_l is the error signal at layer l, and a_{l-1} is the activation from the previous layer. The transpose ensures the matrix dimensions align for the outer product."),
            ("user", "What's the difference between batch, mini-batch, and stochastic gradient descent?"),
            ("assistant", "Batch GD uses the entire dataset per update. Mini-batch uses a subset (typically 32, 64, or 128 samples). Stochastic GD uses a single sample. Mini-batch is most common in practice — it balances noise and computation."),
            ("user", "My exam also covers Adam optimizer. How is it different?"),
            ("assistant", "Adam combines momentum (moving average of gradients) with RMSProp (moving average of squared gradients). It adapts the learning rate per-parameter. Default hyperparameters: beta1=0.9, beta2=0.999, epsilon=1e-8."),
            ("user", "What about vanishing gradients? My professor said that's why ReLU replaced sigmoid."),
            ("assistant", "Sigmoid squashes outputs to (0,1), and its gradient maxes out at 0.25. Through many layers, these small gradients multiply together and vanish. ReLU has gradient 1 for positive inputs, so gradients flow unchanged through the network."),
            ("user", "The exam has a section on regularization. Cover L1 and L2 for me."),
            ("assistant", "L1 (Lasso) adds lambda * sum(|w|) to the loss. It drives weights to exactly zero, enabling feature selection. L2 (Ridge) adds lambda * sum(w^2). It shrinks weights toward zero but doesn't eliminate them. L2 is more common in deep learning as weight decay."),
            ("user", "My professor said the exam will have a question about the bias-variance tradeoff."),
            ("assistant", "Bias is underfitting — the model is too simple. Variance is overfitting — the model memorizes training data. The sweet spot minimizes total error = bias^2 + variance + irreducible noise. Regularization reduces variance at the cost of slightly more bias."),
            ("user", "One more topic — dropout. He said the exam dropout rate is 0.5 for hidden layers and we need to know about inverted dropout."),
            ("assistant", "Dropout randomly zeros out neurons during training with probability p. Inverted dropout scales the remaining activations by 1/(1-p) during training so that expected values stay consistent — no scaling needed at test time. Your professor's 0.5 rate is the original Hinton recommendation."),
            ("user", "Can you help me with a practice problem? If I have a 2-layer network with 784 inputs, 128 hidden units with ReLU, and 10 output units with softmax, how many trainable parameters does it have?"),
            ("assistant", "Layer 1: 784 * 128 weights + 128 biases = 100,480. Layer 2: 128 * 10 weights + 10 biases = 1,290. Total: 101,770 trainable parameters."),
            ("user", "What loss function should I use for this multi-class classification?"),
            ("assistant", "Cross-entropy loss (also called categorical cross-entropy or log loss). For softmax output: L = -sum(y_true * log(y_pred)). In PyTorch, use nn.CrossEntropyLoss which combines log_softmax and NLLLoss."),
            ("user", "My professor also said we need to know about batch normalization and its formula."),
            ("assistant", "Batch norm normalizes each feature across the mini-batch: x_hat = (x - mean) / sqrt(var + epsilon), then scales and shifts: y = gamma * x_hat + beta. Gamma and beta are learnable. It stabilizes training and allows higher learning rates."),
            ("user", "Last topic — learning rate scheduling. He mentioned step decay and cosine annealing."),
            ("assistant", "Step decay reduces LR by a factor (e.g., 0.1) every N epochs. Cosine annealing smoothly decreases LR following a cosine curve from initial LR to near zero. Cosine is popular with warm restarts (SGDR) where LR periodically resets."),
        ],
        "recall_questions": [
            {
                "question": "What learning rate did my professor recommend as a starting point?",
                "answer_must_contain": ["0.01"],
                "planted_at_message": 2,
            },
            {
                "question": "What was the specific weight update formula my professor gave for backpropagation?",
                "answer_must_contain": ["delta_l", "a_{l-1}"],
                "planted_at_message": 8,
            },
            {
                "question": "What are the default hyperparameters for Adam optimizer?",
                "answer_must_contain": ["0.9", "0.999"],
                "planted_at_message": 12,
            },
            {
                "question": "How many trainable parameters did we calculate for the 2-layer network?",
                "answer_must_contain": ["101,770", "101770"],
                "planted_at_message": 20,
            },
            {
                "question": "What dropout rate did my professor say will be on the exam?",
                "answer_must_contain": ["0.5"],
                "planted_at_message": 18,
            },
        ],
    },
    {
        "name": "Medical Consultation Follow-up",
        "task_type": "medical",
        "description": "Patient discussing symptoms and treatment plans across a long session",
        "messages": [
            ("user", "I was diagnosed with Type 2 diabetes last month. My fasting glucose was 187 mg/dL and HbA1c was 8.2%."),
            ("assistant", "Those numbers indicate uncontrolled diabetes. Normal fasting glucose is under 100, and HbA1c target is under 7%. Your doctor will likely start medication and lifestyle changes."),
            ("user", "My doctor prescribed Metformin 500mg twice daily. She said to take it with meals to reduce stomach upset."),
            ("assistant", "Metformin is the first-line treatment for Type 2 diabetes. Taking it with meals is correct — it reduces GI side effects. Your doctor may increase the dose to 1000mg twice daily after a few weeks if tolerated."),
            ("user", "I'm also on Lisinopril 10mg for blood pressure. My BP was 145/92 at the last visit."),
            ("assistant", "Lisinopril is an ACE inhibitor — good choice because it also protects the kidneys, which is important for diabetic patients. Target BP for diabetics is under 130/80."),
            ("user", "My doctor said I need to check my blood sugar 4 times a day: fasting, before lunch, before dinner, and at bedtime."),
            ("assistant", "That's a standard monitoring schedule for newly diagnosed patients on medication. Keep a log of your readings. Your doctor will use these to adjust your Metformin dose."),
            ("user", "I also have high cholesterol. My LDL was 162 mg/dL and total cholesterol was 248 mg/dL."),
            ("assistant", "Both are elevated. For diabetic patients, the LDL target is under 100 mg/dL. Your doctor will likely add a statin — probably Atorvastatin 20mg or 40mg."),
            ("user", "She prescribed Atorvastatin 40mg at bedtime. She said to avoid grapefruit."),
            ("assistant", "Correct — grapefruit inhibits the enzyme that metabolizes Atorvastatin, which can increase side effects. Take it at bedtime because cholesterol synthesis peaks overnight."),
            ("user", "I'm allergic to sulfa drugs. My doctor has that on file. She said to mention it if I ever go to the ER."),
            ("assistant", "Important allergy to track. Sulfa drugs include sulfamethoxazole/trimethoprim (Bactrim), some diuretics like hydrochlorothiazide, and certain diabetes medications like glipizide. Always mention this to any new provider."),
            ("user", "My next appointment is in 3 months. She wants to recheck HbA1c and a full lipid panel."),
            ("assistant", "Standard follow-up timeline. The HbA1c reflects your average glucose over 2-3 months, so a 3-month recheck will show if the Metformin is working. The lipid panel will assess the Atorvastatin's effect."),
            ("user", "What about my diet? She mentioned the Mediterranean diet and counting carbs. She said to stay under 45g of carbs per meal."),
            ("assistant", "The Mediterranean diet emphasizes vegetables, whole grains, fish, and olive oil — it's well-studied for diabetes management. The 45g carb limit per meal is reasonable for blood sugar control. That's roughly 135g total carbs per day."),
            ("user", "I've been exercising 30 minutes a day, 5 days a week. Mostly walking. She said my target heart rate should be around 120 BPM."),
            ("assistant", "Walking is excellent for diabetes management. 150 minutes per week meets the ADA recommendation. The 120 BPM target puts you in a moderate intensity zone, which is ideal for blood sugar control."),
            ("user", "She also mentioned I should get an eye exam for diabetic retinopathy and a foot exam every visit."),
            ("assistant", "Annual dilated eye exams are standard for diabetic patients to catch retinopathy early. Foot exams check for neuropathy and circulation problems. Report any numbness, tingling, or wounds that don't heal."),
            ("user", "One more thing — she wants me to get a kidney function test. She mentioned microalbumin and eGFR."),
            ("assistant", "Microalbumin checks for small amounts of protein in your urine — an early sign of diabetic kidney disease. eGFR estimates how well your kidneys filter waste. Both should be checked annually. The Lisinopril you're taking helps protect kidney function."),
        ],
        "recall_questions": [
            {
                "question": "What were my initial fasting glucose and HbA1c numbers?",
                "answer_must_contain": ["187", "8.2"],
                "planted_at_message": 0,
            },
            {
                "question": "What medication was I prescribed for diabetes and at what dose?",
                "answer_must_contain": ["Metformin", "500"],
                "planted_at_message": 2,
            },
            {
                "question": "What drug allergy do I have?",
                "answer_must_contain": ["sulfa"],
                "planted_at_message": 12,
            },
            {
                "question": "What is my carb limit per meal?",
                "answer_must_contain": ["45"],
                "planted_at_message": 16,
            },
            {
                "question": "What blood pressure medication am I on and what was my last BP reading?",
                "answer_must_contain": ["Lisinopril", "145"],
                "planted_at_message": 4,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

@dataclass
class RecallResult:
    question: str
    planted_at_message: int
    answer_must_contain: List[str]
    llm_answer: Optional[str] = None
    passed: bool = False
    matched_terms: List[str] = field(default_factory=list)
    missed_terms: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    total_messages: int
    raw_tokens: int
    memctrl_tokens: int
    token_savings_pct: float
    messages_in_raw: int
    messages_in_optimized: int
    recall_results: List[RecallResult] = field(default_factory=list)
    recall_accuracy: float = 0.0


def count_messages_tokens(messages: List[Dict[str, str]], model: str) -> int:
    return sum(count_tokens(m["content"], model) for m in messages)


def build_raw_messages(conversation: List[tuple]) -> List[Dict[str, str]]:
    """Build a raw message list — everything, no compression."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    return messages


def evaluate_scenario(
    scenario: Dict,
    token_budget: int,
    tokenizer_model: str,
    llm=None,
) -> ScenarioResult:
    """Run a single scenario through raw vs memctrl and measure everything."""

    conversation = scenario["messages"]
    name = scenario["name"]

    # -- Raw approach: send everything --
    raw_messages = build_raw_messages(conversation)
    raw_tokens = count_messages_tokens(raw_messages, tokenizer_model)

    # -- MemCtrl approach: add messages then optimize --
    ctrl = MemoryController(user_id=f"eval_{name}", llm=llm)

    for role, content in conversation:
        ctrl.add_message(role, content)

    optimized = ctrl.optimize(max_tokens=token_budget)
    memctrl_tokens = count_messages_tokens(optimized, tokenizer_model)

    savings = (1 - memctrl_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0

    result = ScenarioResult(
        name=name,
        total_messages=len(conversation),
        raw_tokens=raw_tokens,
        memctrl_tokens=memctrl_tokens,
        token_savings_pct=round(savings, 1),
        messages_in_raw=len(raw_messages),
        messages_in_optimized=len(optimized),
    )

    # -- Recall test: ask checkpoint questions --
    if llm and llm.provider_name != "echo":
        for rq in scenario.get("recall_questions", []):
            recall = RecallResult(
                question=rq["question"],
                planted_at_message=rq["planted_at_message"],
                answer_must_contain=rq["answer_must_contain"],
            )

            # Add the recall question to the optimized context
            recall_messages = list(optimized)
            recall_messages.append({"role": "user", "content": rq["question"]})

            try:
                answer = llm.generate(recall_messages, max_tokens=256)
                recall.llm_answer = answer

                # Normalize: strip commas, underscores, braces for flexible matching
                answer_normalized = answer.lower().replace(",", "").replace("_", "").replace("{", "").replace("}", "")
                for term in rq["answer_must_contain"]:
                    term_normalized = term.lower().replace(",", "").replace("_", "").replace("{", "").replace("}", "")
                    if term_normalized in answer_normalized:
                        recall.matched_terms.append(term)
                    else:
                        recall.missed_terms.append(term)

                recall.passed = len(recall.missed_terms) == 0
            except Exception as e:
                recall.llm_answer = f"ERROR: {e}"

            result.recall_results.append(recall)

        total_recall = len(result.recall_results)
        passed_recall = sum(1 for r in result.recall_results if r.passed)
        result.recall_accuracy = (
            round(passed_recall / total_recall * 100, 1) if total_recall > 0 else 0
        )

    return result


# ---------------------------------------------------------------------------
# Endurance test: find the breaking point
# ---------------------------------------------------------------------------

def endurance_test(
    token_budget: int,
    tokenizer_model: str,
) -> Dict:
    """Keep adding messages until raw context exceeds budget. Show where memctrl
    still fits but raw doesn't."""

    from memctrl.llm.backend import EchoLLM

    llm = EchoLLM()
    ctrl = MemoryController(user_id="endurance_test", llm=llm)

    padding_messages = [
        "Can you help me debug this React component? It's not rendering the state correctly after a useEffect update.",
        "The issue is in the dependency array. I have [data] but data is an object reference that changes on every render.",
        "I tried useMemo but it didn't help. The component still re-renders 47 times on page load.",
        "Let me share the full component code. It's about 200 lines with three useEffect hooks and a custom reducer.",
        "The reducer handles five action types: FETCH_START, FETCH_SUCCESS, FETCH_ERROR, UPDATE_ITEM, and DELETE_ITEM.",
        "Each action updates a different part of the state object. The state shape is {items: [], loading: false, error: null, pagination: {page: 1, total: 0}}.",
        "I also have a context provider wrapping this component that passes down the dispatch function and a memoized selector.",
        "The selector function filters items by category and sorts them by date. Categories are: electronics, clothing, books, and furniture.",
        "I added a search feature that debounces input with a 300ms delay using a custom useDebounce hook.",
        "The API endpoint is /api/v2/products?category=X&sort=date&page=N&limit=20 with Bearer token auth.",
    ]

    raw_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    raw_exceeds_at = None
    memctrl_exceeds_at = None

    for i in range(200):
        msg_content = padding_messages[i % len(padding_messages)]
        role = "user" if i % 2 == 0 else "assistant"

        # Raw: keep appending
        raw_messages.append({"role": role, "content": msg_content})
        raw_tokens = count_messages_tokens(raw_messages, tokenizer_model)

        if raw_exceeds_at is None and raw_tokens > token_budget:
            raw_exceeds_at = i

        # MemCtrl: add and optimize
        ctrl.add_message(role, msg_content)
        optimized = ctrl.optimize(max_tokens=token_budget)
        memctrl_tokens = count_messages_tokens(optimized, tokenizer_model)

        if memctrl_exceeds_at is None and memctrl_tokens > token_budget:
            memctrl_exceeds_at = i

        if raw_exceeds_at and (memctrl_exceeds_at or i > raw_exceeds_at * 3):
            break

    return {
        "token_budget": token_budget,
        "raw_exceeds_at_message": raw_exceeds_at or ">200",
        "memctrl_exceeds_at_message": memctrl_exceeds_at or ">200",
        "total_messages_tested": i + 1,
        "endurance_multiplier": (
            round((memctrl_exceeds_at or 200) / raw_exceeds_at, 1)
            if raw_exceeds_at
            else "N/A"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MemCtrl Real-World Evaluation")
    parser.add_argument("--provider", default="echo", help="LLM provider (anthropic, openai, ollama, echo)")
    parser.add_argument("--api-key", default=None, help="API key (memory only, never logged)")
    parser.add_argument("--model", default=None, help="Model name (e.g. llama3, gpt-4o-mini)")
    parser.add_argument("--budget", type=int, default=2048, help="Token budget for optimize()")
    args = parser.parse_args()

    tokenizer_model = "distilbert-base-uncased"

    # Create LLM backend
    from memctrl.llm.backend import create_llm_backend
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    llm = create_llm_backend(args.provider, api_key=args.api_key, **kwargs)

    print("=" * 70)
    print("MEMCTRL REAL-WORLD EVALUATION")
    print("=" * 70)
    print(f"Provider: {llm.provider_name}")
    print(f"Token budget: {args.budget}")
    print(f"Scenarios: {len(SCENARIOS)}")
    if llm.provider_name == "echo":
        print("(Running in dry-run mode — token savings only, no recall test)")
    print()

    # -- Run each scenario --
    all_results = []
    total_raw = 0
    total_memctrl = 0

    for scenario in SCENARIOS:
        print(f"--- {scenario['name']} ---")
        print(f"    {scenario['description']}")
        print(f"    Messages: {len(scenario['messages'])}")

        result = evaluate_scenario(scenario, args.budget, tokenizer_model, llm)
        all_results.append(result)
        total_raw += result.raw_tokens
        total_memctrl += result.memctrl_tokens

        print(f"    Raw tokens:     {result.raw_tokens:,}")
        print(f"    MemCtrl tokens: {result.memctrl_tokens:,}")
        print(f"    Savings:        {result.token_savings_pct}%")
        print(f"    Messages kept:  {result.messages_in_optimized}/{result.messages_in_raw}")

        if result.recall_results:
            print(f"    Recall accuracy: {result.recall_accuracy}%")
            for r in result.recall_results:
                status = "PASS" if r.passed else "FAIL"
                print(f"      [{status}] \"{r.question[:60]}...\"")
                if r.missed_terms:
                    print(f"             Missing: {r.missed_terms}")
        print()

    # -- Summary --
    print("=" * 70)
    print("TOKEN SAVINGS SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':45s} | {'Raw':>8s} | {'MemCtrl':>8s} | {'Saved':>6s}")
    print("-" * 75)
    for r in all_results:
        print(f"{r.name:45s} | {r.raw_tokens:>8,} | {r.memctrl_tokens:>8,} | {r.token_savings_pct:>5.1f}%")
    print("-" * 75)
    total_savings = (1 - total_memctrl / total_raw) * 100 if total_raw > 0 else 0
    print(f"{'TOTAL':45s} | {total_raw:>8,} | {total_memctrl:>8,} | {total_savings:>5.1f}%")
    print()

    # -- Cost estimate --
    # GPT-4o: $2.50/1M input tokens, Claude Sonnet: $3/1M input tokens
    raw_cost_1k = total_raw / 1_000_000 * 3.0 * 1000
    memctrl_cost_1k = total_memctrl / 1_000_000 * 3.0 * 1000
    print("Estimated cost for 1,000 such conversations (at $3/1M tokens):")
    print(f"  Raw:     ${raw_cost_1k:.2f}")
    print(f"  MemCtrl: ${memctrl_cost_1k:.2f}")
    print(f"  Saved:   ${raw_cost_1k - memctrl_cost_1k:.2f}")
    print()

    # -- Endurance test --
    print("=" * 70)
    print("ENDURANCE TEST")
    print("=" * 70)
    print(f"Adding messages until raw context exceeds {args.budget} tokens...")

    endurance = endurance_test(args.budget, tokenizer_model)
    print(f"  Raw context exceeded budget at message:     {endurance['raw_exceeds_at_message']}")
    print(f"  MemCtrl still fits at message:              {endurance['memctrl_exceeds_at_message']}")
    print(f"  Endurance multiplier:                       {endurance['endurance_multiplier']}x")
    print()
    print(f"  MemCtrl lets you have {endurance['endurance_multiplier']}x longer conversations")
    print("  before hitting the same token limit.")
    print()

    # -- Recall summary (if LLM was used) --
    recall_results = [r for res in all_results for r in res.recall_results]
    if recall_results:
        print("=" * 70)
        print("RECALL ACCURACY (Does the LLM still remember after compression?)")
        print("=" * 70)
        total_q = len(recall_results)
        passed_q = sum(1 for r in recall_results if r.passed)
        print(f"  {passed_q}/{total_q} recall questions answered correctly ({passed_q/total_q*100:.0f}%)")
        print()
        failed = [r for r in recall_results if not r.passed]
        if failed:
            print("  Failed recalls:")
            for r in failed:
                print(f"    - \"{r.question[:70]}\"")
                print(f"      Missing terms: {r.missed_terms}")
                if r.llm_answer:
                    print(f"      LLM said: \"{r.llm_answer[:100]}...\"")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
