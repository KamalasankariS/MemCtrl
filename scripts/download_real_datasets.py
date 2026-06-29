"""Download real datasets and create synthetic data for task classification training."""

import json
import random
import hashlib
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def download_writing_prompts():
    print("\n[1/5] Loading WritingPrompts (real data)...")
    try:
        dataset = load_dataset("writingPrompts/writingPrompts", split="train")
        conversations = []
        for i, example in enumerate(tqdm(dataset, total=5000, desc="Writing")):
            if i >= 5000:
                break
            prompt = example.get("prompt", "")
            if len(prompt) > 50:
                conversations.append({
                    "text": prompt[:500],
                    "label": "writing",
                    "source": "WritingPrompts",
                    "is_synthetic": False,
                    "id": hashlib.md5(prompt.encode()).hexdigest(),
                })
        print(f"  Loaded {len(conversations)} writing prompts")
        return conversations
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def download_gsm8k():
    print("\n[2/5] Loading GSM8K (real data)...")
    try:
        dataset = load_dataset("gsm8k", "main", split="train")
        conversations = []
        for i, example in enumerate(tqdm(dataset, total=5000, desc="Tutoring")):
            if i >= 5000:
                break
            question = example.get("question", "")
            if len(question) > 30:
                conversations.append({
                    "text": question[:500],
                    "label": "tutoring",
                    "source": "GSM8K",
                    "is_synthetic": False,
                    "id": hashlib.md5(question.encode()).hexdigest(),
                })
        print(f"  Loaded {len(conversations)} tutoring questions")
        return conversations
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def download_code_search_net():
    print("\n[3/5] Trying CodeSearchNet (real data)...")
    try:
        dataset = load_dataset("code_search_net", "python", split="train")
        conversations = []
        for i, example in enumerate(tqdm(dataset, total=5000, desc="Code")):
            if i >= 5000:
                break
            docstring = example.get("func_documentation_string", "")
            if len(docstring) > 50:
                conversations.append({
                    "text": docstring[:500],
                    "label": "code",
                    "source": "CodeSearchNet",
                    "is_synthetic": False,
                    "id": hashlib.md5(docstring.encode()).hexdigest(),
                })
        print(f"  Loaded {len(conversations)} code examples")
        return conversations
    except Exception as e:
        print(f"  Failed: {e}")
        return _create_synthetic_code()


def _create_synthetic_code():
    print("  Creating synthetic programming examples...")
    templates = [
        "Getting {error} when calling {function} in {language}. The traceback shows {location}.",
        "My {language} {component} throws {error}. How can I resolve this?",
        "{error} at {location}. Working on {task} using {language} and {library}.",
        "How do I {task} in {language}? The {library} documentation isn't clear about {error}.",
        "Debugging {error} in my {language} {component}. Stack trace points to {location}.",
        "{language} function {function} returns {error} instead of expected output.",
        "Help with {library} in {language}: trying to {task} but encountering {error}.",
        "Question about {component} in {language}: {error} when attempting {task}.",
        "Why does {function} fail with {error}? Using {language} version 3.9.",
        "Trying to {task} but get {error}. My setup: {language}, {library}.",
        "{error} appears when I {task}. {component} seems broken.",
        "Looking for solution to {error} in {language}. Happens at {location}.",
        "Exception {error} raised during {task}. Using {library} framework.",
        "Problem with {component}: {error} thrown unexpectedly.",
        "Need help resolving {error} in {language} application.",
    ]
    errors = [
        "TypeError: expected str, got int", "SyntaxError: invalid syntax",
        "IndexError: list index out of range", "KeyError: key not found",
        "AttributeError: no such attribute", "ValueError: invalid literal",
        "RuntimeError: recursion depth exceeded", "ImportError: module not found",
        "NameError: name not defined", "ZeroDivisionError: division by zero",
        "FileNotFoundError: file missing", "PermissionError: access denied",
        "TimeoutError: operation timed out",
    ]
    functions = ["process_data", "parse_json", "connect_db", "read_file", "make_request", "sort_array", "handle_error", "validate_input", "fetch_records", "transform_data", "calculate_sum", "filter_items"]
    tasks = ["parse JSON from API response", "connect to PostgreSQL database", "read and process CSV files", "make authenticated HTTP requests", "sort nested dictionaries", "handle async operations", "validate user input", "implement caching logic", "process uploaded files", "query database efficiently"]
    languages = ["Python", "JavaScript", "Java", "TypeScript", "C++", "Go", "Rust", "Ruby", "PHP", "Swift"]
    libraries = ["pandas", "numpy", "requests", "flask", "django", "react", "express", "axios", "lodash", "matplotlib"]
    components = ["parser", "controller", "service", "model", "view", "router", "middleware", "handler", "utility", "helper"]
    locations = ["line 45", "main.py:23", "utils.js:67", "models/user.py", "components/Header.tsx", "index.html:12"]

    conversations = []
    seen_hashes = set()

    while len(conversations) < 5000:
        template = random.choice(templates)
        text = template.format(
            error=random.choice(errors), function=random.choice(functions),
            task=random.choice(tasks), language=random.choice(languages),
            library=random.choice(libraries), component=random.choice(components),
            location=random.choice(locations),
        )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        conversations.append({
            "text": text, "label": "code", "source": "synthetic",
            "is_synthetic": True, "id": text_hash,
        })

    print(f"  Created {len(conversations)} unique synthetic programming examples")
    return conversations


def _create_synthetic_medical():
    print("\n[4/5] Creating synthetic health examples...")
    templates = [
        "Experiencing {symptom1} and {symptom2} for {duration}. History includes {condition}.",
        "Chief complaint: {symptom1} lasting {duration}. Also {symptom2}. Known {condition}.",
        "Been having {symptom1} for {duration}. Recently noticed {symptom2}. Have {condition}.",
        "Concerned about {symptom1}. Started {duration} ago. Also {symptom2}. History: {condition}.",
        "Allergic to {medication1}. Currently taking {medication2} for {condition}. Having {symptom1}.",
        "Diagnosis of {condition}. Treatment: {medication2}. Now experiencing {symptom1}.",
        "Emergency: severe {symptom1} with {symptom2}. Has {condition}, takes {medication1}.",
        "Question about {condition} management. Having {symptom1} despite {medication2}.",
        "Symptoms include {symptom1} and {symptom2}. This began {duration}.",
        "Experiencing {symptom1}. Taking {medication2} for {condition}.",
        "New {symptom1} symptoms appeared {duration}. History of {condition}.",
        "Worsening {symptom1} over {duration}. Also {symptom2}.",
        "Consultation needed for {symptom1} and {symptom2}.",
        "Follow-up: {symptom1} persists after {medication2} treatment.",
        "Questions about {symptom1} management with {condition}.",
    ]
    symptoms = [
        "persistent chest discomfort", "severe headache", "high fever (103F)", "chronic cough",
        "extreme fatigue", "nausea and vomiting", "dizziness", "shortness of breath",
        "sharp abdominal discomfort", "joint swelling", "skin rash", "blurred vision",
        "irregular heartbeat", "numbness in extremities", "muscle weakness", "tremors",
    ]
    conditions = [
        "type 2 diabetes", "hypertension", "asthma", "rheumatoid arthritis",
        "seasonal allergies", "coronary artery disease", "COPD", "chronic migraine",
        "hypothyroidism", "anxiety disorder", "depression", "osteoporosis",
    ]
    medications = [
        "penicillin", "aspirin", "ibuprofen", "amoxicillin",
        "metformin", "lisinopril", "atorvastatin", "albuterol",
        "levothyroxine", "omeprazole", "sertraline", "prednisone",
    ]
    durations = ["2 days", "1 week", "3 weeks", "2 months", "6 months", "several years", "3 days", "10 days"]

    conversations = []
    seen_hashes = set()

    while len(conversations) < 5000:
        template = random.choice(templates)
        text = template.format(
            symptom1=random.choice(symptoms), symptom2=random.choice(symptoms),
            condition=random.choice(conditions), medication1=random.choice(medications),
            medication2=random.choice(medications), duration=random.choice(durations),
        )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        conversations.append({
            "text": text, "label": "medical", "source": "synthetic",
            "is_synthetic": True, "id": text_hash,
        })

    print(f"  Created {len(conversations)} unique synthetic health examples")
    return conversations


def _create_synthetic_general():
    print("\n[5/5] Creating synthetic casual examples...")
    templates = [
        "Hey! How's your {time_period} going? Did you {activity} yet?",
        "What do you think about {topic}? I heard it's {adjective}.",
        "Planning to {activity} this {time_period}. Want to join?",
        "Have you tried {place}? Thinking about going {time_period}.",
        "Did you see {event}? It was so {adjective}!",
        "How was your {time_period}? Mine was pretty {adjective}.",
        "Any recommendations for {topic}? Looking for something {adjective}.",
        "Curious about your thoughts on {topic}. Seems {adjective}.",
        "Wondering if you've checked out {place}. Heard it's {adjective}.",
        "Quick question: what's your take on {topic}?",
        "Been meaning to {activity} but haven't had time.",
        "Saw {event} yesterday, really {adjective} experience.",
        "Looking for suggestions on {topic}, any ideas?",
        "What's your opinion about {event}?",
        "Thinking of trying {activity} sometime soon.",
    ]
    time_periods = ["weekend", "day", "week", "morning", "evening", "vacation", "holiday", "break", "month"]
    activities = ["watch that new movie", "try the new restaurant", "go hiking", "play the new game", "go shopping", "check out the exhibition", "visit the museum", "attend the concert", "read that book", "learn guitar"]
    topics = ["the weather forecast", "that new show on Netflix", "the latest news", "the game last night", "that viral video", "current events", "the new iPhone", "that trending topic", "climate change", "travel plans"]
    adjectives = ["interesting", "amazing", "disappointing", "surprising", "fun", "boring", "exciting", "weird", "cool", "odd"]
    places = ["that new cafe downtown", "the shopping mall", "the park", "that Italian restaurant", "the beach", "the gym", "the library", "the farmers market", "that bookstore"]
    events = ["the basketball game", "that concert last night", "the award show", "that event downtown", "the presentation", "the ceremony", "the festival", "the parade"]

    conversations = []
    seen_hashes = set()

    while len(conversations) < 5000:
        template = random.choice(templates)
        text = template.format(
            time_period=random.choice(time_periods), activity=random.choice(activities),
            topic=random.choice(topics), adjective=random.choice(adjectives),
            place=random.choice(places), event=random.choice(events),
        )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        conversations.append({
            "text": text, "label": "general", "source": "synthetic",
            "is_synthetic": True, "id": text_hash,
        })

    print(f"  Created {len(conversations)} unique synthetic casual examples")
    return conversations


def _remove_duplicates(all_data):
    seen_ids = set()
    unique_data = []
    for item in all_data:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_data.append(item)

    removed = len(all_data) - len(unique_data)
    if removed > 0:
        print(f"\nRemoved {removed} duplicate examples")
    return unique_data


def main():
    print("=" * 70)
    print("DOWNLOADING TASK CLASSIFICATION DATASETS")
    print("=" * 70)

    all_data = []
    all_data.extend(download_writing_prompts())
    all_data.extend(download_gsm8k())
    all_data.extend(download_code_search_net())
    all_data.extend(_create_synthetic_medical())
    all_data.extend(_create_synthetic_general())

    all_data = _remove_duplicates(all_data)
    random.shuffle(all_data)

    total = len(all_data)
    real_count = sum(1 for item in all_data if not item["is_synthetic"])
    synthetic_count = total - real_count

    print(f"\nTotal: {total:,} (Real: {real_count:,}, Synthetic: {synthetic_count:,})")

    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    dataset = {
        "train": all_data[:train_end],
        "val": all_data[train_end:val_end],
        "test": all_data[val_end:],
    }

    output_file = "data/task_classifier_data.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"Split: {len(dataset['train']):,} train / {len(dataset['val']):,} val / {len(dataset['test']):,} test")
    print("=" * 70)


if __name__ == "__main__":
    main()
