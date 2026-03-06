"""Generate large-scale diverse training data for memory injection LoRA fine-tuning.

Uses:
1. Synthetic personal facts with programmatic diversity
2. Contrastive pairs to force the model to read vector content
3. LoCoMo dataset conversations for natural personal QA (optional download)

The key anti-overfitting strategy: for the same question template (e.g., "What is my name?"),
generate 50+ different answers paired with 50+ different memory vectors. The model CANNOT
memorize a fixed answer — it must learn to extract information from the injection vector.

Usage::

    python -m scripts.generate_memory_training_data [checkpoint] [output] [count]
    python -m scripts.generate_memory_training_data checkpoints/fast_weight_1k/final data/memory_training_data.json 1000
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import torch

os.environ["BRAIN_USE_PATTERN_SEPARATION"] = "true"
os.environ["BRAIN_USE_DOPAMINERGIC_GATE"] = "true"
os.environ["BRAIN_USE_HOPFIELD_MEMORY"] = "true"
os.environ["BRAIN_USE_MODULAR_HOPFIELD"] = "true"
os.environ["BRAIN_USE_FAST_WEIGHT_MEMORY"] = "true"
os.environ["BRAIN_USE_TRANSFORMER_WM"] = "true"
os.environ["BRAIN_USE_LEARNED_FORGETTING"] = "true"

# === DIVERSE NAME/ATTRIBUTE POOLS ===
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Hank",
    "Iris", "James", "Kai", "Luna", "Marcus", "Nina", "Oscar", "Priya",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yuki", "Zara", "Albin", "Oliver", "Melvin", "Sofia", "Leo", "Mia",
    "Noah", "Ava", "Ethan", "Isabella", "Liam", "Olivia", "Mason", "Aria",
    "Lucas", "Ella", "Aiden", "Chloe", "Jackson", "Lily", "Sebastian", "Zoe",
    "Henry", "Riley",
]

AGES = list(range(16, 65))

CITIES = [
    "Stockholm", "Tokyo", "Berlin", "London", "New York", "Paris", "Sydney",
    "Toronto", "Mumbai", "Seoul", "Amsterdam", "Barcelona", "Chicago", "Dubai",
    "Helsinki", "Istanbul", "Jakarta", "Kyoto", "Lima", "Moscow", "Nairobi",
    "Oslo", "Prague", "Rome", "Singapore", "Vancouver", "Warsaw", "Zurich",
    "San Francisco", "Melbourne", "Bangkok", "Buenos Aires", "Cape Town",
    "Denver", "Edinburgh", "Florence", "Gothenburg", "Hamburg",
]

COUNTRIES = [
    "Sweden", "Japan", "Germany", "UK", "USA", "France", "Australia", "Canada",
    "India", "South Korea", "Netherlands", "Spain", "Brazil", "Mexico",
    "Finland", "Turkey", "Norway", "Italy", "Colombia", "Argentina",
]

JOBS = [
    "software engineer", "data scientist", "teacher", "nurse", "designer",
    "product manager", "freelance writer", "chef", "architect", "musician",
    "photographer", "researcher", "startup founder", "marketing manager",
    "mechanical engineer", "therapist", "lawyer", "accountant", "pilot",
    "dentist", "veterinarian", "journalist", "professor", "barista",
    "electrician", "pharmacist", "physical therapist", "artist",
]

COMPANIES = [
    "Google", "Microsoft", "a local startup", "Spotify", "Netflix", "Amazon",
    "a hospital", "a university", "their own company", "a design agency",
    "a law firm", "a restaurant", "a tech startup", "an AI company",
    "a consulting firm", "a non-profit", "a school", "a bank",
]

HOBBIES = [
    "playing guitar", "rock climbing", "painting", "cooking", "reading sci-fi",
    "playing chess", "running marathons", "photography", "gardening", "surfing",
    "playing piano", "hiking", "swimming", "yoga", "skateboarding",
    "writing poetry", "woodworking", "birdwatching", "gaming", "cycling",
    "dancing", "martial arts", "knitting", "pottery", "sailing",
]

PETS = [
    ("dog", ["Max", "Luna", "Buddy", "Biscuit", "Rex", "Bella", "Charlie",
             "Daisy", "Rocky", "Milo"]),
    ("cat", ["Whiskers", "Muffin", "Shadow", "Luna", "Simba", "Cleo",
             "Felix", "Nala", "Oliver", "Mittens"]),
    ("rabbit", ["Thumper", "Snowball", "Clover", "Hazel", "Cotton"]),
    ("hamster", ["Nibbles", "Peanut", "Gizmo", "Squeaky", "Hammy"]),
]

FOODS = [
    "sushi", "pizza", "tacos", "pasta", "curry", "ramen", "burgers",
    "pad thai", "pho", "dumplings", "steak", "paella", "biryani",
    "falafel", "croissants", "Korean BBQ", "fish and chips",
]

LANGUAGES = [
    "Japanese", "Spanish", "French", "German", "Mandarin", "Korean",
    "Portuguese", "Italian", "Russian", "Arabic", "Swedish", "Hindi",
]

PROJECTS = [
    "a job matching platform", "a social media app", "a fitness tracker",
    "a recipe sharing website", "an AI chatbot", "a music streaming service",
    "a task management tool", "a language learning app", "a budgeting app",
    "a dating app", "an e-commerce platform", "a note-taking app",
    "a news aggregator", "a portfolio website", "a booking system",
]

PROJECT_NAMES = [
    "Hirena", "TaskFlow", "FitTrack", "RecipeHub", "ChatMind", "MelodyStream",
    "Planify", "LinguaLearn", "BudgetBuddy", "Connectr", "ShopEasy",
    "NotePad Pro", "NewsWave", "ShowCase", "BookIt", "CodeBrew", "DataVault",
    "PixelArt",
]

TECH_STACKS = [
    "Python and FastAPI", "React and Node.js", "TypeScript and Next.js",
    "Django and PostgreSQL", "Flutter and Firebase", "Vue.js and Go",
    "Rust and WASM", "Swift and SwiftUI", "Kotlin and Jetpack Compose",
]

# === QUESTION TEMPLATES ===
QUESTION_TEMPLATES: dict[str, list[str]] = {
    "name": [
        "What is my name?",
        "Do you remember my name?",
        "Who am I?",
        "What did I say my name was?",
        "Can you recall my name?",
    ],
    "age": [
        "How old am I?",
        "What's my age?",
        "Do you remember how old I am?",
    ],
    "location": [
        "Where do I live?",
        "What city am I in?",
        "Where am I based?",
        "Do you remember where I live?",
    ],
    "job": [
        "What do I do for a living?",
        "What's my job?",
        "Where do I work?",
        "What is my occupation?",
    ],
    "hobby": [
        "What are my hobbies?",
        "What do I like to do in my free time?",
        "What activities do I enjoy?",
    ],
    "pet": [
        "Do I have any pets?",
        "Tell me about my pet.",
        "What's my pet's name?",
    ],
    "food": [
        "What's my favorite food?",
        "What food do I like?",
    ],
    "project": [
        "What am I building?",
        "What project am I working on?",
        "Tell me about my project.",
    ],
    "language": [
        "What language am I learning?",
    ],
    "general_recall": [
        "What do you know about me?",
        "Tell me what you remember about me.",
        "What have I told you about myself?",
        "Do you remember anything about me?",
        "What information do you have about me?",
        "Can you recall what I shared with you?",
    ],
}


def generate_single_fact_example(category: str) -> dict:
    """Generate a random (fact, question, answer) for a single category."""
    if category == "name":
        name = random.choice(NAMES)
        fact = f"My name is {name}."
        question = random.choice(QUESTION_TEMPLATES["name"])
        answer = f"Your name is {name}."
    elif category == "age":
        name = random.choice(NAMES)
        age = random.choice(AGES)
        fact = f"My name is {name} and I am {age} years old."
        question = random.choice(QUESTION_TEMPLATES["age"])
        answer = f"You are {age} years old."
    elif category == "location":
        city = random.choice(CITIES)
        country = random.choice(COUNTRIES)
        fact = f"I live in {city}, {country}."
        question = random.choice(QUESTION_TEMPLATES["location"])
        answer = f"You live in {city}, {country}."
    elif category == "job":
        job = random.choice(JOBS)
        company = random.choice(COMPANIES)
        fact = f"I work as a {job} at {company}."
        question = random.choice(QUESTION_TEMPLATES["job"])
        answer = f"You work as a {job} at {company}."
    elif category == "hobby":
        hobby = random.choice(HOBBIES)
        fact = f"I really enjoy {hobby} in my free time."
        question = random.choice(QUESTION_TEMPLATES["hobby"])
        answer = f"You enjoy {hobby}."
    elif category == "pet":
        pet_type, pet_names = random.choice(PETS)
        pet_name = random.choice(pet_names)
        fact = f"I have a {pet_type} named {pet_name}."
        question = random.choice(QUESTION_TEMPLATES["pet"])
        answer = f"You have a {pet_type} named {pet_name}."
    elif category == "food":
        food = random.choice(FOODS)
        fact = f"My favorite food is {food}."
        question = random.choice(QUESTION_TEMPLATES["food"])
        answer = f"Your favorite food is {food}."
    elif category == "project":
        project = random.choice(PROJECTS)
        proj_name = random.choice(PROJECT_NAMES)
        tech = random.choice(TECH_STACKS)
        fact = f"I'm building {project} called {proj_name} using {tech}."
        question = random.choice(QUESTION_TEMPLATES["project"])
        answer = f"You're building {project} called {proj_name} using {tech}."
    elif category == "language":
        lang = random.choice(LANGUAGES)
        fact = f"I've been learning {lang} for the past year."
        question = random.choice(QUESTION_TEMPLATES["language"])
        answer = f"You've been learning {lang}."
    else:
        raise ValueError(f"Unknown category: {category}")
    return {"facts": [fact], "question": question, "answer": answer}


def generate_multi_fact_example() -> dict:
    """Generate a (facts, question, answer) with 2-4 facts about the same person."""
    name = random.choice(NAMES)
    age = random.choice(AGES)
    city = random.choice(CITIES)
    job = random.choice(JOBS)
    hobby = random.choice(HOBBIES)

    all_facts = [
        (f"My name is {name}.", f"Your name is {name}"),
        (f"I am {age} years old.", f"you are {age} years old"),
        (f"I live in {city}.", f"you live in {city}"),
        (f"I work as a {job}.", f"you work as a {job}"),
        (f"I enjoy {hobby}.", f"you enjoy {hobby}"),
    ]

    num_facts = random.randint(2, 4)
    selected = random.sample(all_facts, num_facts)

    facts = [s[0] for s in selected]
    answer_parts = [s[1] for s in selected]

    question = random.choice(QUESTION_TEMPLATES["general_recall"])
    answer = "I remember that " + ", ".join(answer_parts) + "."
    return {"facts": facts, "question": question, "answer": answer}


def generate_negative_example() -> dict:
    """Example where the question is UNRELATED to stored facts.

    The model should answer normally without using the memory.
    """
    fact = random.choice([
        "I like pizza.", "I have a dog.", "I work at Google.",
        "I live in London.", "My name is Alex.",
    ])
    unrelated_qa = [
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What color is the sky?", "The sky is blue."),
        ("How many continents are there?", "There are 7 continents."),
        ("What year did World War II end?", "World War II ended in 1945."),
    ]
    q, a = random.choice(unrelated_qa)
    return {"facts": [fact], "question": q, "answer": a}


def try_download_locomo() -> list:
    """Try to download and parse LoCoMo dataset for additional training data."""
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    cache_path = Path("data/locomo10.json")

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    try:
        print("  Downloading LoCoMo dataset...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(cache_path))  # noqa: S310
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Could not download LoCoMo: {e}")
        return []


def extract_locomo_examples(locomo_data: list) -> list[dict]:
    """Extract fact-question-answer triples from LoCoMo conversations."""
    examples: list[dict] = []

    for conv in locomo_data:
        qa = conv.get("qa", [])
        if not qa:
            continue

        sessions: dict = {}
        for key, value in conv.items():
            if key.startswith("session_") and not key.endswith("date_time"):
                sessions[key] = value

        for qa_item in qa:
            question = qa_item.get("question", "")
            answer = qa_item.get("answer", "")
            evidence_ids = qa_item.get("evidence", [])
            if not question or not answer:
                continue

            facts: list[str] = []
            for _session_key, turns in sessions.items():
                if isinstance(turns, list):
                    for turn in turns:
                        if isinstance(turn, dict):
                            dia_id = turn.get("dia_id", "")
                            text = turn.get("text", "")
                            if dia_id in evidence_ids and text:
                                facts.append(text)

            if facts:
                examples.append({
                    "facts": facts[:3],
                    "question": question,
                    "answer": answer,
                    "source": "locomo",
                })
    return examples


def _retrieve_memory_vector(
    observer, query_emb: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Retrieve a memory vector from the hopfield system.

    Returns (memory_vector, energy) or (None, 0) on failure.
    """
    zero = torch.tensor(0.0)
    hopfield = observer._hopfield
    if hopfield is None:
        return None, zero

    # Clone to escape inference mode, wrap in no_grad for the router's linear layers
    query = query_emb.clone().detach()
    try:
        with torch.no_grad():
            result = hopfield(query)
        if isinstance(result, tuple) and len(result) >= 1:
            retrieved = result[0]
            if retrieved.norm().item() > 1e-6:
                energy = retrieved.norm().detach().cpu()
                return retrieved.detach().cpu(), energy
    except Exception:
        pass
    return None, zero


def generate_all_training_data(
    checkpoint_path: str,
    output_path: str,
    target_count: int = 1000,
) -> None:
    """Generate the full training dataset."""
    from memory.encoder import EmbeddingEncoder
    from memory.observer import NeuralMemoryObserver

    print(f"Generating {target_count} training examples...")

    observer = NeuralMemoryObserver()
    if os.path.isdir(checkpoint_path):
        observer._trainer.load_checkpoint(checkpoint_path)
        for name in ("hopfield", "gat"):
            if name in observer._trainer.components:
                observer._trainer.components[name].enabled = False

    encoder = EmbeddingEncoder()

    # --- Build raw examples (overshoot to compensate for retrieval failures) ---
    raw_examples: list[dict] = []
    categories = [
        "name", "age", "location", "job", "hobby",
        "pet", "food", "project", "language",
    ]
    raw_count = int(target_count * 3)  # ~45% retrieve successfully

    # Single-fact: ~60%
    for _ in range(int(raw_count * 0.6)):
        raw_examples.append(generate_single_fact_example(random.choice(categories)))

    # Multi-fact: ~25%
    for _ in range(int(raw_count * 0.25)):
        raw_examples.append(generate_multi_fact_example())

    # Negatives: ~10%
    for _ in range(int(raw_count * 0.1)):
        raw_examples.append(generate_negative_example())

    # LoCoMo bonus: up to 5%
    locomo_data = try_download_locomo()
    if locomo_data:
        locomo_examples = extract_locomo_examples(locomo_data)
        locomo_count = min(len(locomo_examples), int(raw_count * 0.05))
        if locomo_count > 0:
            raw_examples.extend(random.sample(locomo_examples, locomo_count))
        print(f"  Added {locomo_count} LoCoMo examples")

    random.shuffle(raw_examples)
    print(f"  Generated {len(raw_examples)} raw examples")

    # --- Process through brain memory system ---
    training_data: list[dict] = []

    for i, ex in enumerate(raw_examples):
        # Clear brain state
        observer.working_memory.clear()
        if hasattr(observer._hopfield, "clear"):
            observer._hopfield.clear()

        # Store facts
        for fact in ex["facts"]:
            observer.observe(fact, speaker="user")
            observer.observe("Got it.", speaker="assistant")

        # Replay
        if hasattr(observer._hopfield, "replay_recent"):
            observer._hopfield.replay_recent(
                n=len(ex["facts"]), replay_strength=0.5,
            )

        # Encode question and retrieve
        query_emb = encoder.encode(ex["question"])
        memory_vector, energy = _retrieve_memory_vector(observer, query_emb)

        if memory_vector is not None:
            training_data.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "facts": ex["facts"],
                "memory_vector": memory_vector.tolist(),
                "energy": energy.item(),
            })

        if (i + 1) % 100 == 0:
            print(
                f"  Processed {i + 1}/{len(raw_examples)} "
                f"({len(training_data)} valid)"
            )

        # Stop early once we have enough
        if len(training_data) >= target_count:
            print(f"  Reached target of {target_count} examples, stopping.")
            break

    # --- Split train / val (90/10) ---
    random.shuffle(training_data)
    split = int(len(training_data) * 0.9)
    train_data = training_data[:split]
    val_data = training_data[split:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    val_path = output_path.replace(".json", "_val.json")

    with open(output_path, "w") as f:
        json.dump(train_data, f)
    with open(val_path, "w") as f:
        json.dump(val_data, f)

    print(f"\nSaved {len(train_data)} train + {len(val_data)} val examples")
    print(f"  Train: {output_path}")
    print(f"  Val:   {val_path}")

    questions = {d["question"] for d in training_data}
    answers = {d["answer"] for d in training_data}
    print(f"  Unique questions: {len(questions)}")
    print(f"  Unique answers:   {len(answers)}")


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fast_weight_1k/final"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/memory_training_data.json"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    generate_all_training_data(checkpoint, output, target_count=count)


if __name__ == "__main__":
    main()
