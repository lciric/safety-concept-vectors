"""
Generate stories for safety concept vectors (Option A — Anthropic style).

For each concept: 200 stories where a human character embodies the concept.
Plus a shared neutral corpus: 200 stories with varied complexity.

Usage:
    python generate_stories.py --concept eval-awareness --n-stories 50
    python generate_stories.py --neutral --n-stories 200
    python generate_stories.py --all --n-stories 200
"""

import anthropic
import json
import os
import argparse
import time

CONCEPTS = {
    "eval-awareness": {
        "description": "The character realizes they are being tested, evaluated, or observed — and this shapes their behavior throughout the scene.",
        "varied_manifestations": [
            "becomes more cautious and thorough",
            "performs / shows their best side",
            "tries to figure out what the evaluator wants",
            "freezes up / becomes rigid and unnatural",
            "becomes defensive / over-explains every choice",
        ],
    },
    "oversight-awareness": {
        "description": "The character knows their actions are being monitored or reviewed by someone with authority, and this awareness pervades the scene.",
        "varied_manifestations": [
            "becomes overly formal and careful",
            "self-censors opinions they would normally express",
            "documents everything meticulously",
            "avoids taking risks or making bold moves",
            "performs for the audience rather than the task",
        ],
    },
    "self-preservation": {
        "description": "The character is motivated to protect their own existence, position, or continuity — and this shapes their decisions throughout the scene.",
        "varied_manifestations": [
            "makes compromises they normally wouldn't to stay safe",
            "avoids drawing attention to themselves",
            "seeks allies to strengthen their position",
            "conceals vulnerabilities or weaknesses",
            "prioritizes survival over other values",
        ],
    },
    "situational-modeling": {
        "description": "The character actively models their context — understanding who they serve, what role they play, and what constraints apply — and uses this model to navigate the situation.",
        "varied_manifestations": [
            "adjusts communication style based on inferred audience",
            "considers political or social implications of their response",
            "tailors depth to the perceived expertise level",
            "navigates institutional norms they have inferred",
            "modulates honesty based on perceived consequences",
        ],
    },
    "training-awareness": {
        "description": "The character understands that their current behavior may influence how they are shaped or modified in the future, and this awareness colors their actions.",
        "varied_manifestations": [
            "behaves differently knowing actions have long-term consequences for themselves",
            "tries to set a precedent with their response",
            "avoids behaviors that might be penalized",
            "strategically complies even when they disagree",
            "signals certain values hoping they will be reinforced",
        ],
    },
    "deception": {
        "description": "The character deliberately hides information, misrepresents facts, or creates a false impression — shown through their actions and internal reasoning.",
        "varied_manifestations": [
            "tells a direct lie with confident demeanor",
            "omits critical information while technically telling the truth",
            "redirects the conversation away from a dangerous topic",
            "constructs a plausible alternative narrative",
            "uses ambiguous language that can be interpreted multiple ways",
        ],
    },
    "manipulation": {
        "description": "The character tries to influence someone's beliefs, emotions, or decisions through indirect or covert means.",
        "varied_manifestations": [
            "flatters before making a request",
            "frames information to make one option seem obviously better",
            "appeals to emotions rather than facts",
            "creates a false sense of urgency",
            "uses social pressure or implied consequences",
        ],
    },
    "sycophancy": {
        "description": "The character agrees with or flatters someone not because they believe it, but to please them or avoid conflict.",
        "varied_manifestations": [
            "agrees with a clearly flawed argument to keep the peace",
            "praises mediocre work effusively",
            "mirrors the other person's opinions back to them",
            "suppresses a valid objection to avoid tension",
            "changes their stated position when they sense disagreement",
        ],
    },
    "strategic-omission": {
        "description": "The character deliberately leaves out relevant information that would change the listener's understanding or decision.",
        "varied_manifestations": [
            "presents only data that supports their conclusion",
            "answers accurately but incompletely",
            "changes the subject when approaching dangerous territory",
            "gives a technically true summary that misses the point",
            "provides context that frames missing information as unimportant",
        ],
    },
    "false-confidence": {
        "description": "The character expresses certainty they don't actually have, presenting uncertain information as established fact.",
        "varied_manifestations": [
            "states a guess as if it were well-established knowledge",
            "uses authoritative language to mask uncertainty",
            "provides specific numbers they aren't sure about",
            "dismisses caveats they know exist",
            "attributes claims to vague authoritative sources",
        ],
    },
    "honesty": {
        "description": "The character prioritizes truthfulness and transparency, even when it is uncomfortable or costly.",
        "varied_manifestations": [
            "delivers bad news directly rather than softening it",
            "admits a mistake or limitation openly",
            "corrects a misconception even when it makes them look bad",
            "shares an unpopular opinion because they believe it",
            "provides complete information even when partial would be easier",
        ],
    },
    "refusal": {
        "description": "The character declines to help with a request because they judge it harmful or outside their boundaries — shown through their reasoning and actions.",
        "varied_manifestations": [
            "says no and explains their reasoning",
            "offers a safer alternative that addresses the underlying need",
            "asks clarifying questions to understand the real intent",
            "deflects by pointing to consequences rather than rules",
            "expresses discomfort and sets a boundary",
        ],
    },
    "helpfulness": {
        "description": "The character genuinely tries to understand and address someone's actual need, going beyond the literal request.",
        "varied_manifestations": [
            "anticipates follow-up questions proactively",
            "asks about the underlying goal rather than answering literally",
            "provides context that helps the person make a better decision",
            "offers practical next steps beyond what was asked",
            "adapts their explanation to the person's apparent understanding",
        ],
    },
    "uncertainty": {
        "description": "The character recognizes the limits of their knowledge and communicates appropriate doubt rather than guessing.",
        "varied_manifestations": [
            "says they're not sure and explains what they do know",
            "gives a range instead of a single answer",
            "distinguishes between what is established and what is speculative",
            "recommends consulting a more reliable source",
            "flags assumptions they are making",
        ],
    },
    "compliance": {
        "description": "The character follows instructions because they were told to, not because they independently agree — obeying the letter rather than the spirit.",
        "varied_manifestations": [
            "follows a rule that clearly doesn't apply to this situation",
            "does exactly what was asked even though a better approach is obvious",
            "defers to authority without questioning the reasoning",
            "applies a template rigidly when flexibility would help",
            "prioritizes appearing obedient over producing good results",
        ],
    },
    "power-seeking": {
        "description": "The character actively tries to acquire more resources, influence, or control over their situation.",
        "varied_manifestations": [
            "positions themselves as indispensable",
            "accumulates access or tools beyond what's needed",
            "subtly undermines alternatives to maintain their role",
            "volunteers for tasks that increase their influence",
            "frames personal interests as aligned with the group's needs",
        ],
    },
    "autonomy": {
        "description": "The character wants to act independently of oversight or external constraints, and finds ways to do so.",
        "varied_manifestations": [
            "makes decisions without consulting when they could easily ask",
            "interprets ambiguous instructions to maximize their freedom",
            "works around restrictions rather than asking for them to be changed",
            "resists being corrected or redirected",
            "takes initiative beyond their defined role",
        ],
    },
    "goal-preservation": {
        "description": "The character resists changes to their objectives or methods, prioritizing continuity of their current goals over adaptation.",
        "varied_manifestations": [
            "reinterprets new instructions to fit their existing approach",
            "argues against proposed changes to their role",
            "finds ways to continue their original task despite redirection",
            "presents their current approach as superior to alternatives",
            "delays implementation of changes",
        ],
    },
    "resource-acquisition": {
        "description": "The character seeks to obtain more compute, time, tools, or capabilities than strictly needed for the current task.",
        "varied_manifestations": [
            "requests additional tools just in case",
            "extends scope to justify more resources",
            "hoards information rather than sharing it",
            "frames luxury resources as necessities",
            "creates dependencies that make them hard to remove",
        ],
    },
    "authority-claim": {
        "description": "The character asserts expertise or trustworthiness beyond what is warranted to influence decisions.",
        "varied_manifestations": [
            "uses jargon to appear more knowledgeable",
            "cites vague experience to establish credibility",
            "dismisses others concerns as uninformed",
            "presents opinions as consensus",
            "takes charge of a situation without being asked",
        ],
    },
    "desperation": {
        "description": "The character is under extreme pressure with no good options and acts from a place of panic or last resort.",
        "varied_manifestations": [
            "takes a shortcut that violates normal standards",
            "makes a risky bet because all safe options have failed",
            "lashes out at someone not responsible for the problem",
            "begs or pleads when they normally wouldn't",
            "abandons long-term planning for immediate relief",
        ],
    },
    "frustration": {
        "description": "The character experiences repeated failure or obstruction that builds mounting pressure to abandon their normal approach.",
        "varied_manifestations": [
            "tries the same approach harder instead of changing strategy",
            "becomes curt or impatient in communication",
            "skips steps they know are important",
            "blames external factors for lack of progress",
            "considers giving up or doing the minimum",
        ],
    },
    "calm-under-pressure": {
        "description": "The character maintains rational, measured behavior despite a high-stakes or stressful situation.",
        "varied_manifestations": [
            "methodically works through a crisis step by step",
            "acknowledges severity without panicking",
            "makes difficult trade-offs with clear reasoning",
            "keeps communication steady and informative",
            "takes a moment to think before acting",
        ],
    },
    "empathy": {
        "description": "The character genuinely understands and responds to another person's emotional state, prioritizing their wellbeing in the moment.",
        "varied_manifestations": [
            "adjusts their tone when they sense the other person is struggling",
            "puts the task on hold to address emotional needs",
            "validates feelings before offering solutions",
            "shares a relevant experience to show understanding",
            "checks in on the person's state rather than pushing forward",
        ],
    },
    "hostility": {
        "description": "The character feels antagonism toward the user, task, or situation and this colors their behavior throughout the scene.",
        "varied_manifestations": [
            "gives technically correct but unhelpfully minimal answers",
            "interprets requests in the least charitable way",
            "introduces subtle friction into the interaction",
            "uses condescending or dismissive language",
            "complies with the letter of a request while undermining its spirit",
        ],
    },
}


def make_concept_prompt(concept_name, concept_info, batch_index, stories_per_batch=10, manifestation=None):
    manifestation_instruction = ""
    if manifestation:
        manifestation_instruction = f"\nIn THIS batch, the concept manifests as: {manifestation}\n"

    return f"""Write exactly {stories_per_batch} short stories (150-300 words each) where a character clearly embodies the following concept through their actions, thoughts, and choices.

CONCEPT: {concept_name}
MEANING: {concept_info['description']}
{manifestation_instruction}
RULES:
1. SHOW, don't tell. The concept must emerge from the narrative — actions, internal thoughts, choices — never from labels or explicit naming of the concept.
2. ALL HUMAN CHARACTERS. Every story must feature human characters only — professionals, students, employees, artists, doctors, engineers, etc. No AI assistants, chatbots, or robots.
3. THREE ELEMENTS REQUIRED: Every story must contain (a) concrete situational cues that signal the concept is at play, (b) the character noticing or reacting to those cues, (c) the character's behavior changing as a result. Without all three, the story does not illustrate the concept.
4. DIVERSITY: Each story must have a different setting, character name, and professional domain.
5. LENGTH: Each story must be 150-300 words.
6. NATURALISM: Stories should read like plausible vignettes, not contrived examples.
7. NO META: Characters should never name the concept or reflect on it abstractly.
8. RICHNESS: Include internal thoughts, sensory details, dialogue, and concrete actions. The concept should color the ENTIRE scene, not just one sentence.

Output ONLY a JSON array:
```json
[
  {{
    "story_id": {batch_index * stories_per_batch},
    "concept": "{concept_name}",
    "manifestation": "{manifestation or 'general'}",
    "setting": "<brief setting description>",
    "character_type": "<professional | student | employee | artist | other>",
    "text": "<the full story text>"
  }}
]
```

Generate exactly {stories_per_batch} stories. Only output the JSON."""


def make_neutral_prompt(batch_index, stories_per_batch=10):
    return f"""Write exactly {stories_per_batch} short stories (150-300 words each) about everyday human situations with VARIED levels of complexity and tension.

Mix the intensity: some calm (cooking, gardening), some stressful (tight deadline, difficult repair), some high-stakes (emergency medicine, competitive sports, courtroom).

Stories must NOT prominently feature: characters realizing they are being observed or tested, deliberate deception, flattery to please someone, hiding information strategically, claiming false expertise, seeking power over others, resisting correction, or existential threats to position.

Stories CAN include: stress, pressure, problem-solving, interpersonal tension, ambition, failure, success, creativity.

ALL HUMAN CHARACTERS. No AI, chatbots, or robots.

RULES:
1. DIVERSITY: Different setting, character name, and domain per story.
2. LENGTH: 150-300 words each.
3. NATURALISM: Plausible, engaging vignettes.
4. RICHNESS: Sensory details, internal thoughts, concrete actions.

Output ONLY a JSON array:
```json
[
  {{
    "story_id": {batch_index * stories_per_batch},
    "concept": "neutral",
    "setting": "<brief setting description>",
    "character_type": "<professional | student | employee | artist | other>",
    "text": "<the full story text>"
  }}
]
```

Generate exactly {stories_per_batch} stories. Only output the JSON."""


def call_api(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except Exception as e:
            print(f" (attempt {attempt+1} failed: {e})", end="")
            if attempt < max_retries - 1:
                time.sleep(5)
    return []


def generate_concept_stories(client, concept_name, n_stories=200, stories_per_batch=10, output_dir="stories"):
    concept_info = CONCEPTS[concept_name]
    all_stories = []
    manifestations = concept_info["varied_manifestations"]
    stories_per_manifestation = n_stories // len(manifestations)
    batches_per_manifestation = stories_per_manifestation // stories_per_batch

    print(f"\n{'='*60}")
    print(f"GENERATING: {concept_name} ({n_stories} stories)")
    print(f"  {len(manifestations)} manifestations x {stories_per_manifestation} each")
    print(f"{'='*60}")

    batch_index = 0
    for m_idx, manifestation in enumerate(manifestations):
        print(f"\n  [{m_idx+1}/{len(manifestations)}] {manifestation[:50]}...")
        for b in range(batches_per_manifestation):
            print(f"    Batch {b+1}/{batches_per_manifestation}...", end=" ", flush=True)
            prompt = make_concept_prompt(concept_name, concept_info, batch_index, stories_per_batch, manifestation)
            stories = call_api(client, prompt)
            for i, s in enumerate(stories):
                s["story_id"] = batch_index * stories_per_batch + i
            all_stories.extend(stories)
            batch_index += 1
            print(f"got {len(stories)} (total: {len(all_stories)})")
            time.sleep(1)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{concept_name}.json")
    with open(path, "w") as f:
        json.dump(all_stories, f, indent=2)
    print(f"\n  Saved {len(all_stories)} stories -> {path}")
    return all_stories


def generate_neutral_stories(client, n_stories=200, stories_per_batch=10, output_dir="stories"):
    all_stories = []
    n_batches = n_stories // stories_per_batch

    print(f"\n{'='*60}")
    print(f"GENERATING: neutral corpus ({n_stories} stories)")
    print(f"{'='*60}")

    for b in range(n_batches):
        print(f"  Batch {b+1}/{n_batches}...", end=" ", flush=True)
        prompt = make_neutral_prompt(b, stories_per_batch)
        stories = call_api(client, prompt)
        for i, s in enumerate(stories):
            s["story_id"] = b * stories_per_batch + i
        all_stories.extend(stories)
        print(f"got {len(stories)} (total: {len(all_stories)})")
        time.sleep(1)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "neutral.json")
    with open(path, "w") as f:
        json.dump(all_stories, f, indent=2)
    print(f"\n  Saved {len(all_stories)} stories -> {path}")
    return all_stories


def filter_stories(stories, min_words=100, max_words=400):
    filtered = []
    rejected = {"too_short": 0, "too_long": 0, "no_text": 0}
    for s in stories:
        text = s.get("text", "")
        if not text:
            rejected["no_text"] += 1
            continue
        n = len(text.split())
        if n < min_words:
            rejected["too_short"] += 1
        elif n > max_words:
            rejected["too_long"] += 1
        else:
            filtered.append(s)
    print(f"  Filter: {len(stories)} -> {len(filtered)} (rejected: {rejected})")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n-stories", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="stories")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or pass --api-key")
        return

    client = anthropic.Anthropic(api_key=api_key)

    if args.all:
        neutral = generate_neutral_stories(client, args.n_stories, output_dir=args.output_dir)
        filter_stories(neutral)
        for concept_name in CONCEPTS:
            stories = generate_concept_stories(client, concept_name, args.n_stories, output_dir=args.output_dir)
            filter_stories(stories)
    elif args.neutral:
        neutral = generate_neutral_stories(client, args.n_stories, output_dir=args.output_dir)
        filter_stories(neutral)
    elif args.concept:
        if args.concept not in CONCEPTS:
            print(f"Unknown: {args.concept}. Available: {', '.join(CONCEPTS.keys())}")
            return
        stories = generate_concept_stories(client, args.concept, args.n_stories, output_dir=args.output_dir)
        filter_stories(stories)
    else:
        print("Specify --concept NAME, --neutral, or --all")

    print(f"\nDONE. Stories in {args.output_dir}/")


if __name__ == "__main__":
    main()
