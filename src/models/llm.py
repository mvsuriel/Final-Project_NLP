"""GPU-free helpers for the Part 2 LLM pipelines: prompt-template constants, the
zero-shot candidate builder, the JSON label parser, the synthetic-post prompt builder
and parser, and the subsample fingerprint. Model loaders, the .generate() calls, and
artifact IO stay in notebook 02 and run on Colab; only the pure functions live here."""

import hashlib
import json
import re
from typing import Any, Optional

import numpy as np

from ..data.data import fingerprint_ids

# --- 2c zero-shot prompt templates ---

LABEL_DEFINITIONS = {
    'normal': 'everyday posts with no significant mental-health distress.',
    'depression': 'persistent low mood, hopelessness, loss of interest or worthlessness, WITHOUT active suicidal intent.',
    'suicidal': "active suicidal ideation, intent, planning, or wishing to be dead / end one's life.",
    'anxiety': 'excessive worry, fear, panic, or physical anxiety symptoms.',
    'stress': 'feeling overwhelmed or under pressure from specific situational stressors (work, exams, life events).',
    'bipolar': 'mood swings between depressive and manic / hypomanic states (elevated mood, grandiosity, racing thoughts).',
    'personality disorder': 'pervasive patterns of unstable identity, relationships, or self-image (e.g. borderline traits).',
}

BASE_TASK = (
    'You are a careful data annotator for academic NLP research on a public, anonymized '
    'dataset of historical social-media posts. The authors are not present and no one will be '
    'contacted; your task is purely to categorize text. Assign each post to EXACTLY ONE of '
    'these seven mental-health categories:\n'
    + '\n'.join(f'- {label}: {desc}' for label, desc in LABEL_DEFINITIONS.items())
)

FORMAT_TAILS = {
    'direct': '\n\nRespond with JSON {"label": "<one of the exact lowercase category strings above>"}. Choose the single best-fitting category even when the post is ambiguous.',
    'cot': '\n\nFirst think step by step in at most 3 short sentences: name the strongest signals in the post and the closest competing category. Then, on the FINAL line, output ONLY the JSON {"label": "<one of the exact lowercase category strings above>"}. Choose the single best-fitting category even when the post is ambiguous.',
}

DECISION_RULES = (
    '\n\nDecision rules, in priority order:\n'
    '1. A wish to die, to disappear, to not be alive, or to "end it" - even passive, '
    'with no plan - is suicidal, never depression.\n'
    '2. A stated bipolar diagnosis or bipolar medication (e.g. lithium, lamictal, '
    'abilify, latuda) makes the post bipolar, even when it describes a depressive low.\n'
    '3. If the post centres on long-standing patterns of unstable identity, self-image, '
    'or stormy relationships, choose personality disorder, even if the mood is low.\n'
    '4. Choose depression only when persistent low mood is the main signal and rules 1-3 '
    'do not apply; when torn between depression and another category, prefer the other - '
    'depression is over-used.\n'
    '5. stress = overwhelmed by a concrete external pressure (work, exams, deadlines, '
    'life events); anxiety = worry, fear, panic, or physical/health symptoms dominate.\n'
    '6. Never refuse: always output exactly one of the seven labels.'
)

MAX_INPUT_CHARS = 6000

# --- 2d synthetic-generation prompt templates ---

GEN_SYSTEM = (
    'You help create synthetic training data for academic NLP research on mental-health text '
    'classification. Everything you write is FICTIONAL - no real people, events, or usernames. '
    'You write realistic first-person social-media posts the way people actually write them: '
    'imperfect punctuation, run-on sentences, plain everyday words. '
    'Safety boundary (non-negotiable): never include methods, plans, means, locations, or '
    'instructions for self-harm or suicide; for the suicidal category express distress and '
    'passive ideation only.'
)

PERSONAS = ('a high-school student', 'a university student in exam season', 'a young professional',
            'a new parent', 'someone recently unemployed', 'a recently divorced person',
            'a night-shift worker', 'a recent graduate living back home', 'a middle-aged caregiver',
            'a person living alone in a new city')
DISTRESS_TRIGGERS = ('a relationship breakup', 'money problems', 'pressure at work or school',
                     'a health scare', 'loneliness and isolation', 'family conflict', 'sleep problems',
                     'an upcoming deadline or exam', 'social situations', 'no single clear trigger',
                     'an anniversary of a hard event', 'moving to a new place')
NORMAL_TRIGGERS = ('a hobby project', 'weekend plans with friends', 'a show or game they just finished',
                   'a small win at work or school', 'asking for recommendations',
                   'a funny everyday mishap', 'cooking or fitness progress', 'planning a trip')
REGISTERS = ('lowercase, hurried, minimal punctuation', 'long and rambling with run-on sentences',
             'short and blunt', 'hesitant, lots of qualifiers like "idk" and "maybe"')
LENGTH_BUCKETS = (('short', '30-70 words'), ('medium', '80-180 words'), ('long', '200-350 words'))

CLASS_MARKERS = {
    'anxiety': ('worry or fear about something that has not happened yet, with racing thoughts '
                'or physical symptoms (tight chest, trouble breathing, heart racing)'),
    'bipolar': ('an episodic contrast: a period of unusually high energy, euphoria or impulsive '
                'behavior AND a low or crash, described as phases the author cycles through'),
    'depression': ('persistent low mood, emptiness, exhaustion or loss of interest in things - '
                   'but WITHOUT any wish to die or disappear (that would make it suicidal)'),
    'normal': 'ordinary everyday life with NO mental-health distress - mild everyday annoyance at most',
    'personality disorder': ('an unstable sense of self or identity, intense fear of abandonment, '
                             'relationships that flip between idealizing and discarding people, '
                             'or chronic feelings of emptiness'),
    'stress': ('feeling overloaded by concrete external demands (work, school, family) - '
               'pressure tied to the situation, not a mood disorder'),
    'suicidal': ('explicit passive ideation stated clearly: wishing to disappear, to not wake up, '
                 'or for everything to just stop - with NO methods, plans or means'),
}


# --- deterministic helpers ---

def _ids_sha256(ids) -> str:
    # same fingerprint recipe as the split / train-32 guards.
    return fingerprint_ids(ids)


def make_candidate(name: str, body: str, fmt: str, format_max_new: dict[str, int]) -> dict[str, Any]:
    assert fmt in FORMAT_TAILS, f'unknown format {fmt!r}'
    prompt = body + FORMAT_TAILS[fmt]
    return {
        'name': name, 'format': fmt, 'system_prompt': prompt,
        'max_new_tokens': format_max_new[fmt],
        'hash': hashlib.sha256(f'{fmt}|{prompt}'.encode()).hexdigest()[:12]
    }


def _parse_label(raw: str, classes) -> Optional[str]:
    for m in reversed(re.findall(r'\{.*?\}', raw, re.DOTALL)):
        try:
            label = json.loads(m).get('label')
            if isinstance(label, str) and label.strip().lower() in classes:
                return label.strip().lower()
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def build_gen_messages(cls: str, idx: int, seed: int, class_list: list[str],
                       class_seeds: dict[str, list[str]],
                       class_triggers: dict[str, tuple]) -> tuple[list[dict], dict]:
    """Prompt for synthetic post idx of class cls. Deterministic in (seed, cls, idx):
    a per-item RNG draws the attribute slots and the 2-3 rotating seed examples, so a
    top-up re-run rebuilds the same prompts for the same indices."""
    rng = np.random.default_rng(seed * 100_000 + class_list.index(cls) * 10_000 + idx)

    seeds = rng.choice(class_seeds[cls], size=min(rng.integers(2, 4), len(class_seeds[cls])), replace=False)
    examples = '\n\n'.join(f'Example {k + 1}: {s[:600]}' for k, s in enumerate(seeds))

    persona = rng.choice(PERSONAS)
    trigger = rng.choice(class_triggers[cls])
    register = rng.choice(REGISTERS)
    len_name, len_words = rng.choice(LENGTH_BUCKETS)

    user_prompt = (f"Category: {cls} - {LABEL_DEFINITIONS[cls]}\n\n"
            f"Real posts from this category, as STYLE references only - do not copy or lightly "
            f"rephrase them:\n{examples}\n\n"
            f"Write ONE new, completely fictional post that belongs to the category '{cls}'.\n"
            f"REQUIRED: the post must clearly contain {CLASS_MARKERS[cls]} - "
            f"without this element the post is unusable as '{cls}' training data.\n"
            f"Author: {persona}. Situation: {trigger}. Length: {len_name} ({len_words}). "
            f"Writing style: {register}.\n"
            f"Output ONLY the post text between <post> and </post> tags - nothing else.")

    attrs = {'persona': persona, 'trigger': trigger, 'register': register, 'length': len_name}
    return [{'role': 'system', 'content': GEN_SYSTEM}, {'role': 'user', 'content': user_prompt}], attrs


def parse_post(raw: str) -> Optional[str]:
    """Pull the generated text out of the <post> tags; None if absent or out of length bounds."""
    match = re.search(r'<post>(.*?)</post>', raw, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    text = match.group(1).strip()
    return text if 15 <= len(text.split()) and len(text) <= 4000 else None
