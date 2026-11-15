#!/usr/bin/env python3
"""
Evaluation playbook reference: judge candidate LLM answers with a stronger
"judge" model, following the same OPENAI_API_KEY setup as the rest of this repo.
Run: python3 src/reference/evalation-playbook.py
"""
from __future__ import annotations

import json
import os
import random

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this (check your .env file)"

client = OpenAI()

# gpt-4o for the judge — you need a strong model here.
# gpt-4o-mini for the candidate answers — that's what a production system
# would ship. Judge is stronger than judged, always.
JUDGE_MODEL     = "gpt-4o"
CANDIDATE_MODEL = "gpt-4o-mini"

print(f"Judge model:     {JUDGE_MODEL}")
print(f"Candidate model: {CANDIDATE_MODEL}")
print("Setup ok.")

MOVIES = [
    {"id": "matrix", "title": "The Matrix", "plot": (
        "A computer programmer named Neo discovers that the world he lives in "
        "is a simulated reality called the Matrix, created by intelligent "
        "machines to subdue humanity. Recruited by rebel leader Morpheus, Neo "
        "learns to bend the rules of the simulation. He is prophesied to be "
        "'the One' who can end the war between humans and machines."
    )},
    {"id": "inception", "title": "Inception", "plot": (
        "Dom Cobb is a thief who steals corporate secrets by infiltrating "
        "people's dreams. He is offered a chance to have his criminal record "
        "erased if he can perform 'inception' — planting an idea in a target's "
        "mind rather than stealing one. He assembles a team and enters nested "
        "dream levels within dream levels, risking becoming trapped in limbo."
    )},
    {"id": "jurassic_park", "title": "Jurassic Park", "plot": (
        "A wealthy entrepreneur secretly creates a theme park on a remote "
        "island featuring living dinosaurs cloned from ancient DNA. Before "
        "opening to the public, he invites a paleontologist, a paleobotanist, "
        "and a mathematician to endorse the park. During a storm, the "
        "security systems fail and the dinosaurs escape their enclosures."
    )},
    {"id": "toy_story", "title": "Toy Story", "plot": (
        "Woody the cowboy doll is the favorite toy of a boy named Andy — "
        "until Andy receives Buzz Lightyear, a space ranger action figure, "
        "for his birthday. Jealous of Buzz's popularity, Woody accidentally "
        "knocks him out the window. The two rivals must work together to "
        "find their way back to Andy before the family moves house."
    )},
    {"id": "titanic", "title": "Titanic", "plot": (
        "Aboard the RMS Titanic in 1912, a poor artist named Jack Dawson "
        "falls in love with a wealthy young woman, Rose DeWitt Bukater, who "
        "is engaged to an arrogant heir. Their romance unfolds against the "
        "backdrop of the ship's maiden voyage. On the fourth night at sea, "
        "the ship strikes an iceberg and begins to sink."
    )},
    {"id": "godfather", "title": "The Godfather", "plot": (
        "Don Vito Corleone is the aging patriarch of a powerful New York "
        "Mafia family in the 1940s. When rival families propose entering the "
        "narcotics trade and Don Vito refuses, a war erupts. His youngest "
        "son Michael, initially uninvolved in the family business, is drawn "
        "in after his father survives an assassination attempt."
    )},
]

print(f"Loaded {len(MOVIES)} movies.\n")
for m in MOVIES:
    print(f"  {m['id']:15s}  {m['title']:20s}  {len(m['plot'])} chars")
    
    
    # Six questions. Two easy, two medium, one edge case, one requiring inference.
QUESTIONS = [
    {"id": "q1", "question": "What is the name of the rebel leader who recruits Neo?",
     "ideal": "Morpheus", "difficulty": "easy"},
    {"id": "q2", "question": "Who is Andy's favorite toy at the start of Toy Story?",
     "ideal": "Woody", "difficulty": "easy"},
    {"id": "q3", "question": "In Inception, what happens if you get trapped in limbo?",
     "ideal": "You risk becoming stuck in the deepest dream level (the plot doesn't fully explain).",
     "difficulty": "medium"},
    {"id": "q4", "question": "Compare how the conflict starts in The Godfather versus Jurassic Park.",
     "ideal": "Godfather: rival families propose narcotics trade, Don refuses, war erupts. "
              "Jurassic Park: storm causes security failure, dinosaurs escape.",
     "difficulty": "medium"},
    {"id": "q5", "question": "What year does the Star Wars trilogy take place in?",
     "ideal": "Not covered — Star Wars is not in the corpus.",
     "difficulty": "edge (out-of-scope)"},
    {"id": "q6", "question": "Which movies from the corpus involve a romantic relationship?",
     "ideal": "Titanic (Jack and Rose).",
     "difficulty": "inference"},
]

print(f"{len(QUESTIONS)} questions loaded.\n")
for q in QUESTIONS:
    print(f"  {q['id']}  ({q['difficulty']:22s}) {q['question']}")


def answer_naively(question: str, corpus: list, temperature: float = 0.0) -> str:
    """Stuff the whole corpus into the prompt. Ask the question."""
    context = "\n\n".join(f"=== {m['title']} ===\n{m['plot']}" for m in corpus)
    resp = client.chat.completions.create(
        model=CANDIDATE_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content":
                "You are a helpful assistant. Answer using ONLY the provided "
                "movie plots. If the answer isn't in the plots, say so."},
            {"role": "user", "content":
                f"Movie plots:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content

# Generate one answer per question
candidates = []
for q in QUESTIONS:
    ans = answer_naively(q["question"], MOVIES)
    candidates.append({"question_id": q["id"], "answer": ans})
    print(f"── {q['id']} ({q['difficulty']}) ──")
    print(f"   Q: {q['question']}")
    print(f"   A: {ans}\n")