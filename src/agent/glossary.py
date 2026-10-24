"""Small structured glossary for AI risk-management terms.

The entries are short local definitions used by the lookup tool. They are not
a substitute for the source publications.
"""

from __future__ import annotations

GLOSSARY: dict[str, dict[str, str]] = {
    "govern": {
        "term": "Govern",
        "definition": (
            "Govern cultivates a risk-management culture, assigns roles, and sets policies for AI systems."
        ),
    },
    "map": {
        "term": "Map",
        "definition": "Map identifies the context, intended use, and risks of an AI system.",
    },
    "measure": {
        "term": "Measure",
        "definition": (
            "Measure analyzes, benchmarks, and monitors AI risks using quantitative and qualitative methods."
        ),
    },
    "manage": {
        "term": "Manage",
        "definition": "Manage prioritizes and acts on identified AI risks, including response and recovery.",
    },
    "profile": {
        "term": "Profile",
        "definition": "A profile tailors the AI Risk Management Framework functions to a specific use case or sector.",
    },
}
