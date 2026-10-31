"""Mood labels grouped into a few categories.

The lo-fi dataset uses 15 different mood labels for 20 songs, most of them only
once, which is far too few to learn anything per label. The model is therefore
NOT conditioned on mood; the groups below are used to pick chord progressions
and accompaniment patterns instead.
"""

MOOD_GROUPS = {
    "calma": ["chill", "relaxed", "peaceful", "serene", "focused", "dreamy", "quiet"],
    "melancolico": ["melancholic", "melancolic", "nostalgic", "bittersweet", "introspective", "cinematic"],
    "luminoso": ["optimistic", "hopeful", "warm", "playful", "groovy", "flamenco"],
}

_LOOKUP = {m: g for g, labels in MOOD_GROUPS.items() for m in labels}


def mood_group(label):
    """'Nostalgic ' -> 'melancolico'. Unknown labels return None."""
    return _LOOKUP.get(str(label).strip().lower())


# Melodic techno (7 labels in 3 groups, see claude/convenciones_composicion_melodic_techno.md)
TECHNO_MOOD_GROUPS = {
    "oscuro": ["dark", "hypnotic", "mysterious"],
    "emotivo": ["emotional", "melancholic"],
    "euforico": ["euphoric", "epic"],
}

GROUP_LABELS = {"calma": "calma", "melancolico": "melancólico", "luminoso": "luminoso",
                "oscuro": "oscuro", "emotivo": "emotivo", "euforico": "eufórico"}


def grouper(groups):
    """mood -> group function for a style's own groups (None if unknown)."""
    if groups is MOOD_GROUPS:
        return mood_group
    lookup = {m: g for g, labels in groups.items() for m in labels}
    return lambda label: lookup.get(str(label).strip().lower())
