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
