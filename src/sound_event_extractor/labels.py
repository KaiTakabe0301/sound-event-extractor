"""Map user labels (Japanese or English) to YAMNet/AudioSet classes."""

from __future__ import annotations

# Concept -> English keywords matched case-insensitively as substrings
# against the 521 AudioSet class display names.
_KEYWORDS: dict[str, list[str]] = {
    "dog": ["dog", "bark", "bow-wow", "yip", "howl", "growling", "whimper"],
    "cat": ["cat", "meow", "purr", "hiss", "caterwaul"],
    "bird": ["bird", "chirp, tweet", "squawk", "crow", "pigeon", "coo"],
    "siren": ["siren"],
    "baby_cry": ["baby cry", "crying, sobbing"],
    "speech": ["speech", "conversation", "narration"],
    "laughter": ["laughter", "giggle", "chuckle"],
    "car_horn": ["vehicle horn", "honking", "toot"],
    "gunshot": ["gunshot", "machine gun", "fusillade", "artillery fire"],
    "music": ["music", "musical instrument", "singing"],
    "sine": ["sine wave"],
}

# User-facing label -> concept key. Japanese aliases first, English second.
ALIASES: dict[str, str] = {
    "犬": "dog",
    "犬の鳴き声": "dog",
    "いぬ": "dog",
    "dog": "dog",
    "猫": "cat",
    "猫の鳴き声": "cat",
    "ねこ": "cat",
    "cat": "cat",
    "鳥": "bird",
    "鳥の鳴き声": "bird",
    "bird": "bird",
    "サイレン": "siren",
    "siren": "siren",
    "赤ちゃんの泣き声": "baby_cry",
    "泣き声": "baby_cry",
    "話し声": "speech",
    "会話": "speech",
    "speech": "speech",
    "笑い声": "laughter",
    "laughter": "laughter",
    "クラクション": "car_horn",
    "銃声": "gunshot",
    "音楽": "music",
    "music": "music",
    "サイン波": "sine",
    "正弦波": "sine",
}

# Suggested labels for the GUI dropdown (Japanese aliases, one per concept).
SUGGESTED_LABELS = [
    "犬の鳴き声",
    "猫の鳴き声",
    "鳥の鳴き声",
    "サイレン",
    "赤ちゃんの泣き声",
    "話し声",
    "笑い声",
    "クラクション",
    "銃声",
    "音楽",
]


def resolve_keywords(query: str) -> list[str]:
    """Resolve a user label to search keywords.

    Known aliases expand to a curated keyword set; anything else is used
    verbatim as a substring match against the AudioSet class names.
    """
    q = query.strip().lower()
    if q in ALIASES:
        return _KEYWORDS[ALIASES[q]]
    return [q] if q else []


def match_classes(class_names: list[str], query: str) -> list[int]:
    """Return indices of AudioSet classes matching the user label."""
    keywords = resolve_keywords(query)
    return [
        i
        for i, name in enumerate(class_names)
        if any(kw in name.lower() for kw in keywords)
    ]
