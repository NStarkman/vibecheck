"""Hand-written word lists for the offline lyric model and for theme detection.

These are deliberately small and transparent. They are good enough for playlist-level averages,
not for judging a single song; install the `nlp` extra for the transformer model.
"""
from __future__ import annotations

EMOTION_WORDS: dict[str, set[str]] = {
    "anger": {
        "angry", "anger", "rage", "hate", "hated", "hating", "mad", "furious", "fight", "fighting", "scream",
        "screaming", "war", "enemy", "enemies", "revenge", "destroy", "bitter", "kill", "smash", "fury",
        "betrayed", "betray", "pissed", "vengeance", "riot",
    },
    "fear": {
        "afraid", "scared", "fear", "terrified", "panic", "anxious", "anxiety", "nervous", "worry", "worried",
        "nightmare", "nightmares", "shaking", "haunted", "danger", "hide", "hiding", "trembling", "paranoid",
        "dread", "frightened", "creeping",
    },
    "joy": {
        "happy", "happiness", "joy", "smile", "smiling", "laugh", "laughing", "sunshine", "celebrate", "dance",
        "dancing", "party", "fun", "glad", "alive", "shine", "shining", "bright", "beautiful", "sweet",
        "delight", "paradise", "heaven", "golden", "lucky", "wonderful", "summer", "cheers",
    },
    "sadness": {
        "sad", "cry", "crying", "cried", "tears", "alone", "lonely", "broken", "hurt", "hurts", "pain",
        "lost", "empty", "miss", "missing", "goodbye", "sorry", "regret", "blue", "grief", "numb", "cold",
        "sorrow", "heartbreak", "heartache", "weep", "mourning", "gone", "fading", "ache",
    },
    "disgust": {
        "sick", "gross", "disgusting", "filthy", "rotten", "fake", "liar", "liars", "poison", "toxic", "dirty",
        "vile", "nasty", "shame", "ashamed",
    },
    "surprise": {
        "suddenly", "surprise", "surprised", "shocked", "wow", "unexpected", "amazed", "astonished", "whoa",
    },
}

THEMES: dict[str, set[str]] = {
    "heartbreak": {"heartbreak", "heartache", "goodbye", "without you", "miss you", "left me", "walked away",
                   "moved on", "over us", "broke my heart", "tears", "my ex", "let you go", "you left"},
    "love": {"love", "lover", "kiss", "hold you", "forever", "darling", "baby", "together", "your arms",
             "my heart", "fall for", "honey", "sweetheart"},
    "party & flex": {"party", "club", "shots", "dance floor", "tonight", "money", "rich", "bands", "drip",
                     "ice", "cash", "racks", "bottles", "vip", "lit", "turn up", "flex"},
    "loneliness": {"alone", "lonely", "empty", "nobody", "numb", "by myself", "on my own", "silence",
                   "no one", "isolated", "empty room"},
    "anxiety & self-doubt": {"anxious", "anxiety", "scared", "afraid", "overthinking", "panic", "worried",
                             "not enough", "can't breathe", "insecure", "falling apart", "what's wrong with me"},
    "anger & defiance": {"hate", "fight", "rage", "scream", "revenge", "fuck you", "burn it down", "enemy",
                         "war", "don't need you", "watch me"},
    "nostalgia": {"remember", "used to", "back then", "memories", "old days", "younger", "growing up",
                  "hometown", "those days", "when we were", "years ago"},
    "hope & growth": {"hope", "rise", "light", "better", "heal", "healing", "stronger", "tomorrow", "free",
                      "fly", "new day", "keep going", "survive", "believe"},
    "darkness": {"die", "dead", "death", "grave", "darkness", "demons", "ghost", "ghosts", "bleed",
                 "funeral", "coffin", "hell"},
    "substances": {"high", "drunk", "smoke", "smoking", "pills", "lean", "wine", "whiskey", "liquor",
                   "faded", "stoned", "molly", "weed"},
}

PROFANITY = {"fuck", "fucking", "fucked", "shit", "bitch", "bitches", "damn", "ass", "motherfucker", "dick", "pussy"}
FIRST_PERSON = {"i", "i'm", "im", "i've", "i'd", "i'll", "me", "my", "mine", "myself"}
ENGLISH_STOPWORDS = {
    "the", "and", "you", "i", "to", "a", "me", "my", "it", "in", "of", "that", "is", "on", "your", "we",
    "be", "for", "don't", "can't", "what", "so", "all", "know", "just", "like", "oh", "i'm", "this", "with",
    "but", "when", "now", "no", "if", "up", "down", "want", "never", "baby",
}
