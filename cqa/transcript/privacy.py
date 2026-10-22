"""Pseudonymisation: tutor -> TUTOR, students -> S1..Sn, also inside the text.

The returned mapping (real display name -> pseudonym) must never leave the process or local cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from cqa.schemas import Transcript

FUZZY_MIN = 85
DEVICE_WORDS = {
    "iphone", "ipad", "galaxy", "android", "phone", "mobile", "laptop", "pc", "desktop",
    "macbook", "mac", "tablet", "tab", "redmi", "samsung", "oppo", "vivo", "realme", "pixel",
}  # fmt: skip
STOPWORDS = DEVICE_WORDS | {
    # chess
    "chess", "king", "queen", "knight", "bishop", "rook", "pawn", "pawns", "castle", "castling",
    "check", "checkmate", "mate", "stalemate", "board", "white", "black", "move", "game", "fork", "pin",
    # class talk
    "okay", "yes", "yeah", "good", "great", "very", "well", "nice", "sir", "maam", "mam", "madam",
    "teacher", "coach", "class", "hello", "thank", "thanks", "please", "sorry", "today", "homework",
    "student", "guest", "user", "host", "zoom", "academy", "team", "the", "and", "you", "what", "who",
    "where", "when", "why", "how", "this", "that", "can", "did", "now", "let",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "english", "hindi",
}  # fmt: skip
_CAP_TOKEN = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_WORD_TOKEN = re.compile(r"\b[A-Za-z]{3,}\b")


def _base_name(display: str) -> str:
    """Strip device suffixes: "Aarav's iPad" -> "Aarav", "Priya iPhone 12" -> "Priya"."""
    toks = display.replace("'s ", " ").split()
    for i, tok in enumerate(toks):
        if tok.lower().removesuffix("'s") in DEVICE_WORDS:
            toks = toks[:i]
            break
    return " ".join(toks)


def _name_tokens(display: str) -> list[str]:
    base = _base_name(display)
    toks = [t for t in re.split(r"[^A-Za-z]+", base) if len(t) >= 3 and t.lower() not in STOPWORDS]
    return toks


def _assign_pseudonyms(transcript: Transcript) -> dict[str, str]:
    mapping: dict[str, str] = {}
    n = 0
    for t in transcript.turns:
        if t.speaker in mapping or t.role == "unknown":
            continue
        if t.role == "tutor":
            mapping[t.speaker] = "TUTOR"
        else:
            n += 1
            mapping[t.speaker] = f"S{n}"
    return mapping


def _replace_names(text: str, patterns: list[tuple[re.Pattern[str], str]], fuzzy: dict[str, str]) -> str:
    for pat, pseudo in patterns:
        text = pat.sub(pseudo, text)

    def fuzzy_sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        # Lowercase name variants ("arav") leak too (E11); lowercase tokens are mostly ordinary words, so they
        # get the wider allowlist.
        if tok.lower() in (STOPWORDS if tok[0].isupper() else ALLOWED_CAPS):
            return tok
        best, best_score = None, 0.0
        for name, pseudo in fuzzy.items():
            score = fuzz.ratio(tok.lower(), name)
            if score > best_score:
                best, best_score = pseudo, score
        return best if best and best_score >= FUZZY_MIN else tok

    return _WORD_TOKEN.sub(fuzzy_sub, text)


def pseudonymise(transcript: Transcript) -> tuple[Transcript, dict[str, str]]:
    """Return (pseudonymised transcript, mapping). Expects roles to be assigned."""
    mapping = _assign_pseudonyms(transcript)

    # Exact patterns: full base names first (longest first), then single name tokens.
    exact: dict[str, str] = {}
    fuzzy: dict[str, str] = {}
    for display, pseudo in mapping.items():
        base = _base_name(display)
        if base:
            exact[base] = pseudo
        for tok in _name_tokens(display):
            exact.setdefault(tok, pseudo)
            fuzzy.setdefault(tok.lower(), pseudo)
    patterns = [
        (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), pseudo)
        for name, pseudo in sorted(exact.items(), key=lambda kv: -len(kv[0]))
    ]

    turns = [
        t.model_copy(
            update={
                "speaker": mapping.get(t.speaker, t.speaker),
                "text": _replace_names(t.text, patterns, fuzzy),
            }
        )
        for t in transcript.turns
    ]
    return transcript.model_copy(update={"turns": turns}), mapping


# Whisper transcripts have no speaker names to pseudonymise from, so names said aloud are scrubbed
# heuristically: capitalised words mid-sentence, and capitalised words used as a vocative
# ("Arjun, what is ...?"). Best effort only; Phase 2 should improve this.
CHESS_TERMS = {
    "king", "kings", "queen", "queens", "rook", "rooks", "bishop", "bishops", "knight", "knights", "pawn",
    "pawns", "piece", "pieces", "check", "checks", "checkmate", "mate", "stalemate", "castle", "castling",
    "draw", "fork", "pin", "skewer", "gambit", "opening", "middlegame", "endgame", "en", "passant",
    "promotion", "rank", "file", "diagonal", "board", "square", "chess", "lichess", "chesscom", "tactic",
    "tactics", "puzzle", "puzzles", "sicilian", "french", "italian", "spanish", "english", "ruy", "lopez",
    "caro", "kann", "london", "scholar", "fool", "defence", "defense", "attack", "variation", "white", "black",
}  # fmt: skip
SENTENCE_WORDS = {
    "i", "i'm", "i'll", "i've", "i'd", "ok", "okay", "oh", "so", "and", "but", "or", "if", "then", "there",
    "here", "it", "is", "are", "was", "were", "be", "we", "they", "he", "she", "you", "my", "your", "our",
    "his", "her", "their", "no", "not", "yes", "yeah", "all", "one", "two", "three", "four", "five", "first",
    "second", "next", "last", "right", "left", "just", "also", "now", "today", "tomorrow", "yesterday",
    "everyone", "everybody", "anyone", "someone", "what", "why", "how", "when", "where", "who", "which",
    "this", "that", "these", "those", "can", "could", "would", "should", "will", "do", "did", "does",
    "have", "has", "had", "let", "let's", "look", "see", "try", "think", "remember", "move", "play", "take",
    "very", "good", "great", "nice", "well", "excellent", "perfect", "correct", "wrong", "wow", "hello",
    "hi", "hey", "bye", "please", "thank", "thanks", "sorry", "welcome", "after", "before", "because",
    "again", "any", "some", "only", "maybe", "sure", "fine", "with", "for", "from", "in", "on", "at", "to",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "india", "indian", "google", "youtube", "zoom",
}  # fmt: skip
ALLOWED_CAPS = STOPWORDS | CHESS_TERMS | SENTENCE_WORDS
_PSEUDONYM = re.compile(r"^(?:TUTOR|NAME|S\d+)$")
_SENTENCE_START = re.compile(r"(?:^|[.?!]\s+)$")
NAME_TOKEN = "[NAME]"


def scrub_names(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok.lower() in ALLOWED_CAPS or _PSEUDONYM.match(tok):
            return tok
        at_start = bool(_SENTENCE_START.search(text[: m.start()]))
        vocative = text[m.end() : m.end() + 1] in {",", "?", "!"}
        return NAME_TOKEN if (not at_start or vocative) else tok

    return _CAP_TOKEN.sub(sub, text)


def scrub_transcript(transcript: Transcript) -> Transcript:
    turns = [t.model_copy(update={"text": scrub_names(t.text)}) for t in transcript.turns]
    return transcript.model_copy(update={"turns": turns})


def anonymise(transcript: Transcript) -> tuple[Transcript, dict[str, str]]:
    """The only transcript text that may leave the process. Expects roles to be assigned.

    VTT: display names -> pseudonyms, then the [NAME] scrubber for names said aloud that are nobody's display
    name (a parent's account, "iPhone (Arjun)", absent children). Whisper: scrubber only.
    """
    mapping: dict[str, str] = {}
    if transcript.source == "vtt":
        transcript, mapping = pseudonymise(transcript)
    return scrub_transcript(transcript), mapping


# Leak audit (scripts/leak_audit.py). Counts only; matched tokens never leave the local machine.
AUDIT_FUZZY_MIN = 70
_WORD = re.compile(r"[A-Za-z]+")


def display_name_tokens(names: list[str]) -> set[str]:
    """Every alphabetic token (>= 3 chars) of the full display names, minus device words and stopwords.
    Deliberately independent of _base_name, which drops everything after a device word."""
    return {
        tok.lower() for name in names for tok in _WORD.findall(name) if len(tok) >= 3 and tok.lower() not in STOPWORDS
    }


@dataclass
class LeakCounts:
    caps: int  # (a) capitalised tokens not in the allowlist, pseudonyms or prompt template
    caps_distinct: int
    name_hits: int  # (b) tokens fuzzy-matching a display-name token at >= AUDIT_FUZZY_MIN
    names_hit: int  # distinct display-name tokens matched
    hits: list[tuple[str, str, float]]  # (token, name token, score); local dump only


def leak_counts(text: str, template: str, display_names: list[str]) -> LeakCounts:
    """Audit one judge input. Tokens that also occur in the fixed prompt template are ignored."""
    skip = {w.lower() for w in _WORD.findall(template)} | ALLOWED_CAPS | {"tutor", "name"}
    caps = [m.group(0) for m in _CAP_TOKEN.finditer(text) if m.group(0).lower() not in skip]
    names = display_name_tokens(display_names)
    counts: dict[str, int] = {}
    for w in _WORD.findall(text):
        w = w.lower()
        if len(w) >= 3 and w not in skip:
            counts[w] = counts.get(w, 0) + 1
    hits: list[tuple[str, str, float]] = []
    for w in counts:
        best = max(((fuzz.ratio(w, n), n) for n in names), default=(0.0, ""))
        if best[0] >= AUDIT_FUZZY_MIN:
            hits.append((w, best[1], round(best[0], 1)))
    return LeakCounts(
        caps=len(caps),
        caps_distinct=len({c.lower() for c in caps}),
        name_hits=sum(counts[w] for w, _, _ in hits),
        names_hit=len({n for _, n, _ in hits}),
        hits=hits,
    )


def scrub_stats(transcript: Transcript) -> tuple[int, float]:
    """(c) Over-scrubbing: [NAME] replacements, and the share of tutor word tokens that are [NAME]."""
    total = sum(t.text.count(NAME_TOKEN) for t in transcript.turns)
    tutor = [t.text for t in transcript.turns if t.role == "tutor"]
    words = sum(len(x.split()) for x in tutor)
    scrubbed = sum(x.count(NAME_TOKEN) for x in tutor)
    return total, scrubbed / words if words else 0.0
