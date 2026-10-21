"""Pseudonymisation: tutor -> TUTOR, students -> S1..Sn, also inside the text.

The returned mapping (real display name -> pseudonym) must never leave the process or local cache.
"""

from __future__ import annotations

import re

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
        if tok.lower() in STOPWORDS:
            return tok
        best, best_score = None, 0.0
        for name, pseudo in fuzzy.items():
            score = fuzz.ratio(tok.lower(), name)
            if score > best_score:
                best, best_score = pseudo, score
        return best if best and best_score >= FUZZY_MIN else tok

    return _CAP_TOKEN.sub(fuzzy_sub, text)


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
