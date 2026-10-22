"""Pseudonymisation: tutor -> TUTOR, students -> S1..Sn, also inside the text.

The returned mapping (real display name -> pseudonym) must never leave the process or local cache.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import cache

from rapidfuzz import fuzz
from wordfreq import zipf_frequency

from cqa.schemas import Transcript

FUZZY_MIN = 85  # >= this: the same child, replaced by their pseudonym
NAME_FUZZY_MIN = 70  # >= this (below FUZZY_MIN): likely an ASR-mangled name, replaced by [NAME] (E11b)
COMMON_ZIPF = 4.0  # wordfreq zipf (en) at or above this: ordinary English, never fuzzy-matched or scrubbed (E11b)
WORD_NAME_ZIPF = 4.5  # a display-name token this common ("Will") is a name only when capitalised (E11c)
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


@cache
def is_common_word(word: str, min_zipf: float = COMMON_ZIPF) -> bool:
    return zipf_frequency(word.lower(), "en") >= min_zipf


def _keep(tok: str, common_zipf: float) -> bool:
    return tok.lower() in ALLOWED_CAPS or is_common_word(tok, common_zipf)


def is_word_name(name_token: str) -> bool:
    return " " not in name_token and is_common_word(name_token, WORD_NAME_ZIPF)


def _name_pattern(name: str) -> re.Pattern[str]:
    if is_word_name(name):  # capitalised first letter only: "Will" is the child, "will" is the verb
        return re.compile(rf"\b{re.escape(name[0].upper())}(?i:{re.escape(name[1:])})\b")
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


def _replace_names(
    text: str,
    patterns: list[tuple[re.Pattern[str], str]],
    fuzzy: dict[str, str],
    name_min: float = NAME_FUZZY_MIN,
    common_zipf: float = COMMON_ZIPF,
) -> str:
    for pat, pseudo in patterns:
        text = pat.sub(pseudo, text)

    def fuzzy_sub(m: re.Match[str]) -> str:
        # Any case: lowercase and ASR-mangled name variants leak too (E11, E11b). The common-word skip only
        # applies to the loose band; a close match is replaced even when it is an English word.
        tok = m.group(0)
        if tok.lower() in ALLOWED_CAPS:
            return tok
        best, best_score = tok, 0.0
        for name, pseudo in fuzzy.items():
            if tok[0].islower() and is_word_name(name):
                continue
            score = fuzz.ratio(tok.lower(), name)
            if score > best_score:
                best, best_score = pseudo, score
        if best_score >= FUZZY_MIN:
            return best
        if best_score < name_min:
            return tok
        # Loose band: a common word is kept, unless it is capitalised mid-sentence, where it reads as the name
        # (names that are also common words, D16 review), or capitalised at a sentence start and used as a
        # vocative ("Name, ..."), the rule scrub_names applies on the whisper path; otherwise the capital says nothing
        if not is_common_word(tok, common_zipf):
            return NAME_TOKEN
        if not tok[0].isupper():
            return tok
        at_start = bool(_SENTENCE_START.search(m.string[: m.start()]))
        vocative = m.string[m.end() : m.end() + 1] in {",", "?", "!"}
        return NAME_TOKEN if (not at_start or vocative) else tok

    return _WORD_TOKEN.sub(fuzzy_sub, text)


def pseudonymise(
    transcript: Transcript, name_min: float = NAME_FUZZY_MIN, common_zipf: float = COMMON_ZIPF
) -> tuple[Transcript, dict[str, str]]:
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
    patterns = [(_name_pattern(name), pseudo) for name, pseudo in sorted(exact.items(), key=lambda kv: -len(kv[0]))]

    turns = [
        t.model_copy(
            update={
                "speaker": mapping.get(t.speaker, t.speaker),
                "text": _replace_names(t.text, patterns, fuzzy, name_min, common_zipf),
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


def scrub_names(text: str, common_zipf: float = COMMON_ZIPF, strict: bool = False) -> str:
    """[NAME] for capitalised words that read as names. strict (no display names to pseudonymise from, i.e. the
    whisper path, E27): any capitalised non-allowlisted word used as a vocative ("X, is it?") is scrubbed even if it
    is a common word, and a capitalised word that is not common English is scrubbed even at a sentence start."""

    def sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok.lower() in ALLOWED_CAPS or _PSEUDONYM.match(tok):
            return tok
        at_start = bool(_SENTENCE_START.search(text[: m.start()]))
        vocative = text[m.end() : m.end() + 1] in {",", "?", "!"}
        if strict:
            return NAME_TOKEN if vocative or not is_common_word(tok, common_zipf) else tok
        if is_common_word(tok, common_zipf):
            return tok
        return NAME_TOKEN if (not at_start or vocative) else tok

    return _CAP_TOKEN.sub(sub, text)


def scrub_transcript(transcript: Transcript, common_zipf: float = COMMON_ZIPF, strict: bool = False) -> Transcript:
    turns = [t.model_copy(update={"text": scrub_names(t.text, common_zipf, strict)}) for t in transcript.turns]
    return transcript.model_copy(update={"turns": turns})


ENGLISH_LANGUAGES = {"en", "english"}
_LOWER_LATIN = re.compile(r"\b[a-z]{3,}\b")  # Latin script only: Malayalam or other scripts never match


def scrub_non_english(text: str) -> str:
    """E27 (c): a lowercase Latin word that is not English at all (wordfreq zipf 0) and not allowlisted reads as a
    name whisper wrote in lowercase. Only for an English class on the whisper path (see anonymise)."""

    def sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        return tok if tok in ALLOWED_CAPS or zipf_frequency(tok, "en") > 0 else NAME_TOKEN

    return _LOWER_LATIN.sub(sub, text)


# E28: words a tutor says as a whole one-word sentence that are not names
INTERJECTIONS = {
    "okay", "ok", "yes", "no", "yeah", "good", "great", "nice", "right", "correct", "wrong", "perfect", "excellent",
    "hello", "hi", "bye", "sorry", "thanks", "wow", "done", "next", "wait", "listen", "look", "see", "now", "so",
    "well",
}  # fmt: skip
# a capitalised word at a turn or sentence start that is the whole sentence: "Name." or "Name" at the end of a turn
_ONE_WORD_SENTENCE = re.compile(r"(?:^|(?<=[.?!])\s+)([A-Z][a-zA-Z]{2,})(?=\.|\s*$)")


def scrub_one_word_sentences(text: str) -> str:
    """E28: a capitalised word forming a whole sentence ("Name. [NAME], your mic is off.") reads as a name, even
    when it is a common word, unless it is an interjection or a chess term. English class, whisper path only."""

    def sub(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok.lower() in INTERJECTIONS or tok.lower() in CHESS_TERMS or _PSEUDONYM.match(tok):
            return m.group(0)
        return m.group(0)[: m.start(1) - m.start(0)] + NAME_TOKEN

    return _ONE_WORD_SENTENCE.sub(sub, text)


def anonymise(
    transcript: Transcript, name_min: float = NAME_FUZZY_MIN, common_zipf: float = COMMON_ZIPF
) -> tuple[Transcript, dict[str, str]]:
    """The only transcript text that may leave the process. Expects roles to be assigned.

    VTT: display names -> pseudonyms (fuzzy variants below FUZZY_MIN -> [NAME]), then the [NAME] scrubber for
    names said aloud that are nobody's display name (a parent's account, absent children). Whisper: scrubber
    only. Common English words are kept unless they exactly match a display name.
    """
    mapping: dict[str, str] = {}
    if transcript.source == "vtt":
        transcript, mapping = pseudonymise(transcript, name_min, common_zipf)
    # no display names (whisper, or a transcript without speaker names): the scrubber is the only defence (E27)
    transcript = scrub_transcript(transcript, common_zipf, strict=not mapping)
    if transcript.source == "whisper" and transcript.language.lower() in ENGLISH_LANGUAGES:
        # (c): never for another class language, whose words are "not English" by definition (E27)
        turns = [
            t.model_copy(update={"text": scrub_one_word_sentences(scrub_non_english(t.text))}) for t in transcript.turns
        ]
        transcript = transcript.model_copy(update={"turns": turns})
    return transcript, mapping


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
    capitalised: set[str] = field(default_factory=set)  # hit tokens written capitalised at least once


def leak_counts(text: str, template: str, display_names: list[str]) -> LeakCounts:
    """Audit one judge input. Tokens that also occur in the fixed prompt template are ignored."""
    skip = {w.lower() for w in _WORD.findall(template)} | ALLOWED_CAPS | {"tutor", "name"}
    caps = [m.group(0) for m in _CAP_TOKEN.finditer(text) if m.group(0).lower() not in skip]
    names = display_name_tokens(display_names)
    # Same exception as the scrubber: lowercase "will" is not the child "Will" (is_word_name, E11c).
    lower_names = {n for n in names if not is_word_name(n)}
    counts: dict[tuple[str, bool], int] = {}  # (lowercased token, written in lowercase) -> occurrences
    for w in _WORD.findall(text):
        if len(w) >= 3 and w.lower() not in skip:
            key = (w.lower(), w.islower())
            counts[key] = counts.get(key, 0) + 1
    best_by_token: dict[str, tuple[float, str]] = {}
    capitalised: set[str] = set()
    n_hits = 0
    for (w, lower), n in counts.items():
        best = max(((fuzz.ratio(w, x), x) for x in (lower_names if lower else names)), default=(0.0, ""))
        if best[0] >= AUDIT_FUZZY_MIN:
            n_hits += n
            best_by_token[w] = max(best_by_token.get(w, best), best)
            if not lower:
                capitalised.add(w)
    hits = [(w, name, round(score, 1)) for w, (score, name) in best_by_token.items()]
    return LeakCounts(
        caps=len(caps),
        caps_distinct=len({c.lower() for c in caps}),
        name_hits=n_hits,
        names_hit=len({n for _, n, _ in hits}),
        hits=hits,
        capitalised=capitalised,
    )


OVER_SCRUB_ZIPF = 3.5  # "common English word" for metric (c); fixed so scrubber variants are comparable


def scrubbed_words(before: str, after: str) -> list[str]:
    """Words of `before` that anonymisation replaced, by word-level alignment with `after`."""
    a, b = before.split(), after.split()
    out: list[str] = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            out += [w for x in a[i1:i2] for w in _WORD.findall(x) if len(w) >= 3]
    return out


def over_scrub_share(raw: Transcript, anon: Transcript, min_zipf: float = OVER_SCRUB_ZIPF) -> float:
    """(c) Common English words that anonymisation replaced, as a share of tutor words."""
    words = scrubbed = 0
    for r, a in zip(raw.turns, anon.turns, strict=True):
        if r.role == "tutor":
            words += len(r.text.split())
            scrubbed += sum(is_common_word(w, min_zipf) for w in scrubbed_words(r.text, a.text))
    return scrubbed / words if words else 0.0
