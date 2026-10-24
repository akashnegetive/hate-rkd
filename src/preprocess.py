"""Text preprocessing.

Two pipelines are implemented so they can be compared fairly:

1. `denoise_remove`  - the paper's six parsing steps (Sec. III-C, Table I):
       mjb-rep  mojibake / HTML-entity repair
       sps-rem  special-symbol removal   (emojis are DELETED)
       url-rem  URL removal
       rpc-elm  repeated-character elimination
       usr-rem  username removal
       htg-rem  hashtag removal          (hashtags are DELETED)

2. `denoise_spn`     - OUR semantic-preserving normalisation (SPN).
   Same as above, except the two steps that throw away meaning are replaced:
       sps-rem -> emo-txt : emoji  -> its English name     ("\U0001F621" -> "pouting face")
       htg-rem -> htg-seg : #hashtag -> segmented words     ("#BanIslam" -> "ban islam")
   Motivation: in the paper's own error analysis (Table IX, Exp. 4) the hateful
   intent lives in "#StopImmigration #BanIslam"; the paper's denoiser deletes it.

3. Hashtag dropout (training only). Hashtags are also TOPIC SHORTCUTS: in the
   SemEval train split #buildthatwall is 100 % hate, in the test split 24 %.
   With mark=True, segmented hashtags are wrapped in sentinels
   "\u27e6 ... \u27e7"; during training `drop_marked` deletes each wrapped
   hashtag with probability p, so the model must also learn from the body text.
   At evaluation time the text is used unmarked (all hashtags kept).

4. `spn_emoji`: emojis -> text, hashtags removed as in the paper (isolates the
   effect of emojis from the effect of hashtags).
"""
import html
import re

import emoji

try:
    import ftfy  # fixes mojibake such as "Ã©" or "ðŸ˜‚"
except ImportError:  # optional dependency
    ftfy = None

try:
    from wordsegment import load as _ws_load, segment as _ws_segment
    _ws_load()
except Exception:  # wordsegment missing -> fall back to camel-case split only
    _ws_segment = None


URL_RE = re.compile(r"(\S+://\S+|www\.\S+|\b\w+\.(?:co|com|ly|io|org|net)/\S*)", re.I)
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_PUNCT_RE = re.compile(r"([^\w\s])\1+")          # "?????" -> "?"
REPEAT_CHAR_RE = re.compile(r"([a-zA-Z])\1{2,}")       # "soooo" -> "soo"
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
HT_OPEN, HT_CLOSE = "\u27e6", "\u27e7"                  # hashtag-span sentinels
NON_TEXT_RE = re.compile(r"[^A-Za-z0-9\s.,!?'\"\-:;()&\u27e6\u27e7]")  # what sps-rem deletes
MARKED_RE = re.compile("\u27e6([^\u27e7]*)\u27e7")
SPACES_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# shared steps
# --------------------------------------------------------------------------
def repair_mojibake(text: str) -> str:
    """mjb-rep: '&#128588;' -> real emoji, 'ðŸ˜‚' -> real emoji, '&amp;' -> '&'."""
    text = html.unescape(html.unescape(text))
    if ftfy is not None:
        text = ftfy.fix_text(text)
    return text


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_users(text: str) -> str:
    return USER_RE.sub(" ", text)


def squash_repeats(text: str) -> str:
    """rpc-elm: 'bitch?????' -> 'bitch?', 'soooo' -> 'soo'."""
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    return REPEAT_CHAR_RE.sub(r"\1\1", text)


def clean_spaces(text: str) -> str:
    return SPACES_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# paper-only steps (destructive)
# --------------------------------------------------------------------------
def remove_special_symbols(text: str) -> str:
    """sps-rem: delete emojis and any other non-text symbol."""
    return NON_TEXT_RE.sub(" ", text)


def remove_hashtags(text: str) -> str:
    """htg-rem: delete '#word' completely."""
    return HASHTAG_RE.sub(" ", text)


# --------------------------------------------------------------------------
# our steps (semantic-preserving)
# --------------------------------------------------------------------------
def emoji_to_text(text: str) -> str:
    """emo-txt: replace each emoji by its CLDR name, e.g. '\U0001F602' -> ' face with tears of joy '."""
    text = emoji.demojize(text, delimiters=(" <<", ">> "))
    text = re.sub(r"(<<[^<>]+>>)(\s*\1)+", r"\1", text)      # repeated emoji -> one (like rpc-elm)
    return re.sub(r"<<([^<>]+)>>", lambda m: " " + m.group(1).replace("_", " ") + " ", text)


def segment_hashtag(tag: str) -> str:
    """Split a hashtag body into words.

    1. camelCase / PascalCase  : 'StopImmigration' -> 'stop immigration'
    2. all lower / all upper   : 'buildthewall'    -> 'build the wall'  (wordsegment, unigram LM)
    """
    parts = CAMEL_RE.findall(tag)
    if len(parts) > 1:
        return " ".join(p.lower() for p in parts)
    if _ws_segment is not None and len(tag) > 3:
        return " ".join(_ws_segment(tag))
    return tag.lower()


def hashtags_to_words(text: str, mark: bool = False) -> str:
    """htg-seg: '#BanIslam' -> 'ban islam' (the '#' itself carries no meaning)."""
    o, c = (HT_OPEN, HT_CLOSE) if mark else ("", "")
    return HASHTAG_RE.sub(lambda m: " " + o + segment_hashtag(m.group(1)) + c + " ", text)


def strip_marks(text: str) -> str:
    """Marked text -> normal text (all hashtags kept)."""
    return clean_spaces(text.replace(HT_OPEN, " ").replace(HT_CLOSE, " "))


def drop_marked(text: str, p: float, rng) -> str:
    """Hashtag dropout: delete each marked hashtag with probability p."""
    return clean_spaces(MARKED_RE.sub(lambda m: " " if rng.random() < p else " " + m.group(1) + " ", text))


# --------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------
def denoise_remove(text: str) -> str:
    """Paper's denoiser (Table I order)."""
    text = repair_mojibake(text)
    text = remove_urls(text)
    text = remove_users(text)
    text = remove_hashtags(text)
    text = remove_special_symbols(text)
    text = squash_repeats(text)
    return clean_spaces(text)


def denoise_spn(text: str, mark: bool = False) -> str:
    """Our semantic-preserving normalisation (mark=True keeps hashtag sentinels)."""
    text = repair_mojibake(text)
    text = remove_urls(text)
    text = remove_users(text)
    text = hashtags_to_words(text, mark)
    text = emoji_to_text(text)
    text = remove_special_symbols(text)   # only leftovers (e.g. '|', '~') are dropped now
    text = squash_repeats(text)
    return clean_spaces(text)


def denoise_spn_emoji(text: str) -> str:
    """Emojis -> text, hashtags removed (paper's htg-rem)."""
    text = repair_mojibake(text)
    text = remove_urls(text)
    text = remove_users(text)
    text = remove_hashtags(text)
    text = emoji_to_text(text)
    text = remove_special_symbols(text)
    text = squash_repeats(text)
    return clean_spaces(text)


def get_preprocessor(name: str, mark: bool = False):
    if name == "none":
        return lambda t: t
    if name == "remove":
        return denoise_remove
    if name == "spn":
        return (lambda t: denoise_spn(t, mark=True)) if mark else denoise_spn
    if name == "spn_emoji":
        return denoise_spn_emoji
    raise ValueError(f"unknown preprocess '{name}'")


def has_emoji_or_hashtag(text: str) -> bool:
    """Used to build the 'emoji/hashtag' evaluation slice."""
    text = repair_mojibake(text)
    return bool(HASHTAG_RE.search(text)) or emoji.emoji_count(text) > 0
