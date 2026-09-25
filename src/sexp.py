"""Minimal S-expression parser for rcssserver protocol messages.

Messages such as ``(see 12 ((b) 5.2 10) ((f c) 10.5 -3))`` become nested lists
``['see', 12.0, [['b'], 5.2, 10.0], [['f', 'c'], 10.5, -3.0]]``. Numeric atoms
are converted to float; quoted strings keep their content without quotes.
"""

from typing import Any, List, Union

Atom = Union[str, float]
SExp = Union[Atom, List[Any]]


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "()":
            tokens.append(c)
            i += 1
        elif c.isspace() or c == "\x00":
            i += 1
        elif c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise ValueError(f"unterminated string at {i}")
            tokens.append(text[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in '()"\x00' and not text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _atom(token: str) -> Atom:
    if token.startswith('"'):
        return token[1:-1]
    try:
        return float(token)
    except ValueError:
        return token


def parse(text: str) -> SExp:
    """Parse one S-expression (trailing NULs and whitespace are ignored)."""
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("empty message")
    stack: List[List[Any]] = []
    result: Any = None
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            done = stack.pop()
            if stack:
                stack[-1].append(done)
            else:
                result = done
                break
        else:
            if not stack:
                return _atom(tok)
            stack[-1].append(_atom(tok))
    if result is None:
        raise ValueError("unbalanced '('")
    return result


def pairs_to_dict(items: List[Any]) -> dict:
    """Convert ``[['name', value], ...]`` into ``{'name': value}`` (single-value pairs only)."""
    out = {}
    for item in items:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
            out[item[0]] = item[1]
    return out
