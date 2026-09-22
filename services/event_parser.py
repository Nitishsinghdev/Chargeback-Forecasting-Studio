"""Natural-language event parsing for forecast adjustments."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from models.domain_types import ParsedEvent

_PERCENT_RE = re.compile(
    r"(?P<sign>[+-])?\s*(?P<value>\d+(?:\.\d+)?)\s*(?:%|percent|pct)(?!\w)",
    re.IGNORECASE,
)
_DOLLAR_AMOUNT_RE = re.compile(
    r"(?P<sign>[+-])?\s*\$\s*(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_YEAR_MONTH_TOKEN_RE = re.compile(r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(?P<month>[A-Za-z]{3,9})\s+(?P<year>\d{4})\b",
)
_RANGE_RE = re.compile(
    r"\bfrom\s+(?P<start>[^,;]+?)\s+(?:to|through|until)\s+(?P<end>[^,;]+)",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"^[\s\"']*(?P<name>[A-Za-z0-9][A-Za-z0-9 _\-]{2,40}?)[\s\"']*(?:[:\-–]|,)\s+", re.IGNORECASE)
_FOR_TARGET_RE = re.compile(
    r"\bfor\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+?)(?:\s+in\s+|\s+from\s|,|$)",
    re.IGNORECASE,
)
_INCREASE_WORDS = {"increase", "up", "rise", "higher", "growth", "spike", "surge", "expand"}
_DECREASE_WORDS = {"decrease", "down", "lower", "drop", "reduction", "cut", "decline"}
_VOLUME_WORDS = {"volume", "cases", "count", "transactions", "claims"}
_LIABILITY_WORDS = {"liability", "amount", "chargeback", "cost", "dollars", "spend", "usd", "$"}


def _empty_event(raw: str, notes: List[str]) -> ParsedEvent:
    return ParsedEvent(
        raw_text=raw,
        event_name=None,
        effect_type="unknown",
        effect_value=None,
        direction="unknown",
        metric="unknown",
        business_unit=None,
        product=None,
        start_period=None,
        end_period=None,
        segment_hint=None,
        confidence=0.0,
        parse_notes=notes,
    )


def _parse_period_token(token: str) -> Optional[datetime]:
    token = token.strip()
    if not token:
        return None
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        return None
    ts = pd.Timestamp(parsed)
    return datetime(ts.year, ts.month, 1)


def _extract_periods(text: str) -> tuple[Optional[datetime], Optional[datetime], List[str]]:
    notes: List[str] = []
    range_match = _RANGE_RE.search(text)
    if range_match:
        start = _parse_period_token(range_match.group("start"))
        end = _parse_period_token(range_match.group("end"))
        if start:
            notes.append("start_period_from_range")
        if end:
            notes.append("end_period_from_range")
        return start, end, notes

    periods: List[datetime] = []
    for m in _MONTH_YEAR_RE.finditer(text):
        parsed = pd.to_datetime(f"{m.group('month')} {m.group('year')}", errors="coerce")
        if pd.isna(parsed):
            continue
        ts = pd.Timestamp(parsed)
        periods.append(datetime(ts.year, ts.month, 1))
    for m in _YEAR_MONTH_TOKEN_RE.finditer(text):
        try:
            y = int(m.group("year"))
            mo = int(m.group("month"))
            if 1 <= mo <= 12:
                periods.append(datetime(y, mo, 1))
        except ValueError:
            continue

    if not periods:
        return None, None, notes

    periods = sorted(set(periods))
    if len(periods) == 1:
        notes.append("single_period_detected")
        return periods[0], periods[0], notes
    notes.append("multi_period_detected_using_bounds")
    return periods[0], periods[-1], notes


def _extract_direction(text: str, signed_value: Optional[float]) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    if signed_value is not None:
        if signed_value > 0:
            return "increase"
        if signed_value < 0:
            return "decrease"
        return "neutral"
    if words & _INCREASE_WORDS and not (words & _DECREASE_WORDS):
        return "increase"
    if words & _DECREASE_WORDS and not (words & _INCREASE_WORDS):
        return "decrease"
    return "unknown"


def _extract_metric(text: str) -> str:
    lowered = text.lower()
    vol = any(w in lowered for w in _VOLUME_WORDS)
    liab = any(w in lowered for w in _LIABILITY_WORDS)
    if vol and liab:
        return "both"
    if vol:
        return "volume"
    if liab:
        return "liability"
    return "unknown"


def _is_year_token(value: str, span: tuple[int, int], text: str) -> bool:
    if len(value) == 4 and value.isdigit() and 1900 <= int(value) <= 2100:
        start, end = span
        tail = text[end : end + 3]
        head = text[max(0, start - 1) : start]
        if tail.startswith("-") or tail.startswith("/"):
            return True
        if head == "-":
            return True
    return False


def _apply_sign_from_words(text: str, val: float, explicit_sign: Optional[str]) -> float:
    if explicit_sign == "-":
        return -abs(val)
    if explicit_sign == "+":
        return abs(val)
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _DECREASE_WORDS and not (words & _INCREASE_WORDS):
        return -abs(val)
    if words & _INCREASE_WORDS:
        return abs(val)
    return val


def _extract_effect(text: str) -> tuple[str, Optional[float], List[str]]:
    notes: List[str] = []
    pct = _PERCENT_RE.search(text)
    if pct:
        val = float(pct.group("value"))
        val = _apply_sign_from_words(text, val, pct.group("sign"))
        notes.append("percent_effect_detected")
        return "percent", val, notes

    dollar = _DOLLAR_AMOUNT_RE.search(text)
    if dollar:
        val = float(dollar.group("value"))
        val = _apply_sign_from_words(text, val, dollar.group("sign"))
        notes.append("absolute_effect_detected")
        return "absolute", val, notes

    notes.append("effect_unknown")
    return "unknown", None, notes


def _match_known_value(text: str, candidates: Optional[Sequence[str]]) -> Optional[str]:
    if not candidates:
        return None
    lowered = text.lower()
    for cand in sorted((str(c) for c in candidates), key=len, reverse=True):
        c = cand.strip()
        if not c:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(c.lower()) + r"(?![A-Za-z0-9])")
        if pattern.search(lowered):
            return c
    return None


def _extract_for_target(text: str) -> Optional[str]:
    m = _FOR_TARGET_RE.search(text)
    if not m:
        return None
    return m.group("tgt").strip()


def _extract_event_name(text: str) -> Optional[str]:
    m = _NAME_RE.match(text)
    if m:
        return m.group("name").strip()
    q = re.search(r'"([^"]{3,60})"', text)
    if q:
        return q.group(1).strip()
    return None


def parse_event_text(
    text: str,
    segment_candidates: Optional[Sequence[str]] = None,
    business_units: Optional[Sequence[str]] = None,
    products: Optional[Sequence[str]] = None,
) -> ParsedEvent:
    """Parse a single natural-language event description."""
    raw = (text or "").strip()
    if not raw:
        return _empty_event("", ["empty_input"])

    effect_type, effect_value, effect_notes = _extract_effect(raw)
    start_p, end_p, period_notes = _extract_periods(raw)
    segment_hint = _match_known_value(raw, segment_candidates)
    product = _match_known_value(raw, products)
    business_unit = _match_known_value(raw, business_units)
    if business_unit is None and product is None:
        for_phrase = _extract_for_target(raw)
        if for_phrase:
            business_unit = _match_known_value(for_phrase, business_units) or (
                for_phrase if for_phrase and for_phrase[0].isupper() else None
            )
    elif business_unit is None and product is not None:
        for_phrase = _extract_for_target(raw)
        if for_phrase:
            business_unit = _match_known_value(for_phrase, business_units)

    direction = _extract_direction(raw, effect_value)
    metric = _extract_metric(raw)
    event_name = _extract_event_name(raw)

    confidence = 0.3
    if effect_type != "unknown":
        confidence += 0.3
    if start_p is not None:
        confidence += 0.15
    if segment_hint or business_unit or product:
        confidence += 0.15
    if event_name:
        confidence += 0.05
    confidence = min(1.0, confidence)

    notes = effect_notes + period_notes
    if segment_hint:
        notes.append("segment_hint_matched")
    if business_unit:
        notes.append("business_unit_matched")
    if product:
        notes.append("product_matched")

    return ParsedEvent(
        raw_text=raw,
        event_name=event_name,
        effect_type=effect_type,
        effect_value=effect_value,
        direction=direction,
        metric=metric,
        business_unit=business_unit,
        product=product,
        start_period=start_p,
        end_period=end_p or start_p,
        segment_hint=segment_hint,
        confidence=confidence,
        parse_notes=notes,
    )


def parse_event_batch(
    texts: Iterable[str],
    segment_candidates: Optional[Sequence[str]] = None,
    business_units: Optional[Sequence[str]] = None,
    products: Optional[Sequence[str]] = None,
) -> List[ParsedEvent]:
    """Parse multiple event strings preserving order."""
    return [
        parse_event_text(
            t,
            segment_candidates=segment_candidates,
            business_units=business_units,
            products=products,
        )
        for t in texts
    ]
