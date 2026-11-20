"""
Rule-based news classification for AurumAI (Phase 6).

Each article gets:
  - category:              ONE primary category (the main driver the article is about)
  - secondary_categories:  other categories it also mentions
  - matched_terms:         the exact words that triggered each category (explainability)

How the primary category is chosen:
  1. Each category has a regex pattern.
  2. Score = 3 x (distinct terms matched in the headline) + 1 x (distinct terms in the description).
     Headlines say what an article is ABOUT; summaries often mention side topics.
  3. Highest score wins. Ties are broken by CATEGORY_PRIORITY (specific drivers first).
  4. No match -> OTHER. Fed RSS items are always FED (source rule).

These rules are deliberately simple and transparent; Phase 7 compares them with an LLM.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categories and patterns
# ---------------------------------------------------------------------------
# Patterns are case-SENSITIVE by default; (?i:...) marks case-insensitive parts.
# That lets "Fed" (the central bank) match without matching "fed" (the verb).
CATEGORY_PATTERNS: dict[str, str] = {
    "FED": r"\bFed\b|(?i:federal reserve|\bFOMC\b|\bpowell\b|monetary policy|fed funds)",
    "INFLATION": r"\bCPI\b|\bPCE\b|\bPPI\b|(?i:inflation|disinflation|consumer prices?|producer prices?|price index)",
    "LABOR": r"\bJOLTS\b|(?i:payrolls?|jobs report|unemployment|jobless|labou?r market|job openings)",
    # "dollar" as a currency, not "billion-dollar deal" or "$5 million dollar fund"
    "USD": r"\bDXY\b|(?i:(?<!billion )(?<!million )(?<!trillion )(?<!-)\bdollar\b(?!-)|greenback|dollar index)",
    # bond-market language only: not "Treasury Department" or "yield results"
    "YIELDS": r"(?i:Treasury (?:yields?|bonds?|notes?|bills?|market|auction)s?|\bTreasuries\b|bond yields?"
              r"|\b(?:2|10|30)-year (?:yield|note|bond|Treasury)|yield curve|bond market|real yields?)",
    "CENTRAL_BANK": r"\bECB\b|\bBOE\b|\bBOJ\b|\bPBOC\b|\bSNB\b|(?i:central banks?|Bank of England|Bank of Japan"
                    r"|People's Bank of China|Reserve Bank|Bundesbank|gold reserves)",
    "GOLD": r"\bXAU\b|\bGLD\b|(?i:\bgold\b(?! IRA)|bullion|precious metals?)",
    "COMMODITIES": r"\bOPEC\b|\bWTI\b|\bLNG\b|(?i:\boil\b|\bcrude\b|\bBrent\b|natural gas|copper|silver|commodit\w*)",
    "GEOPOLITICS": r"(?i:geopolitic\w*|\bwar\b|missiles?|airstrikes?|sanctions?|military|invasion|ceasefire"
                   r"|troops|drones?|nuclear|blockade|tariffs?|trade war)",
    "MARKET_RISK": r"(?i:recession|financial crisis|sell-?off|banking turmoil|safe[- ]haven|risk-off"
                   r"|market turmoil|volatility spike)",
}
OTHER = "OTHER"

# Tie-break order.
# GOLD first: an article reporting gold's own price/flows is a GOLD market report even if it
# names a driver ("Gold rises as Fed signals cuts"). These reports describe moves that already
# happened, so they must stay separate from driver news to avoid leakage in backtests.
# Then specific, scheduled macro drivers; broad themes last.
CATEGORY_PRIORITY = [
    "GOLD", "FED", "INFLATION", "LABOR", "YIELDS", "USD", "CENTRAL_BANK",
    "COMMODITIES", "GEOPOLITICS", "MARKET_RISK",
]
ALL_CATEGORIES = CATEGORY_PRIORITY + [OTHER]

HEADLINE_WEIGHT = 3
DESCRIPTION_WEIGHT = 1

_REGEX = {cat: re.compile(p) for cat, p in CATEGORY_PATTERNS.items()}
assert set(CATEGORY_PRIORITY) == set(CATEGORY_PATTERNS), "priority list must cover every category"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def _terms(rx: re.Pattern, text: str) -> set[str]:
    """Distinct matched terms, lowercased (so 'Fed' twice counts once)."""
    return {m.group(0).lower() for m in rx.finditer(text or "")}


def match_categories(headline: str, description: str = "") -> dict[str, tuple[int, list[str]]]:
    """Return {category: (score, matched_terms)} for every category that matched."""
    results = {}
    for cat, rx in _REGEX.items():
        h, d = _terms(rx, headline), _terms(rx, description)
        score = HEADLINE_WEIGHT * len(h) + DESCRIPTION_WEIGHT * len(d - h)
        if score > 0:
            results[cat] = (score, sorted(h | d))
    return results


def classify_text(headline: str, description: str = "") -> tuple[str, list[str], dict]:
    """Return (primary category, secondary categories, raw matches)."""
    matches = match_categories(headline, description)
    if not matches:
        return OTHER, [], matches
    ranked = sorted(matches, key=lambda c: (-matches[c][0], CATEGORY_PRIORITY.index(c)))
    return ranked[0], ranked[1:], matches


def classify_news(news: pd.DataFrame) -> pd.DataFrame:
    """Add category, secondary_categories and matched_terms columns to a news frame."""
    out = news.copy()
    cats, secs, terms = [], [], []
    for headline, desc, provider in zip(out["headline"].fillna(""),
                                        out["description"].fillna(""),
                                        out["provider"].fillna("")):
        primary, secondary, matches = classify_text(str(headline), str(desc))
        if provider == "fed_rss" and primary != "FED":   # source rule: the Fed's own releases
            secondary = [c for c in [primary] + secondary if c not in ("FED", OTHER)]
            primary = "FED"
        cats.append(primary)
        secs.append(",".join(secondary))
        terms.append("; ".join(f"{c}:{'|'.join(t)}" for c, (_, t) in matches.items()))
    out["category"] = cats
    out["secondary_categories"] = secs
    out["matched_terms"] = terms
    logger.info("classify_news: %s", out["category"].value_counts().to_dict())
    return out


# ---------------------------------------------------------------------------
# Manual labels and evaluation
# ---------------------------------------------------------------------------
LABEL_COLUMNS = ["label_id", "provider", "source", "headline", "description", "true_category", "notes"]


def export_label_sheet(news: pd.DataFrame, path: str | Path, n: int = 50, seed: int = 42,
                       exclude_providers: tuple[str, ...] = ("fed_rss",)) -> pd.DataFrame:
    """
    Write a random sample of articles for manual labeling.

    - Predictions are NOT included, so they can't bias your labels.
    - Fed RSS items are excluded: the source rule makes them trivially FED.
    - IDs get an "a_" prefix so Excel can't turn hex IDs like "12e45..." into numbers.
    - Refuses to overwrite an existing sheet, so labels are never lost.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; delete or rename it first")
    pool = news[~news["provider"].isin(exclude_providers)]
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    sheet = pd.DataFrame({
        "label_id": "a_" + sample["article_id"].astype(str),
        "provider": sample["provider"],
        "source": sample["source"],
        "headline": sample["headline"],
        "description": sample["description"],
        "true_category": "",
        "notes": "",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    logger.info("Wrote %d articles to %s for labeling", len(sheet), path)
    return sheet


def evaluate_classifier(labels_path: str | Path, news: pd.DataFrame) -> dict:
    """
    Compare rule-based categories with your manual labels.

    Returns accuracy, a per-category precision/recall table, the confusion matrix,
    and the misclassified rows (the most useful output: read them to improve the rules).
    """
    labels = pd.read_csv(labels_path, dtype=str)
    labels = labels[labels["true_category"].fillna("").str.strip() != ""].copy()
    labels["true_category"] = labels["true_category"].str.strip().str.upper()
    bad = set(labels["true_category"]) - set(ALL_CATEGORIES)
    if bad:
        raise ValueError(f"Unknown labels {sorted(bad)}; use one of {ALL_CATEGORIES}")

    labels["article_id"] = labels["label_id"].str.removeprefix("a_")
    pred = classify_news(news)[["article_id", "category", "matched_terms"]]
    df = labels.merge(pred, on="article_id", how="inner")
    if len(df) < len(labels):
        logger.warning("%d labeled rows not found in news data", len(labels) - len(df))

    y_true, y_pred = df["true_category"], df["category"]
    cats = sorted(set(y_true) | set(y_pred), key=ALL_CATEGORIES.index)

    rows = []
    for c in cats:
        tp = ((y_pred == c) & (y_true == c)).sum()
        pred_c, true_c = (y_pred == c).sum(), (y_true == c).sum()
        rows.append({"category": c,
                     "precision": tp / pred_c if pred_c else float("nan"),
                     "recall": tp / true_c if true_c else float("nan"),
                     "support": int(true_c)})

    return {
        "n_labeled": len(df),
        "accuracy": float((y_true == y_pred).mean()),
        "per_category": pd.DataFrame(rows).set_index("category").round(2),
        "confusion": pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["predicted"]),
        "errors": df.loc[y_true != y_pred, ["headline", "true_category", "category", "matched_terms"]],
    }
