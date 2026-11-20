"""
Regression tests for the rule-based classifier.

Run from the project root:  python -m pytest tests -v

Each case is a sentence that once fooled (or could fool) a keyword rule.
If a pattern change breaks one, the test fails before bad labels reach the data.
"""
import pandas as pd
import pytest

from src.classify import OTHER, classify_news, classify_text

PRIMARY_CASES = [
    # (headline, description, expected primary category)
    ("Fed holds rates steady, Powell signals patience", "", "FED"),
    ("US CPI rises 0.4% in October, above forecasts", "", "INFLATION"),
    ("Nonfarm payrolls beat expectations as jobless rate falls", "", "LABOR"),
    ("10-year Treasury yields climb to 4.6%", "", "YIELDS"),
    ("Dollar slips against the yen", "", "USD"),
    ("ECB cuts interest rates for third time", "", "CENTRAL_BANK"),
    ("Gold hits record high above $4,000", "", "GOLD"),
    ("Oil settles flat as Hormuz disruptions offset demand worries", "", "COMMODITIES"),
    ("Russia launches missile strikes on Kyiv", "", "GEOPOLITICS"),
    ("Global stocks tumble as recession fears grow", "", "MARKET_RISK"),
    # headline beats description: the summary mentions oil, but the story is about gold
    ("Gold climbs to two-week high", "Oil and copper were mixed.", "GOLD"),
    # gold market report that names a driver -> GOLD (driver kept as secondary)
    ("Gold rises as Fed signals cuts", "", "GOLD"),
    # pure driver news, gold not mentioned -> the driver
    ("Fed signals cuts as inflation cools", "", "FED"),
]

FALSE_POSITIVE_CASES = [
    # (headline, category that must NOT be the primary)
    ("US Treasury Department sanctions Hezbollah-linked charity", "YIELDS"),
    ("Treasury Secretary says talks will continue", "YIELDS"),
    ("Talks failed to yield a breakthrough", "YIELDS"),
    ("Ford signs billion-dollar battery deal", "USD"),
    ("Startup raises $5 million dollar fund", "USD"),
    ("Investors fed up with slow earnings growth", "FED"),
    ("Goldman Sachs beats profit estimates", "GOLD"),
    ("Best Gold IRA rollover companies 2026", "GOLD"),
]


@pytest.mark.parametrize("headline,description,expected", PRIMARY_CASES)
def test_primary_category(headline, description, expected):
    primary, _, _ = classify_text(headline, description)
    assert primary == expected


@pytest.mark.parametrize("headline,wrong", FALSE_POSITIVE_CASES)
def test_no_false_positive(headline, wrong):
    primary, secondary, _ = classify_text(headline)
    assert primary != wrong and wrong not in secondary


def test_unmatched_is_other():
    assert classify_text("Apple unveils new iPhone")[0] == OTHER


def test_fed_rss_source_rule():
    df = pd.DataFrame({"headline": ["Minutes of the meeting released"], "description": [""],
                       "provider": ["fed_rss"]})
    assert classify_news(df)["category"].iloc[0] == "FED"


def test_secondary_categories_recorded():
    # three-way tie in the headline: GOLD wins on priority, the drivers become secondary
    primary, secondary, _ = classify_text("Gold rises as dollar weakens after soft CPI")
    assert primary == "GOLD"
    assert set(secondary) == {"INFLATION", "USD"}
