"""
Indian pharmaceutical market structure, from the real product master.

Answers the questions a commercial team actually asks of a product catalogue: who
holds the market, what does it cost, what is being withdrawn, and which molecules
are most contested.

All figures are measured from 253,973 real marketed medicines. See
:mod:`src.data.indian_medicines` for provenance and the two limits that shape how
the discontinuation flag is used here - it is **reported, never modelled**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.indian_medicines import load_indian_medicines
from src.logger import get_logger

log = get_logger(__name__)

#: Minimum products before a manufacturer is ranked. Below this a single withdrawn
#: line swings the discontinuation rate by tens of points.
MIN_PRODUCTS_FOR_RANKING: int = int(
    get_config().indian_medicines.min_products_for_ranking)


def _resolve(medicines: pd.DataFrame | None) -> pd.DataFrame:
    """Return the caller's frame, or load the real dataset."""
    return load_indian_medicines() if medicines is None else medicines


def market_kpis(medicines: pd.DataFrame | None = None) -> dict:
    """Headline figures for the Indian medicine market.

    Returns
    -------
    dict
        Product, manufacturer and ingredient counts, price percentiles,
        discontinuation rate and combination-product share.
    """
    df = _resolve(medicines)
    priced = df.loc[df["price_is_recorded"], "price_inr"]

    return {
        "products": int(len(df)),
        "manufacturers": int(df["manufacturer"].nunique()),
        "distinct_ingredients": int(df["primary_ingredient"].nunique()),
        "median_price_inr": round(float(priced.median()), 2),
        "p90_price_inr": round(float(priced.quantile(0.90)), 2),
        "max_price_inr": round(float(priced.max()), 2),
        "discontinued_pct": round(100 * float(df["is_discontinued"].mean()), 2),
        "discontinued_count": int(df["is_discontinued"].sum()),
        "combination_pct": round(100 * float(df["is_combination"].mean()), 2),
        "unrecorded_prices": int((~df["price_is_recorded"]).sum()),
        "top_manufacturer": str(df["manufacturer"].value_counts().index[0]),
        "top_manufacturer_products": int(df["manufacturer"].value_counts().iloc[0]),
    }


def manufacturer_concentration(medicines: pd.DataFrame | None = None,
                               top_n: int = 25) -> pd.DataFrame:
    """Rank manufacturers by portfolio breadth, with cumulative market share.

    Portfolio breadth (product count) is the only share measure this dataset
    supports - there are no sales volumes, so this is **not** revenue share. A firm
    with 50 high-volume products can outsell one with 2,000 niche listings.

    Returns
    -------
    pandas.DataFrame
        Columns ``manufacturer``, ``products``, ``share_pct``,
        ``cumulative_share_pct``, ``median_price_inr``, ``discontinued_pct``,
        ``combination_pct``.
    """
    df = _resolve(medicines)
    out = (df.groupby("manufacturer")
           .agg(products=("id", "size"),
                median_price_inr=("price_inr", "median"),
                discontinued_pct=("is_discontinued", "mean"),
                combination_pct=("is_combination", "mean"))
           .reset_index()
           .sort_values("products", ascending=False))

    out["share_pct"] = (100 * out["products"] / len(df)).round(3)
    out["cumulative_share_pct"] = out["share_pct"].cumsum().round(2)
    out["discontinued_pct"] = (100 * out["discontinued_pct"]).round(2)
    out["combination_pct"] = (100 * out["combination_pct"]).round(2)
    out["median_price_inr"] = out["median_price_inr"].round(2)

    log.info("Concentration: top 10 of %d manufacturers hold %.1f%% of listings",
             len(out), out["cumulative_share_pct"].iloc[min(9, len(out) - 1)])
    return out.head(top_n).reset_index(drop=True)


def concentration_summary(medicines: pd.DataFrame | None = None) -> pd.DataFrame:
    """How concentrated the market is, at a few cut-offs.

    A fragmented market and a concentrated one need different sourcing strategies,
    so this states which it is rather than leaving it to a chart.
    """
    df = _resolve(medicines)
    counts = df["manufacturer"].value_counts()
    shares = (100 * counts / len(df)).cumsum()

    rows = []
    for cut in (10, 25, 50, 100, 500):
        if cut <= len(shares):
            rows.append({"top_n_manufacturers": cut,
                         "cumulative_share_pct": round(float(shares.iloc[cut - 1]), 2)})
    out = pd.DataFrame(rows)
    # Herfindahl index on listing share - a standard concentration measure.
    hhi = float(((100 * counts / len(df)) ** 2).sum())
    out.attrs["hhi"] = round(hhi, 1)
    out.attrs["interpretation"] = (
        "Highly concentrated" if hhi > 2500
        else "Moderately concentrated" if hhi > 1500
        else "Fragmented / competitive")
    return out


def price_distribution(medicines: pd.DataFrame | None = None,
                       bins: int = 12) -> pd.DataFrame:
    """Price histogram on a log scale, with discontinuation rate per bucket.

    Log spacing is not cosmetic: prices span ₹1 to ₹436,000, so linear bins put
    99% of products in the first bucket and tell you nothing.

    Returns
    -------
    pandas.DataFrame
        Columns ``price_bucket``, ``bucket_low``, ``bucket_high``, ``products``,
        ``share_pct``, ``discontinued_pct``.
    """
    df = _resolve(medicines)
    priced = df[df["price_is_recorded"]].copy()

    edges = np.geomspace(max(priced["price_inr"].min(), 1.0),
                         priced["price_inr"].max(), bins + 1)
    priced["bucket"] = pd.cut(priced["price_inr"], bins=edges, include_lowest=True)

    out = (priced.groupby("bucket", observed=True)
           .agg(products=("id", "size"),
                discontinued_pct=("is_discontinued", "mean"))
           .reset_index())
    out["bucket_low"] = out["bucket"].map(lambda b: round(b.left, 2))
    out["bucket_high"] = out["bucket"].map(lambda b: round(b.right, 2))
    out["price_bucket"] = out.apply(
        lambda r: f"Rs {r['bucket_low']:,.0f}-{r['bucket_high']:,.0f}", axis=1)
    out["share_pct"] = (100 * out["products"] / len(priced)).round(2)
    out["discontinued_pct"] = (100 * out["discontinued_pct"]).round(2)
    return out[["price_bucket", "bucket_low", "bucket_high", "products",
                "share_pct", "discontinued_pct"]]


def discontinuation_analysis(medicines: pd.DataFrame | None = None,
                             min_products: int = MIN_PRODUCTS_FOR_RANKING) -> dict:
    """Where withdrawal concentrates - and where it conspicuously does not.

    The finding worth reporting is a contrast. Discontinuation is **flat across
    price** (about 2.6-3.7% in every quartile) but **wildly uneven across
    manufacturers** (3.11% overall against 40%+ for some large firms). A spread
    that large is unlikely to be product economics; it more plausibly reflects how
    recently each catalogue was refreshed in the source.

    That is why this returns a description and an explicit caveat rather than a
    model.

    Returns
    -------
    dict
        ``overall_pct``, ``by_price_band``, ``by_pack_form``,
        ``by_manufacturer`` (worst first), ``price_is_predictive``,
        ``manufacturer_spread_pp``, ``caveat``.
    """
    df = _resolve(medicines)

    by_price = (df.groupby("price_band", observed=True)
                .agg(products=("id", "size"),
                     discontinued_pct=("is_discontinued", "mean"))
                .reset_index())
    by_price["discontinued_pct"] = (100 * by_price["discontinued_pct"]).round(2)

    by_pack = (df.groupby("pack_form")
               .agg(products=("id", "size"),
                    discontinued_pct=("is_discontinued", "mean"))
               .reset_index())
    by_pack = by_pack[by_pack["products"] >= min_products]
    by_pack["discontinued_pct"] = (100 * by_pack["discontinued_pct"]).round(2)
    by_pack = by_pack.sort_values("discontinued_pct", ascending=False)

    by_manufacturer = (df.groupby("manufacturer")
                       .agg(products=("id", "size"),
                            discontinued=("is_discontinued", "sum"),
                            discontinued_pct=("is_discontinued", "mean"),
                            median_price_inr=("price_inr", "median"))
                       .reset_index())
    by_manufacturer = by_manufacturer[by_manufacturer["products"] >= min_products]
    by_manufacturer["discontinued_pct"] = (
        100 * by_manufacturer["discontinued_pct"]).round(2)
    by_manufacturer["discontinued"] = by_manufacturer["discontinued"].astype(int)
    by_manufacturer["median_price_inr"] = by_manufacturer["median_price_inr"].round(2)
    by_manufacturer = by_manufacturer.sort_values(
        "discontinued_pct", ascending=False).reset_index(drop=True)

    price_spread = float(by_price["discontinued_pct"].max()
                         - by_price["discontinued_pct"].min())
    mfr_spread = float(by_manufacturer["discontinued_pct"].max()
                       - by_manufacturer["discontinued_pct"].min())

    return {
        "overall_pct": round(100 * float(df["is_discontinued"].mean()), 2),
        "by_price_band": by_price,
        "by_pack_form": by_pack,
        "by_manufacturer": by_manufacturer,
        "price_band_spread_pp": round(price_spread, 2),
        "manufacturer_spread_pp": round(mfr_spread, 2),
        # A spread of a couple of points across quartiles is not a relationship.
        "price_is_predictive": bool(price_spread > 5.0),
        "caveat": (
            "Discontinuation varies by up to "
            f"{mfr_spread:.0f} percentage points between manufacturers but only "
            f"{price_spread:.1f} points across price quartiles. A manufacturer "
            "effect that large, with essentially no price effect, points at "
            "catalogue refresh timing in the source rather than product economics. "
            "Read this as a property of the listing, not a prediction of "
            "withdrawal.\n\n"
            "Quartiles do hide one real pattern: on log-spaced price buckets the "
            "rate is U-shaped, around 9% for products under Rs 3 and 7-8% above "
            "Rs 5,000 against 2.7% in the Rs 27-230 bulk. Both tails are tiny "
            "(a few hundred products each), so it is a weak signal - but it is "
            "consistent with very cheap and very expensive lines being the first "
            "rationalised."
        ),
    }


def ingredient_analysis(medicines: pd.DataFrame | None = None,
                        top_n: int = 20) -> pd.DataFrame:
    """Most-marketed active ingredients, with competition and price spread.

    ``brands`` is a direct measure of how contested a molecule is: 8,992 listings
    for one ingredient means intense generic competition.

    Returns
    -------
    pandas.DataFrame
        Columns ``ingredient``, ``brands``, ``manufacturers``,
        ``median_price_inr``, ``price_spread_inr``, ``discontinued_pct``.
    """
    df = _resolve(medicines)
    out = (df.groupby("primary_ingredient")
           .agg(brands=("id", "size"),
                manufacturers=("manufacturer", "nunique"),
                median_price_inr=("price_inr", "median"),
                p10=("price_inr", lambda s: s.quantile(0.10)),
                p90=("price_inr", lambda s: s.quantile(0.90)),
                discontinued_pct=("is_discontinued", "mean"))
           .reset_index()
           .rename(columns={"primary_ingredient": "ingredient"}))

    out["price_spread_inr"] = (out["p90"] - out["p10"]).round(2)
    out["median_price_inr"] = out["median_price_inr"].round(2)
    out["discontinued_pct"] = (100 * out["discontinued_pct"]).round(2)
    out = out[out["ingredient"] != "Unknown"]
    return (out.sort_values("brands", ascending=False)
            .head(top_n)
            .drop(columns=["p10", "p90"])
            .reset_index(drop=True))


def pack_form_analysis(medicines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Product mix and pricing by container type."""
    df = _resolve(medicines)
    out = (df.groupby("pack_form")
           .agg(products=("id", "size"),
                median_price_inr=("price_inr", "median"),
                median_pack_quantity=("pack_quantity", "median"),
                combination_pct=("is_combination", "mean"),
                discontinued_pct=("is_discontinued", "mean"))
           .reset_index())
    out = out[out["products"] >= MIN_PRODUCTS_FOR_RANKING]
    out["share_pct"] = (100 * out["products"] / len(df)).round(2)
    for column in ("combination_pct", "discontinued_pct"):
        out[column] = (100 * out[column]).round(2)
    out["median_price_inr"] = out["median_price_inr"].round(2)
    return out.sort_values("products", ascending=False).reset_index(drop=True)


__all__ = [
    "market_kpis", "manufacturer_concentration", "concentration_summary",
    "price_distribution", "discontinuation_analysis", "ingredient_analysis",
    "pack_form_analysis", "MIN_PRODUCTS_FOR_RANKING",
]
