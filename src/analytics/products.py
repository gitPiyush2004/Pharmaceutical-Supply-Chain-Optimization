"""
Product catalogue and procurement pricing, from the real USAID SCMS dataset.

Why this lives on the shipment data
-----------------------------------
SCMS is usually described as a logistics dataset, and it is - but it is *also* a
product catalogue. Every line item carries the molecule, brand, dosage, dosage form,
pack size, the price actually paid and the factory that made it:

======================  =======  ==================================================
``item_description``    184      Distinct catalogue items
``molecule``            86       Active molecules / test types
``brand``               48       Brands, the largest of which is ``Generic``
``dosage``              54       Strengths
``dosage_form``         17       Tablet, capsule, oral solution, test kit, ...
``manufacturing_site``  88       Factories
``vendor``              73       Suppliers
======================  =======  ==================================================

An earlier version of this project used a separate 253,973-row Indian medicine
catalogue for this analysis. It was dropped, because it answered the same question
less well: it carries *list* prices for products nobody in the dataset bought, while
these are **prices actually paid**, on the same rows as the delivery performance. One
dataset, two questions, and the two answers can be joined.

The trap this module is built around
------------------------------------
Pool all ten years together and the same molecule, at the same strength and in the
same form, appears to have been bought at prices differing by a median of **5.0x**
across 30 products. That number is inflated and it would be wrong to quote it.

Antiretroviral prices collapsed over this period - Efavirenz 600mg fell from a
median of $0.56 in 2006 to $0.11 in 2015, a **80.4%** decline - so a pooled spread
mixes "different supplier" with "different decade". Restricted to a **single year**,
the median spread is **2.5x** over 89 product-years. Pooling therefore inflates it by
almost exactly a factor of two.

This is the same mistake as the fulfilment-route finding in
:mod:`src.analytics.experiments`, where a pooled 11.9-point service gap turned out to
be 1.9 points before 2011 and 20.5 after. Finding the identical trap twice in one
dataset is why :func:`price_spread_within_year` exists alongside
:func:`price_spread`, and why the page shows both side by side.

What explains the remaining spread
----------------------------------
A 2.5x range could still be noise, so :func:`brand_premium` checks. It is not noise:
where both a generic and an originator-branded version of the same product were
bought in the same year, the branded one costs a median of **2.1x** more across 41
product-years. Nevirapine 200mg is the clearest case - nine Indian factories supplied
it generically at $0.04-$0.06 per unit against $0.30-$0.39 for ``Viramune``.

That turns a statistic into a procurement finding: the spread is structural, it maps
onto suppliers a buyer can choose between, and it is therefore actionable.

Example
-------
>>> from src.analytics.products import pricing_summary
>>> pricing_summary()["within_year_median_spread_x"]
2.5
>>> pricing_summary()["pooled_median_spread_x"]      # inflated - do not quote
5.0
"""

from __future__ import annotations

import pandas as pd

from src.analytics.procurement import _resolve
from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)

#: Columns that together identify "the same product" for a price comparison.
#: Molecule alone is not enough - a 200mg tablet and a 600mg tablet of the same
#: molecule are different goods and comparing their unit prices is meaningless.
PRODUCT_KEY: tuple[str, ...] = ("molecule", "dosage", "dosage_form")


def _cfg():
    """Product-analysis thresholds from ``config.scms.products``."""
    return get_config().scms.products


def _priced(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable unit price and a complete product key.

    A unit price of zero is not a free medicine - it is an unrecorded one, so those
    rows are excluded rather than treated as a $0 purchase that would drag every
    minimum to zero and make each spread ratio infinite.
    """
    out = df[df["unit_price_usd"] > 0].copy()
    for column in PRODUCT_KEY:
        out = out[out[column].notna()]
    return out


# ---------------------------------------------------------------------------
# Catalogue structure
# ---------------------------------------------------------------------------
def catalogue_kpis(scms: pd.DataFrame | None = None) -> dict:
    """Headline figures describing the catalogue behind the shipments.

    Returns
    -------
    dict
        Distinct counts for each catalogue dimension, the generic share, and the
        priced-row coverage the pricing analysis is measured on.
    """
    df = _resolve(scms)
    priced = _priced(df)
    generic = int((df["brand"].astype("string") == "Generic").sum())

    return {
        "line_items": int(len(df)),
        "catalogue_items": int(df["item_description"].nunique()),
        "molecules": int(df["molecule"].nunique()),
        "brands": int(df["brand"].nunique()),
        "dosages": int(df["dosage"].nunique()),
        "dosage_forms": int(df["dosage_form"].nunique()),
        "manufacturing_sites": int(df["manufacturing_site"].nunique()),
        "vendors": int(df["vendor"].nunique()),
        "product_groups": int(df["product_group"].nunique()),
        "generic_line_items": generic,
        "generic_share_pct": round(100 * generic / len(df), 2),
        "priced_line_items": int(len(priced)),
        "priced_coverage_pct": round(100 * len(priced) / len(df), 2),
        "total_value_usd": round(float(df["line_value_usd"].sum()), 0),
    }


def product_mix(scms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Commodity value and shipment count by product group, largest first.

    Returns
    -------
    pandas.DataFrame
        Columns ``product_group``, ``line_items``, ``value_usd``, ``value_share_pct``,
        ``molecules``, ``on_time_pct``.
    """
    df = _resolve(scms)
    out = (df.groupby("product_group", dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  value_usd=("line_value_usd", "sum"),
                  molecules=("molecule", "nunique"),
                  late=("is_late", "sum"))
             .reset_index())
    total = float(out["value_usd"].sum())
    out["value_share_pct"] = (100 * out["value_usd"] / total).round(2)
    out["on_time_pct"] = (100 * (1 - out["late"] / out["line_items"])).round(2)
    out["value_usd"] = out["value_usd"].round(0)
    return out.sort_values("value_usd", ascending=False).reset_index(drop=True)


def top_products(scms: pd.DataFrame | None = None,
                 top_n: int | None = None) -> pd.DataFrame:
    """Highest-value catalogue items, with the price range paid for each.

    Returns
    -------
    pandas.DataFrame
        Columns ``molecule``, ``dosage``, ``dosage_form``, ``line_items``,
        ``value_usd``, ``sites``, ``vendors``, ``median_unit_price_usd``,
        ``min_unit_price_usd``, ``max_unit_price_usd``.
    """
    df = _priced(_resolve(scms))
    top_n = int(_cfg().top_n_products if top_n is None else top_n)

    out = (df.groupby(list(PRODUCT_KEY), dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  value_usd=("line_value_usd", "sum"),
                  sites=("manufacturing_site", "nunique"),
                  vendors=("vendor", "nunique"),
                  median_unit_price_usd=("unit_price_usd", "median"),
                  min_unit_price_usd=("unit_price_usd", "min"),
                  max_unit_price_usd=("unit_price_usd", "max"))
             .reset_index())
    out["value_usd"] = out["value_usd"].round(0)
    for column in ("median_unit_price_usd", "min_unit_price_usd", "max_unit_price_usd"):
        out[column] = out[column].round(4)
    return (out.sort_values("value_usd", ascending=False)
               .head(top_n).reset_index(drop=True))


def value_concentration(scms: pd.DataFrame | None = None,
                        top_n: int | None = None) -> pd.DataFrame:
    """Products ranked by commodity value, with cumulative share.

    The reason the pricing analysis is worth doing at all. Spend is very heavily
    concentrated: the five largest products account for roughly 63% of the $1.63bn,
    and the largest fifteen for about 94%. A buyer therefore does not need to
    renegotiate 92 products - getting five right captures most of the opportunity.

    Returns
    -------
    pandas.DataFrame
        Columns ``product`` (a unique display label), ``molecule``, ``dosage``,
        ``dosage_form``, ``line_items``, ``value_usd``, ``value_share_pct``,
        ``cumulative_share_pct``, ``sites``. ``.attrs`` carries ``top_5_share_pct``,
        ``top_15_share_pct`` and ``products_total``.
    """
    df = _priced(_resolve(scms))
    out = (df.groupby(list(PRODUCT_KEY), dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  value_usd=("line_value_usd", "sum"),
                  sites=("manufacturing_site", "nunique"))
             .reset_index()
             .sort_values("value_usd", ascending=False)
             .reset_index(drop=True))

    total = float(out["value_usd"].sum())
    out["value_share_pct"] = (100 * out["value_usd"] / total).round(2)
    out["cumulative_share_pct"] = out["value_share_pct"].cumsum().round(2)

    # A readable single label for charting, since the product key is three columns.
    #
    # Molecule plus dosage alone is NOT unique: "Abacavir 300mg" exists as both a
    # tablet and a blister-packed tablet, "Efavirenz 200mg" as both a capsule and a
    # tablet, "Zidovudine 10mg/ml" as both an oral solution and an injection - 19
    # rows across 9 such groups. Those are clinically different products, and a
    # shared label would merge them into one bar.
    #
    # So the dosage form is appended only where it is needed to disambiguate. Adding
    # it everywhere would push most labels past 50 characters to fix 9 of 92.
    base = out["molecule"].astype(str) + " " + out["dosage"].astype(str)
    ambiguous = base.duplicated(keep=False)
    out["product"] = base.where(
        ~ambiguous, base + " (" + out["dosage_form"].astype(str) + ")")
    out["value_usd"] = out["value_usd"].round(0)

    attrs = {
        "products_total": int(len(out)),
        "top_5_share_pct": round(float(out["cumulative_share_pct"].iloc[4]), 1)
        if len(out) >= 5 else float("nan"),
        "top_15_share_pct": round(float(out["cumulative_share_pct"].iloc[14]), 1)
        if len(out) >= 15 else float("nan"),
        "total_value_usd": round(total, 0),
    }
    if top_n is not None:
        out = out.head(int(top_n)).copy()
    out.attrs.update(attrs)
    return out[["product", "molecule", "dosage", "dosage_form", "line_items",
                "value_usd", "value_share_pct", "cumulative_share_pct", "sites"]]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def price_spread(scms: pd.DataFrame | None = None,
                 min_shipments: int | None = None,
                 min_sites: int | None = None) -> pd.DataFrame:
    """Price range paid per product, **pooled across all years**.

    This is the misleading version, and it is computed deliberately so the page can
    show it next to the honest one. Pooling ten years of a falling market inflates
    every spread. Use :func:`price_spread_within_year` to quote a figure.

    Returns
    -------
    pandas.DataFrame
        One row per product meeting the thresholds, with ``spread_x`` (max / min
        unit price) and the sample it rests on.
    """
    df = _priced(_resolve(scms))
    min_shipments = int(_cfg().min_shipments_for_spread
                        if min_shipments is None else min_shipments)
    min_sites = int(_cfg().min_sites_for_spread if min_sites is None else min_sites)

    out = (df.groupby(list(PRODUCT_KEY), dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  sites=("manufacturing_site", "nunique"),
                  vendors=("vendor", "nunique"),
                  value_usd=("line_value_usd", "sum"),
                  min_price=("unit_price_usd", "min"),
                  median_price=("unit_price_usd", "median"),
                  max_price=("unit_price_usd", "max"))
             .reset_index())
    out = out[(out["line_items"] >= min_shipments) & (out["sites"] >= min_sites)]
    out["spread_x"] = (out["max_price"] / out["min_price"]).round(1)
    out["value_usd"] = out["value_usd"].round(0)
    for column in ("min_price", "median_price", "max_price"):
        out[column] = out[column].round(4)
    return out.sort_values("value_usd", ascending=False).reset_index(drop=True)


def price_spread_within_year(scms: pd.DataFrame | None = None,
                             min_shipments: int | None = None,
                             min_sites: int | None = None) -> pd.DataFrame:
    """Price range paid per product **within a single delivery year**.

    The figure worth quoting. Holding the year fixed removes the market-wide price
    decline, so what remains is a genuine difference in what was paid for the same
    goods at the same time.

    Returns
    -------
    pandas.DataFrame
        One row per product-year meeting the thresholds, ordered by ``spread_x``.
    """
    df = _priced(_resolve(scms))
    df = df[df["delivery_year"].notna()]
    min_shipments = int(_cfg().min_shipments_for_spread
                        if min_shipments is None else min_shipments)
    min_sites = int(_cfg().min_sites_for_spread if min_sites is None else min_sites)

    keys = list(PRODUCT_KEY) + ["delivery_year"]
    out = (df.groupby(keys, dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  sites=("manufacturing_site", "nunique"),
                  vendors=("vendor", "nunique"),
                  value_usd=("line_value_usd", "sum"),
                  min_price=("unit_price_usd", "min"),
                  median_price=("unit_price_usd", "median"),
                  max_price=("unit_price_usd", "max"))
             .reset_index())
    out = out[(out["line_items"] >= min_shipments) & (out["sites"] >= min_sites)]
    out["delivery_year"] = out["delivery_year"].astype(int)
    out["spread_x"] = (out["max_price"] / out["min_price"]).round(1)
    out["value_usd"] = out["value_usd"].round(0)
    for column in ("min_price", "median_price", "max_price"):
        out[column] = out[column].round(4)
    return out.sort_values("spread_x", ascending=False).reset_index(drop=True)


def price_trend(scms: pd.DataFrame | None = None,
                molecule: str = "Efavirenz",
                dosage: str | None = "600mg") -> pd.DataFrame:
    """Median unit price by delivery year for one product.

    Defaults to Efavirenz 600mg, the clearest example of the decade-long price
    collapse that makes a pooled spread untrustworthy.

    Returns
    -------
    pandas.DataFrame
        Columns ``delivery_year``, ``line_items``, ``sites``, ``min_price``,
        ``median_price``, ``max_price``. ``.attrs`` carries ``decline_pct``.
    """
    df = _priced(_resolve(scms))
    df = df[(df["molecule"] == molecule) & df["delivery_year"].notna()]
    if dosage is not None:
        df = df[df["dosage"] == dosage]

    out = (df.groupby("delivery_year")
             .agg(line_items=("shipment_id", "size"),
                  sites=("manufacturing_site", "nunique"),
                  min_price=("unit_price_usd", "min"),
                  median_price=("unit_price_usd", "median"),
                  max_price=("unit_price_usd", "max"))
             .reset_index())
    if out.empty:
        out.attrs["decline_pct"] = float("nan")
        return out

    out["delivery_year"] = out["delivery_year"].astype(int)
    for column in ("min_price", "median_price", "max_price"):
        out[column] = out[column].round(4)
    first, last = out["median_price"].iloc[0], out["median_price"].iloc[-1]
    out.attrs["decline_pct"] = round(100 * (1 - last / first), 1) if first else 0.0
    out.attrs["product"] = f"{molecule} {dosage or ''}".strip()
    return out


def site_prices(scms: pd.DataFrame | None = None,
                molecule: str = "Nevirapine",
                dosage: str | None = "200mg",
                year: int | None = None) -> pd.DataFrame:
    """What each factory charged for the same product, optionally in one year.

    This is where a price spread becomes actionable: a spread is only worth acting
    on if it maps onto suppliers a buyer can actually choose between.

    Returns
    -------
    pandas.DataFrame
        Columns ``manufacturing_site``, ``line_items``, ``median_unit_price_usd``,
        ``value_usd``, ``on_time_pct``, ordered cheapest first.
    """
    df = _priced(_resolve(scms))
    df = df[df["molecule"] == molecule]
    if dosage is not None:
        df = df[df["dosage"] == dosage]
    if year is not None:
        df = df[df["delivery_year"] == year]

    out = (df.groupby("manufacturing_site", dropna=False)
             .agg(line_items=("shipment_id", "size"),
                  median_unit_price_usd=("unit_price_usd", "median"),
                  value_usd=("line_value_usd", "sum"),
                  late=("is_late", "sum"))
             .reset_index())
    if out.empty:
        return out
    out["on_time_pct"] = (100 * (1 - out["late"] / out["line_items"])).round(2)
    out["median_unit_price_usd"] = out["median_unit_price_usd"].round(4)
    out["value_usd"] = out["value_usd"].round(0)
    return (out.drop(columns=["late"])
               .sort_values("median_unit_price_usd").reset_index(drop=True))


def brand_premium(scms: pd.DataFrame | None = None,
                  min_shipments: int | None = None) -> pd.DataFrame:
    """Generic versus branded unit price for the same product in the same year.

    This is what *explains* the price spread, and it is the reason the spread is
    worth reporting at all. A 2.5x range could be random supplier noise; it is not.
    The ``brand`` column separates ``Generic`` from originator brands, and comparing
    them within a single product-year shows the gap is structural.

    The clearest case is Nevirapine 200mg: nine Indian factories supplied it as a
    generic at $0.04-$0.06 per unit, while the originator's product (``Viramune``,
    Boehringer Ingelheim) was bought at $0.30-$0.39 - roughly seven times more for
    the same molecule at the same strength in the same year.

    Only product-years where **both** a generic and a branded version were bought are
    included, so this is a like-for-like comparison rather than a mix effect.

    Returns
    -------
    pandas.DataFrame
        Columns ``molecule``, ``dosage``, ``delivery_year``, ``label`` (unique per
        row, for charting), ``line_items``, ``generic_price``, ``branded_price``,
        ``premium_x``, ``branded_brand``.
        ``.attrs`` carries ``median_premium_x`` and ``product_years``.
    """
    df = _priced(_resolve(scms))
    df = df[df["delivery_year"].notna()].copy()
    min_shipments = int(_cfg().min_shipments_for_spread
                        if min_shipments is None else min_shipments)
    df["is_generic"] = df["brand"].astype("string").eq("Generic")

    rows = []
    for (molecule, dosage, year), group in df.groupby(
            ["molecule", "dosage", "delivery_year"], dropna=False):
        # Both arms must be present, otherwise this is not a comparison.
        if group["is_generic"].nunique() != 2 or len(group) < min_shipments:
            continue
        generic = float(group.loc[group["is_generic"], "unit_price_usd"].median())
        branded = float(group.loc[~group["is_generic"], "unit_price_usd"].median())
        if not generic:
            continue
        rows.append({
            "molecule": molecule, "dosage": dosage, "delivery_year": int(year),
            # One label per row. Charting on `molecule` alone silently stacked four
            # Nevirapine product-years into a single bar and rendered their premiums
            # as a sum - 25x, when the true maximum is 7.6x. A price multiple is not
            # an additive quantity, so the plotted category has to be unique.
            "label": f"{molecule} {dosage} ({int(year)})",
            "line_items": int(len(group)),
            "generic_price": round(generic, 4),
            "branded_price": round(branded, 4),
            "premium_x": round(branded / generic, 1),
            "branded_brand": ", ".join(sorted(
                group.loc[~group["is_generic"], "brand"].dropna().unique())[:2]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        out.attrs.update(median_premium_x=float("nan"), product_years=0)
        return out
    out = out.sort_values("premium_x", ascending=False).reset_index(drop=True)
    out.attrs["median_premium_x"] = round(float(out["premium_x"].median()), 1)
    out.attrs["product_years"] = int(len(out))
    log.info("Brand premium: median %.1fx over %d product-years",
             out.attrs["median_premium_x"], len(out))
    return out


def pricing_summary(scms: pd.DataFrame | None = None) -> dict:
    """Pooled versus within-year price spread, and what to quote.

    Returns
    -------
    dict
        Both medians, the counts behind them, the widest single case, and a plain
        English ``verdict`` naming which figure is safe to report.
    """
    df = _resolve(scms)
    pooled = price_spread(df)
    within = price_spread_within_year(df)
    premium = brand_premium(df)

    pooled_median = float(pooled["spread_x"].median()) if len(pooled) else float("nan")
    within_median = float(within["spread_x"].median()) if len(within) else float("nan")
    worst = within.iloc[0] if len(within) else None
    trend = price_trend(df)
    decline = trend.attrs.get("decline_pct", float("nan"))

    premium_median = premium.attrs.get("median_premium_x", float("nan"))
    verdict = (
        f"Pooled across all years the median spread is {pooled_median:.1f}x, but that "
        f"mixes supplier differences with a market-wide price collapse "
        f"({trend.attrs.get('product', 'Efavirenz 600mg')} fell {decline:.0f}% over "
        f"the decade). Within a single year it is {within_median:.1f}x, measured on "
        f"{len(within)} product-years - quote that one. The remaining gap is not "
        f"noise: branded product costs a median of {premium_median:.1f}x more than "
        f"the same molecule bought generically in the same year."
    )
    log.info("Pricing: pooled %.1fx vs within-year %.1fx over %d product-years",
             pooled_median, within_median, len(within))

    return {
        "pooled_median_spread_x": round(pooled_median, 1),
        "pooled_products": int(len(pooled)),
        "within_year_median_spread_x": round(within_median, 1),
        "within_year_products": int(len(within)),
        "widest_case": (
            f"{worst['molecule']} {worst['dosage']} in {worst['delivery_year']}: "
            f"${worst['min_price']:.2f} to ${worst['max_price']:.2f} "
            f"({worst['spread_x']:.1f}x across {worst['sites']} factories)"
            if worst is not None else "none"),
        "widest_spread_x": float(worst["spread_x"]) if worst is not None else float("nan"),
        "reference_decline_pct": decline,
        "brand_premium_x": premium_median,
        "brand_premium_product_years": premium.attrs.get("product_years", 0),
        "inflation_factor": (round(pooled_median / within_median, 2)
                             if within_median else float("nan")),
        "verdict": verdict,
    }


__all__ = [
    "PRODUCT_KEY", "catalogue_kpis", "product_mix", "top_products",
    "value_concentration",
    "price_spread", "price_spread_within_year", "price_trend", "site_prices",
    "brand_premium", "pricing_summary",
]
