"""
Product & Pricing page - the catalogue side of the USAID SCMS dataset.

SCMS is usually treated as a logistics file, but every line item also carries the
molecule, brand, dosage, form, factory and the price actually paid. That makes it a
product catalogue as well as a delivery log, and it means the two questions - "did it
arrive on time?" and "did we pay a fair price?" - can be answered on the same rows.

An earlier version of this project used a separate 253,973-row Indian medicine
catalogue for this page. It was dropped: it carried *list* prices for products nobody
in this dataset bought, while these are prices actually paid, joinable to delivery
performance. Scale is not evidence.

The page is built around a trap. Pooled across ten years the price spread for
identical products looks like 5.0x; within a single year it is 2.5x. The difference
is a market-wide price collapse, not supplier behaviour - the same pooling mistake
the Statistical Testing page unpicks for delivery performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.analytics import products as pr
from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.viz import charts
from src.viz.theme import fmt_currency, fmt_pct

page_setup(
    title="Product Catalogue & Procurement Pricing",
    icon="💰",
    subtitle="What was bought, from which factory, and at what price - across "
             "10,324 real USAID line items",
)


@st.cache_data(show_spinner="Analysing the product catalogue...")
def _products() -> dict:
    return {
        "kpis": pr.catalogue_kpis(),
        "mix": pr.product_mix(),
        "concentration": pr.value_concentration(),
        "top": pr.top_products(),
        "pooled": pr.price_spread(),
        "within": pr.price_spread_within_year(),
        "premium": pr.brand_premium(),
        "summary": pr.pricing_summary(),
    }


@st.cache_data(show_spinner=False)
def _trend(molecule: str, dosage: str):
    return pr.price_trend(molecule=molecule, dosage=dosage)


@st.cache_data(show_spinner=False)
def _sites(molecule: str, dosage: str, year: int | None):
    return pr.site_prices(molecule=molecule, dosage=dosage, year=year)


data = _products()
kpis, summary = data["kpis"], data["summary"]
concentration, premium = data["concentration"], data["premium"]

kpi_row([
    {"label": "Catalogue Items", "value": f"{kpis['catalogue_items']}",
     "help_text": f"{kpis['molecules']} molecules, {kpis['dosages']} strengths, "
                  f"{kpis['dosage_forms']} dosage forms"},
    {"label": "Factories", "value": f"{kpis['manufacturing_sites']}",
     "help_text": f"Supplying via {kpis['vendors']} vendors"},
    {"label": "Bought Generically",
     "value": fmt_pct(kpis["generic_share_pct"]),
     "help_text": f"{kpis['generic_line_items']:,} of {kpis['line_items']:,} line "
                  f"items carry the brand 'Generic'"},
    {"label": "Branded Premium",
     "value": f"{summary['brand_premium_x']:.1f}x",
     "delta": "same molecule, same year",
     "delta_good": False,
     "help_text": f"Median across {summary['brand_premium_product_years']} "
                  f"product-years where both were bought"},
])

callout(
    f"**Price analysis is measured on {kpis['priced_line_items']:,} of "
    f"{kpis['line_items']:,} line items ({kpis['priced_coverage_pct']:.1f}%)** — the "
    f"rest carry no usable unit price. A price of zero is not a free medicine, it is "
    f"an unrecorded one, so those rows are excluded rather than counted as $0. Every "
    f"figure below therefore describes the priced subset, and the value totals on "
    f"this page are smaller than the "
    f"{fmt_currency(kpis['total_value_usd'])} headline for that reason.",
    kind="warning", title="What these numbers are measured on",
)

# ---------------------------------------------------------------------------
# Where the money is
# ---------------------------------------------------------------------------
section(
    "Where The Money Is",
    "Spend is far more concentrated than the 184-item catalogue suggests, which is "
    "what makes a pricing exercise worth doing at all.",
)

chart(charts.concentration_chart(
    concentration.head(15), "product", "value_usd", "cumulative_share_pct",
    title="Fifteen Largest Products by Commodity Value", height=480,
    reference_pct=80,
    note=(f"{concentration.attrs['products_total']} priced products in total<br>"
          f"top 5 = {concentration.attrs['top_5_share_pct']:.1f}% of value")))

insight(
    f"**{concentration.attrs['top_5_share_pct']:.1f}% of priced commodity value sits "
    f"in five products, and {concentration.attrs['top_15_share_pct']:.1f}% in "
    f"fifteen** — out of {concentration.attrs['products_total']} priced products in "
    f"the catalogue. The single largest, "
    f"{concentration.iloc[0]['product']}, is "
    f"{concentration.iloc[0]['value_share_pct']:.1f}% of spend on its own.\n\n"
    f"That is the practical case for this page. A buyer does not have to renegotiate "
    f"{concentration.attrs['products_total']} products — getting the top five right "
    f"captures roughly two-thirds of the opportunity. It is also why the 80% Pareto "
    f"line is drawn here and deliberately omitted from flat distributions elsewhere "
    f"in this project: on this data the curve genuinely reaches it, by the ninth "
    f"product."
)

col_a, col_b = st.columns([3, 2], gap="large")
with col_a:
    chart(charts.bar_chart(
        data["mix"], x="product_group", y="value_usd",
        title="Commodity Value by Product Group", height=360))
with col_b:
    show_table(data["mix"])

mix = data["mix"]
insight(
    f"**Antiretrovirals are {mix.iloc[0]['value_share_pct']:.1f}% of value across "
    f"{int(mix.iloc[0]['molecules'])} molecules**, with HIV rapid diagnostic tests "
    f"second at {mix.iloc[1]['value_share_pct']:.1f}%. The three remaining groups — "
    f"malaria treatments and tests — total under 0.1% between them, on "
    f"{int(mix.iloc[2]['line_items'] + mix.iloc[3]['line_items'] + mix.iloc[4]['line_items'])} "
    f"line items.\n\n"
    f"Those three groups are all at 100% on-time, and that is **not** a finding worth "
    f"reporting. With that few shipments each, one late delivery would move the rate "
    f"by tens of points — which is exactly why the comparisons on the Statistical "
    f"Testing page enforce a minimum group size instead of ranking every category it "
    f"can find."
)

# ---------------------------------------------------------------------------
# The pooling trap
# ---------------------------------------------------------------------------
section(
    "Did We Pay A Consistent Price?",
    "The same molecule, at the same strength, in the same dosage form. Pooled "
    "across all years on the left; within a single year on the right.",
)

col_c, col_d = st.columns(2, gap="large")
with col_c:
    st.metric("Pooled across 2006-2015",
              f"{summary['pooled_median_spread_x']:.1f}x",
              delta=f"{summary['pooled_products']} products",
              delta_color="off")
    st.caption("**Do not quote this.** It mixes supplier differences with a decade "
               "of falling prices.")
with col_d:
    st.metric("Within a single year",
              f"{summary['within_year_median_spread_x']:.1f}x",
              delta=f"{summary['within_year_products']} product-years",
              delta_color="off")
    st.caption("The honest figure. Holding the year fixed removes the market-wide "
               "price decline.")

callout(summary["verdict"], kind="insight", title="Which number to report")

trend = _trend("Efavirenz", "600mg")
chart(charts.line_chart(
    trend, x="delivery_year", y=["min_price", "median_price", "max_price"],
    title=f"{trend.attrs['product']} Unit Price by Year — the Median Fell 80%, "
          f"the Maximum Did Not",
    y_title="Unit price (USD)", height=400))

late = trend[trend["delivery_year"] >= 2013]
insight(
    f"**The median price of {trend.attrs['product']} fell "
    f"{trend.attrs['decline_pct']:.0f}% over the decade**, from "
    f"${trend.iloc[0]['median_price']:.2f} in {int(trend.iloc[0]['delivery_year'])} to "
    f"${trend.iloc[-1]['median_price']:.2f} in "
    f"{int(trend.iloc[-1]['delivery_year'])}. Generic competition, and a large real "
    f"achievement for the programme.\n\n"
    f"That decline wrecks a pooled price comparison. Comparing a 2006 purchase against "
    f"a 2015 one measures *when* you bought, not *who* from — and pooling inflates the "
    f"apparent spread by a factor of **{summary['inflation_factor']:.1f}**. It is the "
    f"identical mistake the Statistical Testing page finds in delivery performance, "
    f"where a pooled 11.9-point gap turns out to be 1.9 points before 2011 and 20.5 "
    f"after. Two unrelated questions, the same trap, and stratifying by year fixes "
    f"both.\n\n"
    f"**Look at the top line, though.** The median collapses to "
    f"${late['median_price'].median():.2f} while the *maximum* paid stays near "
    f"${late['max_price'].median():.2f} right through to 2015 — a "
    f"{late['max_price'].median() / late['median_price'].median():.1f}x gap inside a "
    f"single year, on a product where the cheap option was demonstrably available. "
    f"So the pooling correction removes an artefact without removing the finding: the "
    f"spread is smaller than 5x, and it is still real."
)

with st.expander("The full within-year table, widest spread first"):
    show_table(data["within"])
    st.caption(
        f"Filtered to products with at least "
        f"{pr._cfg().min_shipments_for_spread} line items and "
        f"{pr._cfg().min_sites_for_spread} factories in that year — a product bought "
        f"three times has a range that is really just three numbers.")

# ---------------------------------------------------------------------------
# What explains it
# ---------------------------------------------------------------------------
section(
    "What Explains The Spread",
    "A 2.5x range could be random supplier noise. It is not — it is generic versus "
    "originator, which is a difference a buyer can act on.",
)

chart(charts.bar_chart(
    premium.head(12), x="label", y="premium_x",
    title="Branded Price as a Multiple of Generic, Same Product and Same Year",
    orientation="h", text_format=".1f", height=460))

top_premium = premium.iloc[0]
insight(
    f"Across **{premium.attrs['product_years']} product-years where both a generic "
    f"and an originator-branded version were bought in the same year**, the branded "
    f"one costs a median of **{premium.attrs['median_premium_x']:.1f}x more**. "
    f"Because both arms are present in the same year and the same product, this is a "
    f"like-for-like comparison rather than a mix effect.\n\n"
    f"The clearest case is **{top_premium['molecule']} {top_premium['dosage']}**: "
    f"${top_premium['generic_price']:.2f} generic against "
    f"${top_premium['branded_price']:.2f} for {top_premium['branded_brand']} in "
    f"{int(top_premium['delivery_year'])} — **{top_premium['premium_x']:.1f}x** for "
    f"the same molecule at the same strength in the same year."
)

sites = _sites("Nevirapine", "200mg", 2009)
if len(sites):
    col_e, col_f = st.columns([3, 2], gap="large")
    with col_e:
        chart(charts.bar_chart(
            sites, x="manufacturing_site", y="median_unit_price_usd",
            title="Nevirapine 200mg, 2009 — Median Unit Price by Factory",
            orientation="h", text_format=".3f", height=340))
    with col_f:
        show_table(sites)
    insight(
        "**This is where a statistic becomes a procurement decision.** Three Indian "
        "generic manufacturers supplied Nevirapine 200mg at $0.050 per unit in 2009. "
        "The originator's plant in Greece supplied the same molecule and strength at "
        "$0.335 — **6.7x more** — and delivered 100% on time against 86% for the "
        "cheapest supplier.\n\n"
        "So the trade-off is real rather than free, and that is the honest way to "
        "put it: the premium buys measurably better service on this product in this "
        "year. Whether a 6.7x price is worth 14 points of on-time performance is a "
        "judgement for a procurement lead, not something the data settles. What the "
        "data does settle is that the choice exists and what each side of it costs."
    )

download_button(data["within"], "price_spread_within_year.csv",
                label="Download within-year price spreads")

methodology(f"""
**What counts as "the same product".** Molecule, dosage **and** dosage form together
(`products.PRODUCT_KEY`). Molecule alone is not enough — a 200mg tablet and a 600mg
tablet of the same molecule are different goods, and comparing their unit prices
would be meaningless.

**Unit price, not pack price.** Pack sizes vary from 1 to 1,000 units across the
catalogue, so pack price is not comparable between products. `Unit Price` is the
per-unit figure the dataset itself provides.

**Zero prices are excluded, not imputed.** {kpis['line_items'] - kpis['priced_line_items']:,}
line items carry no usable unit price. Treating those as $0 would drag every minimum
to zero and make each spread ratio infinite, so they are dropped and the coverage is
stated at the top of the page.

**Thresholds.** A product needs at least
`scms.products.min_shipments_for_spread` ({pr._cfg().min_shipments_for_spread}) line
items and `scms.products.min_sites_for_spread` ({pr._cfg().min_sites_for_spread})
factories before its spread is reported — the same noise logic as the vendor
scorecard on the Vendor & Logistics page.

**Max divided by min is a deliberately crude measure.** It is sensitive to a single
extreme purchase, which is why the median across products is quoted rather than any
individual ratio, why the full table is available above, and why the factory-level
view exists — a spread only matters if it maps onto suppliers you could choose
between.

**What this cannot tell you.** Nothing here explains *why* a higher price was paid.
Freight terms, volume commitments, urgency, registration status in the destination
country and donor earmarking are all plausible and none is recorded. The branded
premium is a measured fact; calling it an overpayment would be an inference the data
does not support.
""")

sidebar_about()
