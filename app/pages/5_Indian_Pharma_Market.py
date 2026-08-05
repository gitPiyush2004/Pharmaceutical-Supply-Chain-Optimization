"""
Indian Pharma Market page - product master of 253,973 marketed medicines.

Why this page is descriptive rather than predictive
--------------------------------------------------
The obvious move with a 253,973-row dataset carrying an `Is_discontinued` flag is
to train a third classifier on it. That would have been the wrong call, and the
reason is worth stating rather than hiding behind a model that happens to run:

* The `type` column has exactly one value (`allopathy`) across all 253,973 rows,
  so it carries no information.
* The target is 31:1 imbalanced (7,905 discontinued, 3.11%).
* The file has no launch date, no sales volume and no therapeutic class - the
  features that would genuinely predict a withdrawal decision are simply absent.

A model trained on brand name, price and manufacturer would have learned which
manufacturers happen to have refreshed their catalogue, dressed up as a
withdrawal-risk score. So this page reports market structure, which the data
supports, and states plainly where the discontinuation signal actually comes from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not put the repository root on sys.path, and neither does
# pytest or Streamlit's AppTest harness. Doing it here - before any first-party
# import - means this page runs identically under all three.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.analytics import market as mk
from src.dashboard.components import (callout, chart, download_button, insight,
                                      kpi_row, methodology, page_setup, section,
                                      show_table, sidebar_about)
from src.data.indian_medicines import provenance
from src.viz import charts

page_setup(
    title="Indian Pharma Market Structure",
    icon="🇮🇳",
    subtitle="253,973 marketed medicines, 7,642 manufacturers, 1,585 active ingredients",
)


@st.cache_data(show_spinner="Loading Indian medicine product master...")
def _market() -> dict:
    return {
        "kpis": mk.market_kpis(),
        "concentration": mk.manufacturer_concentration(),
        "summary": mk.concentration_summary(),
        "prices": mk.price_distribution(),
        "discontinuation": mk.discontinuation_analysis(),
        "ingredients": mk.ingredient_analysis(),
        "packs": mk.pack_form_analysis(),
        "provenance": provenance(),
    }


data = _market()
kpis, conc, summary = data["kpis"], data["concentration"], data["summary"]
disc = data["discontinuation"]
prov = data["provenance"]

callout(
    f"**{prov['name']}** — {prov['description']} {prov['records']:,} products, "
    f"{prov['coverage']}. Source: {prov['source_url']}. Every figure on this page is "
    "counted from these records.",
    kind="insight", title="Data provenance",
)

kpi_row([
    {"label": "Products Listed", "value": f"{kpis['products']:,}",
     "help_text": f"Across {kpis['distinct_ingredients']:,} distinct active ingredients"},
    {"label": "Manufacturers", "value": f"{kpis['manufacturers']:,}",
     "help_text": f"Largest is {kpis['top_manufacturer']} with "
                  f"{kpis['top_manufacturer_products']:,} products"},
    {"label": "Median Price", "value": f"Rs {kpis['median_price_inr']:,.0f}",
     "help_text": f"P90 Rs {kpis['p90_price_inr']:,.0f}, "
                  f"maximum Rs {kpis['max_price_inr']:,.0f}"},
    {"label": "Combination Products",
     "value": f"{kpis['combination_pct']:.1f}%",
     "help_text": "Two or more active ingredients in the composition"},
])

# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
section(
    "Market Concentration",
    "How many manufacturers it takes to cover the catalogue.",
)

chart(charts.concentration_chart(
    conc.head(20), "manufacturer", "products", "cumulative_share_pct",
    title="Twenty Largest Manufacturers by Products Listed",
    hhi=summary.attrs["hhi"], height=460))

col_a, col_b = st.columns([2, 3], gap="large")
with col_a:
    show_table(summary)
with col_b:
    insight(
        f"**This market has no incumbent.** The Herfindahl index is "
        f"**{summary.attrs['hhi']:,.0f}** — anything below 1,500 counts as "
        f"fragmented, and the competitive-concern threshold used by regulators is "
        f"2,500. The largest manufacturer holds 1.2% of listings, the top ten hold "
        f"{summary.loc[0, 'cumulative_share_pct']:.1f}% between them, and it takes "
        f"500 firms to reach {summary.loc[4, 'cumulative_share_pct']:.1f}%.\n\n"
        f"The sourcing consequence is direct: with {kpis['manufacturers']:,} suppliers "
        f"and 44% of products being combinations of the same 1,585 molecules, "
        f"negotiating leverage comes from the credible ability to *switch* supplier, "
        f"not from volume commitments to any one of them."
    )

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
section(
    "Price Distribution",
    "Log-spaced buckets, because a linear price axis on this data puts 99% of "
    "products in the first bin and tells you nothing.",
)

chart(charts.bar_chart(
    data["prices"], x="price_bucket", y="products",
    title="Products by Price Bucket (log-spaced)", height=420))

insight(
    f"**Pricing is tightly clustered and then has a very long tail.** "
    f"{data['prices'].loc[3, 'share_pct'] + data['prices'].loc[4, 'share_pct']:.0f}% "
    f"of products fall between Rs 27 and Rs 230, against a median of "
    f"Rs {kpis['median_price_inr']:,.0f}. Above that, prices run to "
    f"Rs {kpis['max_price_inr']:,.0f} — a four-order-of-magnitude range across only "
    f"a few hundred products, which is what a speciality and biologics tail looks "
    f"like sitting on top of a high-volume generics market.\n\n"
    f"Only {kpis['unrecorded_prices']} of {kpis['products']:,} products have no "
    f"usable price, so this is one of the cleanest fields in any dataset in this "
    f"project."
)

# ---------------------------------------------------------------------------
# Discontinuation - reported, not modelled
# ---------------------------------------------------------------------------
section(
    "Discontinuation Rate",
    "Reported descriptively. The reason it is not modelled is below the charts.",
)

col_c, col_d = st.columns(2, gap="large")
with col_c:
    chart(charts.bar_chart(
        disc["by_price_band"], x="price_band", y="discontinued_pct",
        title="Discontinued by Price Quartile", height=360))
with col_d:
    chart(charts.bar_chart(
        disc["by_manufacturer"].head(12), x="manufacturer", y="discontinued_pct",
        title="Highest Discontinuation Rate (>=50 products)",
        orientation="h", height=360))

callout(disc["caveat"], kind="warning",
        title="Why discontinuation is reported and not predicted")

insight(
    f"The spread tells the story. Discontinuation varies by "
    f"**{disc['manufacturer_spread_pp']:.0f} percentage points between "
    f"manufacturers** but only **{disc['price_band_spread_pp']:.1f} points across "
    f"price quartiles**. A manufacturer effect that large with essentially no price "
    f"effect is not product economics — it is catalogue refresh timing in the source "
    f"data. Some firms have pruned their listings and some have not.\n\n"
    f"So the honest reading is that this flag describes the *listing*, not the "
    f"market. Modelling it would have produced a confident-looking risk score for "
    f"something the data cannot see."
)

# ---------------------------------------------------------------------------
# Ingredients and packs
# ---------------------------------------------------------------------------
section(
    "Active Ingredients and Pack Forms",
    "Which molecules are most crowded, and how product is physically presented.",
)

tab_ingredients, tab_packs = st.tabs(["Active ingredients", "Pack forms"])

with tab_ingredients:
    top = data["ingredients"].head(15)
    chart(charts.bar_chart(
        top, x="ingredient", y="brands",
        title="Most Crowded Active Ingredients by Brand Count",
        orientation="h", height=460))
    show_table(data["ingredients"].head(25))
    leader = data["ingredients"].iloc[0]
    insight(
        f"**{leader['ingredient']} alone is sold under {leader['brands']:,} brand "
        f"names by {leader['manufacturers']:,} manufacturers**, at a median of "
        f"Rs {leader['median_price_inr']:,.0f} and a spread of "
        f"Rs {leader['price_spread_inr']:,.0f} between the cheapest and dearest "
        f"quartile. That price dispersion on a single off-patent molecule is the "
        f"clearest procurement opportunity in this dataset: for a formulary buyer, "
        f"identical therapeutic content is available across a wide price range."
    )

with tab_packs:
    chart(charts.bar_chart(
        data["packs"].head(10), x="pack_form", y="products",
        title="Products by Pack Form", orientation="h", height=420))
    show_table(data["packs"])
    insight(
        f"**Strips dominate at {data['packs'].loc[0, 'share_pct']:.0f}% of "
        f"listings**, which is what an oral-solid generics market looks like. The "
        f"interesting rows are the small ones: prefilled syringes carry a median "
        f"price of Rs {data['packs'].loc[7, 'median_price_inr']:,.0f} against "
        f"Rs {data['packs'].loc[0, 'median_price_inr']:,.0f} for strips, and the "
        f"highest discontinuation rates sit on those same low-volume injectable "
        f"forms — consistent with cold-chain and handling cost making marginal "
        f"presentations the first to be rationalised."
    )

download_button(conc, "manufacturer_concentration.csv",
                label="Download manufacturer table")

methodology("""
**Manufacturer names.** Corporate suffixes (`Ltd`, `Limited`, `Pvt Ltd`) are
stripped and casing normalised before counting. This is honest about its own
impact: it merges only 6 of 7,648 raw names. The fragmentation in this market is
real, not an artefact of inconsistent spelling.

**Prices.** `price(Rs)` is parsed to a numeric `price_inr`. Four products carry no
usable price and are excluded from price statistics rather than imputed — with a
median of Rs 79 and a maximum of Rs 436,000, any imputed value would be arbitrary.

**Pack labels.** `pack_size_label` is free text (`"strip of 10 tablets"`), parsed
into a `pack_form` and a numeric `pack_quantity`. Forms below 50 products are
grouped out of the ranked tables under `indian_medicines.min_products_for_ranking`.

**The constant column.** `type` is `allopathy` for all 253,973 rows and is dropped
at load. A column with one value is not a feature.
""")

sidebar_about()
