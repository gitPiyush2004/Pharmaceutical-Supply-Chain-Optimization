"""
Indian Medicine Dataset - real Indian pharmaceutical product master data.

Source
------
253,973 medicines marketed in India, published as open data:
https://github.com/junioralive/Indian-Medicine-Dataset (mirrored on Kaggle as
*A-Z Medicine Dataset of India*). Fetched once and cached under ``data/external``
so every build is reproducible offline.

Columns as published: ``id``, ``name``, ``price(₹)``, ``Is_discontinued``,
``manufacturer_name``, ``type``, ``pack_size_label``, ``short_composition1``,
``short_composition2``.

What this dataset is good for, and what it is not
------------------------------------------------
It is a **product master**: what is sold, by whom, at what price, in what pack.
That supports genuine market-structure analysis - manufacturer concentration,
price distributions, portfolio breadth, discontinuation patterns.

It is **not** a modelling dataset, and the platform deliberately does not train on
it. Two measured reasons:

* ``type`` has exactly **one value** (``allopathy``) across all 253,973 rows, so it
  carries no information at all.
* ``Is_discontinued`` is **31:1 imbalanced** (7,905 True, 3.11%), and the file has
  no launch date, no sales volume and no therapeutic class - the variables that
  would actually explain why a product was withdrawn. A classifier here would be
  fitting price and pack size to an outcome they do not drive.

Reporting discontinuation descriptively is honest; predicting it from these columns
would not be.

Real data quality characteristics
---------------------------------
Unusually clean for real data, and the cleaning here is normalisation rather than
repair:

* **No nulls** except ``short_composition2`` (55.8% empty), which is legitimate -
  most medicines have a single active ingredient.
* **No duplicate rows**, though 4,575 names repeat because the same brand is sold
  in several pack sizes.
* ``price(₹)`` is fully numeric: median ₹79, maximum ₹436,000, and **4 zero
  prices** which are flagged rather than dropped.
* ``manufacturer_name`` has **7,648 distinct values**. Consolidating suffix and
  spacing variants (``Ltd`` / ``Limited`` / ``Pvt Ltd``) merges only **6** of them,
  so the published column was already close to consistent. The normalisation is
  kept because it is cheap and makes the concentration analysis robust to a messier
  refresh of the source, but it is not doing heavy lifting here - measured, not
  assumed.

What the discontinuation flag actually tracks
---------------------------------------------
Worth knowing before anyone is tempted to model it. Across the whole file 3.11% of
products are discontinued, and that rate is **flat across price quartiles**
(3.7 / 2.6 / 2.8 / 3.3%) - price does not explain withdrawal.

By manufacturer it is anything but flat: Glenmark Pharmaceuticals sits at **41.7%**
and Abbott at **36.4%** against a 3.11% base rate. A gap that large is unlikely to
be pure commercial strategy; it more plausibly reflects how recently each
manufacturer's catalogue was refreshed in the source. Treat the flag as a property
of the *listing* as much as of the product, and report it descriptively rather than
predicting it.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

#: Public source, from config. Raw GitHub needs no authentication, unlike the
#: Kaggle mirror of the same file.
SOURCE_URL: str = str(get_config().indian_medicines.source_url)

#: Corporate suffixes stripped before comparing manufacturer names. Order matters:
#: longer forms first, so "Private Limited" is not left as "Private" by an earlier
#: rule removing "Limited".
_SUFFIXES = [
    "private limited", "pvt limited", "pvt ltd", "private ltd",
    "limited", "ltd", "llp", "inc", "incorporated", "corporation", "corp",
    "company", "co", "plc", "gmbh", "sa", "srl", "bv", "ag",
]

#: Pack labels are free text ("strip of 10 tablets", "bottle of 100 ml Syrup").
#: This pulls out the container and the numeric quantity.
_PACK_QUANTITY = re.compile(r"(\d+(?:\.\d+)?)")
_PACK_FORMS = [
    "strip", "bottle", "packet", "tube", "vial", "ampoule", "sachet", "jar",
    "box", "tin", "prefilled syringe", "syringe", "cartridge", "pen", "inhaler",
    "respule", "transdermal patch", "patch", "kit", "unit", "carton", "capsule",
]

#: Rows priced at zero are almost certainly unrecorded rather than free. Flagged,
#: not dropped, because dropping them would silently change every count.
ZERO_PRICE_IS_MISSING = True


def dataset_path():
    """Absolute path of the cached CSV inside ``data/external``."""
    return resolve_path(get_config().paths.data_external) / "indian_medicine_data.csv"


def download(force: bool = False):
    """Fetch the CSV from the public mirror if it is not already cached.

    Parameters
    ----------
    force
        Re-download even when the cached file exists.

    Returns
    -------
    pathlib.Path
        Location of the cached CSV.
    """
    path = dataset_path()
    if path.exists() and not force:
        return path

    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Indian Medicine Dataset from %s", SOURCE_URL)
    urllib.request.urlretrieve(SOURCE_URL, path)
    log.info("Cached to %s (%.1f MB)", path, path.stat().st_size / 1_048_576)
    return path


def normalise_manufacturer(name: str) -> str:
    """Collapse corporate-suffix and spacing variants of one company name.

    ``Sun Pharmaceutical Industries Ltd``, ``SUN PHARMACEUTICAL INDUSTRIES
    LIMITED`` and ``Sun Pharmaceutical Industries Pvt. Ltd.`` are one firm. Without
    this, concentration analysis splits a company across several rows and
    understates how much of the market the largest players hold.

    Returns
    -------
    str
        Title-cased name with punctuation and corporate suffixes removed.
    """
    if not isinstance(name, str) or not name.strip():
        return "Unknown"

    cleaned = re.sub(r"[.,()]", " ", name.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip suffixes repeatedly - "Pharma Pvt Ltd" carries two.
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if cleaned.endswith(" " + suffix):
                cleaned = cleaned[: -len(suffix) - 1].strip()
                changed = True
    return cleaned.title() if cleaned else "Unknown"


def _parse_pack(label: str) -> tuple[str, float]:
    """Split a free-text pack label into (container form, numeric quantity)."""
    if not isinstance(label, str) or not label.strip():
        return "Unknown", np.nan

    lowered = label.lower()
    form = next((f for f in _PACK_FORMS if f in lowered), "Other")
    match = _PACK_QUANTITY.search(lowered)
    quantity = float(match.group(1)) if match else np.nan
    return form.title(), quantity


def _split_composition(first: str, second: str) -> tuple[list[str], int]:
    """Return the active ingredients and how many there are.

    ``short_composition2`` is empty for 55.8% of rows, which is not missing data -
    most medicines are single-ingredient.
    """
    ingredients: list[str] = []
    for value in (first, second):
        if not isinstance(value, str) or not value.strip():
            continue
        # Strip the parenthesised strength: "Amoxycillin (500mg)" -> "Amoxycillin".
        for part in value.split(","):
            token = re.sub(r"\(.*?\)", "", part).strip()
            if token:
                ingredients.append(token.title())
    return ingredients, len(ingredients)


def load_raw() -> pd.DataFrame:
    """Load the CSV exactly as published, for data quality profiling."""
    frame = pd.read_csv(download(), low_memory=False)
    log.debug("Loaded raw Indian medicines (%d rows x %d cols)", *frame.shape)
    return frame


@lru_cache(maxsize=1)
def load_indian_medicines() -> pd.DataFrame:
    """Load and normalise the Indian medicine master into analysis-ready form.

    Normalisation applied, in order:

    1. Rename ``price(₹)`` to ``price_inr`` and coerce to float.
    2. Flag zero prices as unrecorded rather than free.
    3. Consolidate manufacturer name variants (see :func:`normalise_manufacturer`).
    4. Parse ``pack_size_label`` into ``pack_form`` and ``pack_quantity``.
    5. Split composition into an ingredient list and a count.
    6. Drop the constant ``type`` column, which carries no information.
    7. Band price into quartile labels for descriptive comparison.

    Returns
    -------
    pandas.DataFrame
        One row per marketed medicine.
    """
    raw = load_raw()
    df = raw.rename(columns={"price(₹)": "price_inr",
                             "Is_discontinued": "is_discontinued"}).copy()

    # --- Price -------------------------------------------------------------
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df["price_is_recorded"] = (df["price_inr"] > 0) if ZERO_PRICE_IS_MISSING \
        else df["price_inr"].notna()

    # --- Discontinuation flag ----------------------------------------------
    df["is_discontinued"] = df["is_discontinued"].astype(bool)

    # --- Manufacturer consolidation ----------------------------------------
    df["manufacturer_raw"] = df["manufacturer_name"].astype(str)
    df["manufacturer"] = df["manufacturer_raw"].map(normalise_manufacturer)

    # --- Pack parsing -------------------------------------------------------
    parsed_packs = df["pack_size_label"].map(_parse_pack)
    df["pack_form"] = [p[0] for p in parsed_packs]
    df["pack_quantity"] = [p[1] for p in parsed_packs]

    # --- Composition --------------------------------------------------------
    compositions = [
        _split_composition(a, b)
        for a, b in zip(df["short_composition1"], df["short_composition2"])
    ]
    df["ingredients"] = [c[0] for c in compositions]
    df["ingredient_count"] = [c[1] for c in compositions]
    df["is_combination"] = df["ingredient_count"] > 1
    df["primary_ingredient"] = [
        c[0][0] if c[0] else "Unknown" for c in compositions]

    # --- Price bands --------------------------------------------------------
    # Quartiles over recorded prices only, so the 4 zero-price rows do not drag
    # the lowest band's boundary to zero.
    priced = df.loc[df["price_is_recorded"], "price_inr"]
    edges = [0, *priced.quantile([0.25, 0.5, 0.75]).tolist(), float("inf")]
    df["price_band"] = pd.cut(
        df["price_inr"], bins=edges, include_lowest=True,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    # `type` is constant across every row, so it cannot inform any analysis.
    df = df.drop(columns=["type"], errors="ignore")

    log.info(
        "Loaded Indian medicines: %d products | %d manufacturers "
        "(%d before consolidation) | %.2f%% discontinued | median Rs %.0f",
        len(df), df["manufacturer"].nunique(), df["manufacturer_raw"].nunique(),
        100 * df["is_discontinued"].mean(), priced.median(),
    )
    return df


def provenance() -> dict:
    """Source metadata, surfaced in the dashboard."""
    return {
        "name": "Indian Medicine Dataset",
        "publisher": "Open dataset, community-maintained",
        "description": ("Product master of medicines marketed in India: brand, "
                        "price, manufacturer, pack size and composition."),
        "records": 253_973,
        "coverage": "India, allopathy medicines",
        "licence": "Open (see repository)",
        "source_url": "https://github.com/junioralive/Indian-Medicine-Dataset",
        "kaggle_mirror": ("https://www.kaggle.com/datasets/shudhanshusingh/"
                          "az-medicine-dataset-of-india"),
    }


__all__ = [
    "load_indian_medicines", "load_raw", "download", "dataset_path",
    "normalise_manufacturer", "provenance", "SOURCE_URL",
]
