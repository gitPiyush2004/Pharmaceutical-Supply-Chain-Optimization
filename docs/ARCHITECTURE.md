# Architecture

How the platform is put together, and why each decision was made that way.

---

## Layering

Six layers, each depending only on the ones below it. No layer reaches upward, so
any layer can be tested in isolation and the analytics can be used from a notebook,
a CLI script or the dashboard without change.

```
Presentation   app/                     Streamlit pages (13)
Visualisation  src/viz/                 theme + chart builders
Analytics/ML   src/analytics/, src/ml/, src/quality/
Data           src/data/                scms (REAL), generator, cleaning, loader, database
Foundation     src/config.py, src/logger.py, config/config.yaml
```

**The rule that keeps this honest:** nothing below the presentation layer imports
Streamlit. Every analytics function takes a DataFrame and returns a DataFrame or a
dict of scalars. That is what makes the same code usable from the notebook, testable
under pytest, and callable from a future API without a rewrite.

---

## Data flow

```
data/external/SCMS_*.csv ─────────────────┐   (REAL USAID data, 10,324 shipments)
  │  src/data/scms.py                      │
  │    mixed-format date parsing           │
  │    business-string numeric parsing     │
  │    reason codes: parsed / structural / │
  │      missing / cross_reference         │
  └──────────────────────────────────────► │  consumed by src/analytics/procurement.py
                                           │  and the late_delivery model
data/raw/drug200.csv ─────────────────────┤   (real Kaggle data, never modified)
                                          │
src/data/generator.py                     │
  │  seeded, calibrated digital twin       │
  │  + deliberate defect injection         │
  ▼                                        │
data/raw/*.csv  ── BRONZE ────────────────┤
  │                                        │
  │  src/data/cleaning.py                  │
  │    1. whitespace / casing              │
  │    2. master-data repair               │
  │    3. category canonicalisation        │
  │    4. deduplication                    │
  │    5. range checks → NaN               │
  │    6. grouped imputation               │
  ▼                                        │
in-memory  ── SILVER ─────────────────────┤
  │                                        │
  ├─→ src/data/database.py → SQLite star schema
  │                                        │
  ├─→ src/quality/assessment.py  (profiles BRONZE, scores both)
  ├─→ src/analytics/*            (consume SILVER)
  └─→ src/ml/*                   (consume SILVER + clinical)
              │
              ▼
        models/*.joblib + *_metadata.json
              │
              ▼
        app/ (reads artefacts, never trains)
```

`src/data/loader.py` is the single entry point. `load_table(name)` returns the silver
layer; `load_table(name, raw=True)` returns bronze. Results are cached per
`(name, raw)` pair, so a Streamlit session or notebook pays the cleaning cost once.

---

## Module responsibilities

### Foundation

| Module | Responsibility |
|---|---|
| `config/config.yaml` | Every tunable constant: paths, generation parameters, funnel definition, quality weights, ML grids, inventory thresholds, forecasting settings, stability constants, A/B experiments, simulation elasticities, economics, palette |
| `src/config.py` | Loads and caches the YAML; wraps it in a dot-accessible `ConfigNode`; resolves repo-relative paths to absolute |
| `src/logger.py` | One namespaced logger tree (`pharmachain.*`) with coloured console output and a rotating file handler. File logging degrades gracefully on read-only filesystems |

**Why a config file rather than constants.** A reviewer can audit every assumption in
one place, and sensitivity testing becomes a config edit rather than a code change.
The funnel stage list itself lives here, so adding a ninth stage is a config change
plus data, not a refactor.

### Data layer

| Module | Responsibility |
|---|---|
| `scms.py` | **Real data.** Loads and parses the USAID SCMS delivery history; classifies every ambiguous value with a reason code; derives delivery, lead-time and freight metrics |
| `generator.py` | Builds the digital twin: three dimensions, four facts. Encodes four structural signals. Injects realistic defects and logs what it injected |
| `cleaning.py` | Promotes bronze to silver in six ordered steps; returns a remediation log |
| `loader.py` | Cached access to any table in either layer; auto-generates data if missing |
| `database.py` | Materialises the CSVs into SQLite with indexes; holds eight named analytical queries |

**Why generate the supply chain data.** No public dataset carries batch-level funnel
telemetry — per-stage timestamps, per-stage yields, storage conditions and shipment
legs on one key. The choice was between stitching incomplete extracts together and
pretending the joins were sound, or shipping a documented, calibrated, reproducible
generator. The generator also makes the analytics *verifiable*: because ground truth
is known, tests can assert the analysis recovers the planted signals.

**Why inject defects.** A quality module that finds nothing proves nothing, and an ML
pipeline advertising "missing value imputation" over complete data is hollow. Real
pharmaceutical extracts have IoT dropouts, ERP double-postings and free-text region
fields, so the bronze layer reproduces them — and the generator records exactly what
it injected, giving a ground-truth test of the profiler.

### Analytics layer

| Module | Key outputs |
|---|---|
| `procurement.py` | **Real data.** Procurement milestone coverage, lead-time breakdown, vendor/country/region/mode scorecards, freight economics, delivery trend |
| `funnel.py` | Stage conversion, drop-off, dwell time, bottleneck ranking with severity scoring, loss attribution, QA root causes, quarterly trend |
| `inventory.py` | Turnover, ABC classification, stock-out / overstock / expiry registers, warehouse utilisation |
| `shipments.py` | Supplier scorecard, regional and carrier performance, transit-time distribution, late-shipment analysis |
| `forecasting.py` | Aggregation, decomposition, three forecast methods, backtest ranking |
| `stability.py` | Binned condition effects, OLS degradation model, shelf-life estimation, excursion significance test |
| `ab_testing.py` | Experiment simulation, two-proportion z-test, chi-square, Welch t-test, power analysis, segments, costed recommendation |
| `simulation.py` | Seven-lever elasticity model, tornado analysis, sensitivity sweeps, preset scenarios |

**Uniform contract.** Every public function takes an optional pre-loaded DataFrame as
its first argument, defaulting to `None` → load from `loader`. That single convention
is what lets the dashboard apply sidebar filters and pass the filtered frame straight
through, with no duplicate filtering logic in the analytics layer.

### ML layer

| Module | Responsibility |
|---|---|
| `preprocess.py` | Cleaning, categorical normalisation, feature engineering, column groups, `ColumnTransformer` construction, stratified splitting |
| `train.py` | Grid search across three algorithms under stratified CV for each of the three models, evaluation (confusion matrix, ROC/PR, per-class report), feature importance, artefact persistence |
| `predict.py` | Loads artefacts, validates input, returns prediction + probabilities + a human-readable explanation |

**One pipeline object.** Imputation, encoding, scaling and the estimator live inside a
single sklearn `Pipeline`. The serialised `.joblib` therefore needs no separate
transformer at serving time, which makes train/serve skew structurally impossible
rather than merely unlikely.

**Metadata is a first-class artefact.** Alongside each model, a JSON file records the
chosen algorithm, tuned hyper-parameters, the full algorithm comparison, test
metrics, confusion matrix, ROC and PR curve data, complete feature importances, the
seed and a UTC timestamp. The dashboard renders the entire evaluation from this file
and never retrains in the browser.

### Visualisation and presentation

| Module | Responsibility |
|---|---|
| `viz/theme.py` | Palette from config, registered Plotly template, number formatters, semantic colour lookups |
| `viz/charts.py` | 28 chart builders, each taking a DataFrame and returning a themed `go.Figure` |
| `dashboard/components.py` | `page_setup`, `section`, `kpi_row`, `callout`/`insight`, `show_table`, `chart`, `sidebar_filters`, `methodology` |
| `app/Home.py` + `app/pages/` | Page composition and narrative only |

**Why components live in `src/dashboard/` rather than `app/`.** Streamlit puts the main
script's directory on `sys.path`, so `import components` works under `streamlit run`
but fails under pytest and Streamlit's `AppTest` harness. Moving the toolkit into the
source tree means pages import it as `src.dashboard.components` and run identically
under all three — which is what makes the 12-page smoke test possible.

**Every page carries a `methodology()` panel.** A reviewer should always be able to
check the definition behind a metric without leaving the dashboard.

---

## Key design decisions

### 1. Bronze/silver separation instead of cleaning in place

*Alternative considered:* generate clean data and skip the cleaning layer entirely.

*Why rejected:* it would make the data quality module decorative and the imputation
claims hollow. Separating layers also means the quality audit profiles what
*arrived* — auditing only the cleaned table scores the cleaning code, not the data.

### 2. Grouped imputation, and master-data repair where possible

A missing cold-chain temperature imputed with the portfolio median (~25 °C) would
fabricate a 20 °C excursion, which then propagates into the stability model, the risk
labels and every downstream conclusion. So imputation happens within group:
temperature by product, humidity by region, transit time by transport mode.

`supplier_reliability` is treated differently again. It is an attribute of the
supplier, not the batch, so the true value is *knowable* and is restored by lookup
from the dimension table. Preferring repair over imputation wherever the value is
recoverable is the general principle. Tests assert both behaviours.

### 2b. Structural absence is not missing data

The single most consequential decision in the real-data layer. `N/A - From RDC`
appears in the purchase-order date column on 5,404 of 10,324 line items. It is not a
gap: it correctly records that no vendor purchase order existed, because those goods
were drawn from regional distribution centre stock.

Imputing it would fabricate 5,404 purchase orders and corrupt every lead-time
statistic downstream. So `src/data/scms.py` attaches a reason code to every value it
cannot parse — `structural`, `missing`, `cross_reference` or `unparseable` — and the
analytics exclude structural absences rather than filling them. Every function that
reports a lead time also reports the denominator it used.

The same logic applies to `Freight Included in Commodity Cost` (a real cost recorded
elsewhere, not a zero) and `See DN-304 (ID#:10589)` (a pointer to another row).

A related finding worth stating: a generic completeness check scores this file 99.3%
complete and grade A, because all of these are non-null strings in text columns.
Type-aware parsing shows 55% of purchase-order dates and 40% of freight costs are
unusable. Profiling types is not the same as profiling meaning.

### 3. Model selection on cross-validated score, never on the test set

Three algorithms are grid searched under identical stratified 5-fold CV, and the
winner is chosen on CV macro F1. On the clinical model random forest edges out the
decision tree on the test split while the decision tree wins on CV — the decision
tree ships, and the dashboard explains why. Selecting whichever model wins on the
test set is how leakage enters a pipeline and inflates every number quoted
afterwards.

Macro F1 rather than accuracy, because the classes are imbalanced and every class
matters equally.

### 4. Three bars before an intervention justifies capital

Statistical significance, adequate power, and practical significance. A large sample
makes almost any difference significant; that alone is never a reason to spend money.
Power is reported so an inconclusive result can be read correctly as *under-powered*
rather than as evidence of no effect.

Chi-square is cross-checked against the z-test (χ² ≈ z² for a 2×2 table) — two
independent routes to the same question, which catches implementation errors.

### 5. Forecast method chosen by backtest, with bias reported separately

A naive moving average is included deliberately: a sophisticated method that cannot
beat it is not earning its complexity. Bias is reported alongside MAPE because MAPE
treats over- and under-forecasting symmetrically and a supply chain does not —
persistent over-forecasting becomes expiry write-off, under-forecasting becomes
stock-out.

### 6. Outliers flagged, never silently dropped

In stability analytics an extreme storage temperature is the signal, not noise. The
cleaning layer only nulls values that are *physically impossible* (a negative
duration, potency above label claim), never values that are merely extreme. A test
asserts that genuine excursions survive cleaning.

### 7. Elasticity simulation rather than discrete-event

The simulator starts from the *measured* baseline and applies configured elasticities.
It is a first-order model for exploring directional trade-offs in a planning
conversation — fast enough to be interactive, and stated as such rather than
presented as a queueing simulation.

---

## Testing strategy

131 tests, ~9 seconds. Structured around invariants rather than implementation
details, because the failure mode that matters is a plausible-but-wrong dashboard.

| Suite | What it protects |
|---|---|
| `test_data_layer.py` | Config completeness and normalised weights; **generation determinism** (byte-identical under the same seed); funnel volume monotonicity; stage date ordering; calibration bounds; the four planted structural signals; that bronze really is dirty and silver really is clean; grouped-imputation semantics; cleaning idempotence |
| `test_analytics.py` | Conversion reconciling with raw units; drop-off complementing conversion; shares summing to 100; ABC thresholds; turnover/days-of-inventory reciprocity; forecasts landing in the future; decomposition reconstructing the series; **a hand-computed z-test anchor**; χ² ≈ z²; power rising with sample size; simulation directionality |
| `test_ml_and_quality.py` | Preprocessing normalisation and reproducible splits; artefact completeness; accuracy floors; **batch risk beating the majority-class baseline**; confusion-matrix totals matching test rows; importances summing to 1; known clinical rules reproduced from real drug200 rows; input validation; that the profiler finds the injected defects; that cleaning improves every table; that `drug200.csv` is never modified |
| `test_scms_real_data.py` | **Real data.** Published row count and scale; that cleaning never drops a shipment; mixed-format date parsing; that structural absences stay null and labelled; that reason codes partition every row; that parsed numerics are plain float64; the documented 44% PO coverage and 88.5% on-time rate; small-sample exclusion from scorecards; the RDC-is-worst and ocean-vs-air findings; **that the late-delivery model uses no leaking feature and no high-cardinality identifier**; that its gains curve is monotonic and beats random |

Two testing choices worth naming:

- **Determinism is tested directly.** The reproducibility claim in the README is only
  credible if something enforces it.
- **The planted signals are asserted.** `test_quality_testing_is_flagged_as_bottleneck`
  and `test_weakest_region_is_middle_east_africa` fail if generator calibration drifts,
  which is exactly when the documented findings would silently stop being true.

The dashboard is separately smoke-tested through Streamlit's `AppTest` harness — all
12 pages must render without exception.

---

## Extension points

| To add | Change |
|---|---|
| A funnel stage | `config.funnel.stages` + unit/date column maps, then generator support |
| A quality dimension | `config.data_quality.weights` + a report function in `assessment.py` |
| An A/B experiment | One entry under `config.ab_testing.experiments` — no code change |
| A forecasting method | `config.forecasting.methods` + a `_fit_*` function in `forecasting.py` |
| A model algorithm | `config.ml.*.models` + `param_grid` + a branch in `train.get_model` |
| A simulation lever | `config.simulation.levers` + an elasticity in `config.simulation.elasticity` |
| A dashboard page | A file in `app/pages/`, composed from `src.dashboard.components` |
| A real-data source | A parser in `src/data/`, an accessor in `loader.py`, an entry in `config.datasets` |

Most extensions are configuration edits, which is the point of pushing every constant
into `config.yaml`.
