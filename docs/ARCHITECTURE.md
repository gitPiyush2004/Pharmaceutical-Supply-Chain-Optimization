# Architecture

How the platform is put together, and why each decision was made that way.

---

## Layering

Five layers, each depending only on the ones below it. No layer reaches upward, so
any layer can be tested in isolation and the analytics can be used from a notebook,
a CLI script or the dashboard without change.

```
Presentation   app/                     Streamlit pages (Home + 8)
Visualisation  src/viz/                 theme + chart builders
Analytics/ML   src/analytics/, src/ml/, src/quality/
Data           src/data/                scms · loader
Foundation     src/config.py, src/logger.py, config/config.yaml
```

**The rule that keeps this honest:** nothing below the presentation layer imports
Streamlit. Every analytics function takes a DataFrame and returns a DataFrame or a
dict of scalars. That is what makes the same code usable from the notebook, testable
under pytest, and callable from a future API without a rewrite.

---

## Data flow

Both sources are real. There is no generation step.

```
data/raw/drug200.csv ─────────────────────────────┐  (tracked in the repo)
  no parsing needed - published clean             │
                                                   │
data/external/SCMS_*.csv ─────────────────────────┤  (downloaded, cached)
  │  src/data/scms.py                              │
  │    per-column date formats (%m/%d/%y vs        │
  │      %d-%b-%y - one rule for both is wrong)    │
  │    business-string numeric parsing             │
  │    reason codes: parsed / structural /         │
  │      missing / cross_reference                 │
  └─────────────────────────────────────────────►  │
                       ┌───────────────────────────┘
                       ▼
        src/quality/assessment.py   scores both layers: as published, and interpreted
        src/analytics/*             pipeline · procurement · products · experiments
        src/ml/*                    drug_classification · late_delivery
                       │
                       ▼
              models/*.joblib + *_metadata.json
                       │
                       ▼
              app/ (reads artefacts, never trains)
```

`src/data/loader.py` is the single entry point. `load_table(name)` returns the file
exactly as published; `load_scms()` returns the interpreted form. Results are cached,
so a Streamlit session or notebook pays the parsing cost once.

**Why there are two datasets and not three.** An Indian medicine catalogue (253,973
products) used to sit alongside these, driving a market-structure page. It was dropped
because SCMS answers the same question better: it carries molecule, brand, dosage,
form, factory and the price *actually paid*, on the same rows as delivery performance,
so pricing and service can be joined. A separate list-price catalogue for products
nobody in the dataset bought added row count but no new evidence. The cost of that
decision is scale - the project dropped from ~264k rows to ~10.5k - and the benefit is
that both remaining datasets earn their place.

### Why there is no cleaning layer

An earlier version routed every table through a generic `clean_table` step that
imputed and canonicalised on the way through. It existed to service a simulated
extract with deliberately injected defects, and it went with it. Two reasons it was
not kept for the real data:

1. **It would have caused damage.** Step 6 mode-imputed *every* object column, for
   any table. On SCMS that means filling `N/A - From RDC` with the modal INCO term —
   fabricating 5,404 purchase orders that never existed.
2. **The real cleaning is dataset-specific and inseparable from interpreting the
   source.** Knowing that `%m/%d/%y` applies to two columns and `%d-%b-%y` to three
   is not a generic operation; it is knowledge about this file.

So each dataset that needs interpreting owns it. `drug200` needs none — verified
zero nulls, zero duplicates, no out-of-range values. The consequence is that there
is no longer a hidden transformation between the file on disk and the frame you get
back.

---

## Module responsibilities

### Foundation

| Module | Responsibility |
|---|---|
| `config/config.yaml` | Every tunable constant: paths, dataset locations, quality weights, ML grids, SCMS and product-pricing thresholds, statistical decision rules, the one economic assumption, palette. All of it live — a test asserts no dead simulation blocks survive |
| `src/config.py` | Loads and caches the YAML; wraps it in a dot-accessible `ConfigNode`; resolves repo-relative paths to absolute |
| `src/logger.py` | One namespaced logger tree (`pharmachain.*`) with coloured console output and a rotating file handler. File logging degrades gracefully on read-only filesystems |

**Why a config file rather than constants.** A reviewer can audit every assumption
in one place. It matters most for the two thresholds that decide how a *statistic*
is reported: `ab_testing.skew_limit` (above which a rank test replaces a mean test)
and `ab_testing.min_group_size` (below which a comparison is not reported at all).
Those are the kind of choices that otherwise get made silently, per chart.

### Data layer

| Module | Responsibility |
|---|---|
| `scms.py` | Loads and parses the USAID SCMS delivery history; classifies every ambiguous value with a reason code; derives delivery, lead-time and freight metrics |
| `loader.py` | Cached access to both datasets, published or interpreted; downloads the SCMS export on first use |

### Analytics layer

| Module | Key outputs |
|---|---|
| `pipeline.py` | Value-based service funnel, cumulative lateness funnel, milestone traceability, headline KPIs |
| `procurement.py` | Lead-time breakdown, vendor/country/region/mode scorecards, freight economics, delivery trend, delay distribution |
| `products.py` | Catalogue structure, spend concentration, price spread pooled versus within-year, the generic-versus-branded premium, per-factory prices |
| `experiments.py` | Real group comparisons, stratified comparison (the guard against a misleading pooled average), continuous comparisons, the comparison catalogue with confounds |
| `ab_testing.py` | The statistics themselves: z-test, chi-square, Welch, Mann-Whitney, power, minimum detectable effect, decision rules |

**Why `experiments.py` and `ab_testing.py` are separate.** `ab_testing.py` is a pure
statistics library — it takes counts or Series, reads no data, and knows nothing
about pharmaceuticals. `experiments.py` decides *which* groups to compare, on what
metric, and what confounds the comparison. That split makes the statistics unit
testable against hand-computed values, and it keeps the domain judgement (which
comparison is worth making, and what limits it) in one reviewable place.

**Uniform contract.** Every public function takes an optional pre-loaded DataFrame
as its first argument, defaulting to `None` → load from `loader`. That convention is
what lets the dashboard apply sidebar filters and pass the filtered frame straight
through, with no duplicate filtering logic in the analytics layer.

### ML layer

| Module | Responsibility |
|---|---|
| `preprocess.py` | Cleaning, categorical normalisation, feature engineering, column groups, `ColumnTransformer` construction, stratified splitting |
| `train.py` | Grid search across three algorithms under stratified CV for each model, evaluation (confusion matrix, ROC/PR, per-class report), feature importance, artefact persistence |
| `predict.py` | Loads artefacts, validates input, returns prediction + probabilities + a human-readable explanation; plus the late-delivery gains curve |

**One pipeline object.** Imputation, encoding, scaling and the estimator live inside
a single sklearn `Pipeline`. The serialised `.joblib` therefore needs no separate
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
| `viz/charts.py` | 15 chart builders, each taking a DataFrame and returning a themed `go.Figure` |
| `dashboard/components.py` | `page_setup`, `section`, `kpi_row`, `callout`/`insight`, `show_table`, `chart`, `sidebar_filters`, `methodology` |
| `app/Home.py` + `app/pages/` | Page composition and narrative only |

**Why components live in `src/dashboard/` rather than `app/`.** Streamlit puts the
main script's directory on `sys.path`, so `import components` works under
`streamlit run` but fails under pytest and Streamlit's `AppTest` harness. Moving the
toolkit into the source tree means pages import it as `src.dashboard.components` and
run identically under all three — which is what makes the 9-page smoke test
possible.

**Every page carries a `methodology()` panel.** A reviewer should always be able to
check the definition behind a metric without leaving the dashboard.

---

## Key design decisions

### 1. Structural absence is not missing data

The most consequential decision in the project. `N/A - From RDC` appears in the
purchase-order date column on 5,404 of 10,324 line items. It is not a gap: it
correctly records that no vendor purchase order existed, because those goods were
drawn from regional distribution centre stock.

So `src/data/scms.py` attaches a reason code to every value it cannot parse —
`structural`, `missing`, `cross_reference` or `unparseable` — and the analytics
exclude structural absences rather than filling them. Every function that reports a
lead time also reports the denominator it used.

The same logic applies to `Freight Included in Commodity Cost` (a real cost recorded
elsewhere, not a zero) and `See DN-304 (ID#:10589)` (a pointer to another row, whose
embedded ID makes the value *recoverable* rather than lost).

**The corollary, which is the more interesting half.** A generic completeness check
scores this file 99.3% complete and grade A, because all of these are non-null
strings. Type-aware parsing shows 55% of purchase-order dates and 40% of freight
costs are unusable. And parsing correctly *lowers* the generic score by 1.70 points,
because honest nulls score worse than non-null garbage. A test asserts that negative
sign: if the uplift ever turned positive, it would mean parsing had started imputing.

### 2. No unit funnel, because the data cannot support one

SCMS states `Line Item Quantity` once at order time and never restates it at
delivery. There is no ordered-versus-received pair, no scrap quantity and no
per-stage weight — and every line item in the file was ultimately delivered. A
unit-attrition funnel would have been fabricated.

`pipeline.py` measures attrition in *timeliness* instead: a value funnel where each
band is a strict subset of the one above it, plus a cumulative lateness funnel that
is monotone by construction. Milestone coverage is drawn as a **stacked bar, not a
funnel**, because its coverage runs 74% → 44% → 100% and a funnel shape would imply
sequential loss where a fulfilment route simply has no vendor order to record.

Two tests guard this: one asserts the value funnel is monotone, another asserts the
traceability series is *not* — so if the data ever changed, the chart choice would be
revisited rather than silently becoming misleading.

### 3. Stratify before reporting a group difference

The pooled fulfilment-route gap is +11.9 points. Stratified by era it is +1.9 points
before 2011 and +20.5 after. Those imply opposite actions — replace the channel
versus investigate what changed in 2011 — so `experiments.stratified_comparison`
runs as a matter of course rather than as a special check.

**Name it precisely.** The code separates `is_simpsons_paradox` (requires the gap to
reverse sign) from `interaction_detected` (a gap range exceeding half the pooled gap).
The real `fulfil_via` case trips the second, not the first — direct drop leads in both
eras. Both make a pooled average misleading, and conflating them would be an
overstatement the dashboard would then be repeating everywhere.

The detector is tested in both directions: it must fire on the real case *and* stay
quiet on a synthetic stable-effect fixture. A detector that fires on everything is
worthless.

### 3b. The same pooling trap, found twice

`experiments.stratified_comparison` was written for delivery performance: a pooled
11.9-point service gap between fulfilment routes is 1.9 points before 2011 and 20.5
after. `products.price_spread_within_year` exists because the identical mistake shows
up on a completely unrelated question — the pooled price spread for identical products
reads 5.0x, and within a single year it is 2.5x, because antiretroviral prices fell 80%
over the decade.

Both are reported as a *pair* on their pages, misleading figure beside honest one,
rather than quietly replacing one with the other. A reader who only sees the corrected
number learns a fact; a reader who sees both learns the method. A test asserts the
pooled figure stays larger than the within-year one, so if that ever inverted the
page's argument would fail loudly rather than silently become wrong.

### 4. Model selection on cross-validated score, never on the test set

Three algorithms are grid searched under identical stratified 5-fold CV, and the
winner is chosen on CV macro F1. On the clinical model, random forest edges out the
decision tree on the test split while the decision tree wins on CV — the decision
tree ships, and the dashboard explains why. Selecting whichever model wins on the
test set is how leakage enters a pipeline and inflates every number quoted
afterwards.

Macro F1 rather than accuracy, because the classes are imbalanced and every class
matters equally. Both models use the same metric, which is what makes their numbers
comparable.

### 5. Leakage control is asserted, not assumed

For the late-delivery model, every lead-time measure except the *scheduled* one is
derived from the delivery date and would leak the target. Vendor identity and
manufacturing site are excluded too: with 73 vendors, several appearing a handful of
times, the model would memorise suppliers rather than learn transferable structure —
and it could not score a vendor it had never seen.

A test reads the persisted feature list and fails if any of those columns appear.
Documenting leakage control in prose is not the same as enforcing it.

### 6. Ranking, not classification, when the classes are imbalanced

The late-delivery model's accuracy (0.881) sits *below* the majority-class baseline
(0.885), because only 11.5% of shipments are late. Reported as a classifier it looks
worse than a constant prediction.

The deployable output is the ranking: reviewing the top 20% by predicted risk catches
63.3% of late deliveries, a 3.2× lift. A test pins both facts — that accuracy stays
at or below baseline, and that the gains curve beats random — so nobody later
"improves" the model by optimising the wrong metric.

### 7. Test selection for continuous metrics follows a written rule

Freight as a share of value has a mean of 2,548% against a median of 10.6%. Compared
across product group, Welch returns p = 0.44 and Mann-Whitney p = 6.0e-10 on the same
data. `compare_continuous` runs both, measures skew against
`config.ab_testing.skew_limit`, and states which to quote.

The subtlety worth knowing: the disagreement runs in **both** directions. When only
the rank test is significant, outliers have masked a real difference and Welch is
wrong. When only Welch is significant — as with delivery delay by era, where both
medians are exactly 0 because 61% of deliveries land on their scheduled day —
*neither* is wrong: the mean moved and the typical case did not. The module returns
`"both"` and says the effect lives in the tail.

### 8. Null results are judged on the minimum detectable effect, never post-hoc power

Post-hoc power is computed at the *observed* effect size, so it is a deterministic
function of the p-value: a genuine null mechanically returns low power however large
the sample. Using it to judge a null is circular reasoning dressed as a statistic.

`minimum_detectable_effect` asks the right question instead — given these sample
sizes, how large a gap *would* have shown up? On the first-line-designation
comparison the answer is 1.82 points, so the null rules out anything larger than
that. Since the practical threshold at that baseline is 1.33 points, the verdict is
`NO EFFECT (BOUNDED)` rather than a flat "no difference". Three distinct null
verdicts, because collapsing them loses the useful part.

### 9. Three bars before a difference justifies action

Statistical significance, adequate sensitivity, and practical significance. A large
sample makes almost any difference significant; that alone is never a reason to spend
money. A test asserts that a real-but-trivial 0.2-point gap on 500,000 rows returns
`DO NOT ACT`.

Chi-square is cross-checked against the z-test (χ² = z² for a 2×2 table) — two
independent routes to the same question, which catches contingency-table errors. This
only holds with Yates' correction disabled, so a test guards that setting too.

### 10. Outliers flagged, never silently dropped

The largest values in this data are real. A freight ratio of 13,449,400% is a genuine
record — a tiny line value against a real shipping cost — and it is precisely why the
statistical tests use medians and rank-based tests on freight. Removing it would
improve the accuracy score and delete the finding.

### 11. A rule is removed when its premise is wrong, not tuned

The consistency checker briefly asserted that active-ingredient count could not
exceed pack quantity. On the Indian catalogue that was in the project at the time it
flagged 7,886 products as inconsistent — and every one was correct, because a single
vial of Augmentin contains two molecules. The rule was deleted rather than
threshold-tuned, because the premise was wrong rather than the threshold.

The same principle applies to what is *not* checked now: `drug200` reports zero
cross-field invariants, and that is the honest answer for a flat table with no
chronology and no derived quantities. An audit that invents a rule so it has something
to report is worse than one that says there is nothing to check.

Relatedly, consistency checks now require both columns to be comparably typed. The
raw SCMS date columns are strings, and `"5/3/13" <= "02-Jun-13"` is a lexicographic
comparison that reported an 82% violation rate out of nothing.

---

## Testing strategy

141 tests, ~6 seconds. Structured around invariants rather than implementation
details, because the failure mode that matters is a plausible-but-wrong dashboard,
not a crash.

| Suite | What it protects |
|---|---|
| `test_data_layer.py` | Config completeness and normalised weights; **that no simulation config block survives**; published row counts (a truncated download must fail loudly); that loaders return copies; that `drug200` is clean as published and that **its label is a pure function of its features** — the claim the ML page rests on |
| `test_analytics.py` | Value-funnel monotonicity and reconciliation with raw totals; cumulative lateness ordering; that traceability is deliberately *non*-monotone; **that the pooled price spread exceeds the within-year one**; that zero prices are excluded rather than counted; spend concentration; the Efavirenz price decline; that the branded premium is like-for-like; **a hand-computed z-test anchor**; χ² = z²; inadequate-expected-count flagging; that the MDE shrinks with sample size while post-hoc power stays low on a true null; both directions of Welch/Mann-Whitney disagreement |
| `test_ml_and_quality.py` | Preprocessing normalisation and reproducible splits; artefact completeness; accuracy floors; **that late-delivery accuracy stays at or below baseline while AUC and gains hold up**; **that no leaking or high-cardinality feature is in the model**; that parsing *lowers* the SCMS quality score and leaves drug200 untouched; that string columns are never compared as ordered; that a list-valued column does not break profiling |
| `test_scms_real_data.py` | Published row count and scale; mixed-format date parsing with **regressions for the 478 negative lead times and 1,128 inverted purchase orders**; unit-versus-pack accounting (9.98B units, not 189M); that structural absences stay null and labelled; that reason codes partition every row; the documented 44% PO coverage and 88.5% on-time rate; small-sample exclusion from scorecards |
| `test_experiments.py` | Group comparisons; **that the interaction detector fires on the real case and stays quiet on a synthetic stable effect**; skewed-metric handling |

Three testing choices worth naming:

- **Several tests assert a number is *bad*.** The SCMS quality score must go *down*
  after parsing; the late-delivery model's accuracy must stay *below* baseline. A
  suite that only ever demanded higher numbers would have blessed exactly the
  behaviour this project argues against.
- **Row counts are pinned.** Two of the three datasets are downloaded at runtime. A
  partial CSV would otherwise pass every other test while putting quietly wrong
  numbers on every page.
- **Regression tests name their bug.** The date-parsing and unit-accounting tests
  cite the wrong figures they replaced, so the next person to touch that code knows
  what breaking it looks like.

The dashboard is separately smoke-tested through Streamlit's `AppTest` harness — all
9 pages must render without exception.

---

## Extension points

| To add | Change |
|---|---|
| A lateness threshold | `pipeline.LATENESS_THRESHOLDS` |
| A comparison dimension | One entry in `experiments.COMPARISON_DIMENSIONS`, **with its confound** — the confound is required, not optional |
| A quality dimension | `config.data_quality.weights` + a report function in `assessment.py` |
| A cross-field invariant | `assessment._PAIRWISE_CHECKS` — it is applied automatically wherever both columns exist and are comparably typed |
| A validity rule | `assessment._NAMED_RULES` |
| A model algorithm | `config.ml.*.models` + `param_grid` + a branch in `train.get_model` |
| A dashboard page | A file in `app/pages/`, composed from `src.dashboard.components` |
| A chart type | A builder in `src/viz/charts.py` returning a themed `go.Figure` |
| A README figure | A function in `scripts/export_figures.py` — then **look at the PNG**; the horizontal-bar bug was caught no other way |
| A price threshold | `config.scms.products` — the spread noise floors and table size |
| A dataset | A parser in `src/data/`, an accessor in `loader.py`, an entry in `config.datasets` and `EXPECTED_ROWS` in `scripts/fetch_data.py`. Consider first whether SCMS already carries the field — it did for the entire pricing analysis |

Most extensions are configuration edits, which is the point of pushing every
constant into `config.yaml`.
