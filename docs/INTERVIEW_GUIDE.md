# Interview Guide

How to present this project, and how to answer the questions it invites.

The goal is not to recite features. It is to show that you can take an ambiguous
business problem, build something that answers it, and be honest about the limits of
your own analysis. That last part is what separates a strong candidate from a
competent one.

---

## 1. The 60-second version

Use this when someone says *"tell me about a project."*

> I built a pharmaceutical analytics platform with three parts.
>
> **First, a machine learning pipeline** that predicts which of five drugs suits a
> patient from their clinical readings — blood pressure, cholesterol, age and
> sodium-to-potassium ratio. Full workflow: a data quality report that catches
> missing values, duplicates and invalid entries; median and mode imputation;
> encoding and standardisation; then a Decision Tree with balanced class weights,
> because the classes are imbalanced almost six to one. It gets 98% accuracy with
> per-class AUC above 0.96.
>
> **Second, a supply chain funnel analysis** tracing product through eight stages.
> Only about 63% of what's procured reaches a patient, and quality testing is the
> bottleneck — it loses the most volume *and* takes the longest.
>
> **Third, A/B testing** on four operational fixes, using chi-square and
> two-proportion z-tests to establish which produce a real improvement rather than
> noise.
>
> Part of it runs on **real USAID data — 10,324 actual shipments to 43 countries**.
> The most interesting finding came from there: I built a vendor scorecard expecting
> a bad manufacturer, and the worst performer turned out to be their own internal
> distribution channel at 82.9% on-time.
>
> It's a Streamlit dashboard, nine pages, backed by 104 tests.

**Then stop.** Let them pick where to go. Whatever they ask about, you have depth.

> **Two things to get in early:** say **"real data"**, and say **"balanced class
> weights"**. The first differentiates you from every other portfolio project; the
> second signals you understand imbalanced classification rather than just calling
> `.fit()`.

---

## 2. The story arc

If you get five minutes instead of one, tell it as a narrative. Analysts who tell
stories get remembered; analysts who list technologies don't.

**Act 1 — The question.**
"A global health programme ships HIV and malaria medicines to 43 developing
countries. Every late delivery is a stock-out risk at a clinic. But with 73 vendors,
four transport modes and ten thousand shipments, nobody could say *where* the
reliability problem actually was — so nobody knew where to spend money fixing it."

**Act 2 — Getting the data honest first.**
"USAID publishes this as open data, and it's genuinely messy in ways that matter.
Freight cost is a text column containing things like `Freight Included in Commodity
Cost` and cross-references like `See DN-304`. Dates are mixed formats in the same
file. A standard completeness check scores it 99.3% complete and grade A — and that's
wrong. Over half the purchase-order dates and 40% of freight costs are unusable
strings sitting in text columns. So before any analysis I built a parser that
classifies every ambiguous value and records *why* it couldn't be used."

**Act 3 — The distinction that changed the analysis.**
"The most important call was recognising that `N/A - From RDC` isn't missing data.
It correctly records that no vendor purchase order existed, because those goods came
from regional distribution centre stock. Five thousand rows. If I'd imputed those I'd
have invented five thousand purchase orders that never happened. So I gave every
gap a reason code — structural absence, genuinely missing, or cross-referenced — and
excluded structural absences from the statistics rather than filling them in."

**Act 4 — The finding.**
"I built the vendor scorecard expecting a bad manufacturer. The worst performer was
**their own regional distribution centre channel** — 82.9% on-time across $1.09
billion of product. That's a far more actionable finding, because an internal channel
doesn't need a contract renegotiation to fix. And the mode analysis gave me the
cost-service frontier straight from real numbers: ocean is six times cheaper per kilo
than air and eight points less reliable."

**Act 5 — Making the model useful, not just accurate.**
"I trained a model to predict late delivery. ROC AUC 0.84 — but its accuracy was
*below* the majority-class baseline, because only 11.5% of shipments are late. Rather
than tune toward a vanity number, I changed the decision rule: I built a gains curve
showing that reviewing the top 20% by predicted risk catches 61% of all late
deliveries. That's an expeditor's work queue, and it's genuinely deployable."

**Act 6 — The honest bit.**
"The manufacturing half — storage conditions, batch quality, inventory — is
simulated, because SCMS doesn't cover manufacturing and no public dataset has
per-batch telemetry. I label every simulated figure as simulated. And because I know
that data's ground truth, I could verify the analytics actually work: I planted four
findings in the generator and the tests assert the analysis recovers all four."

That last act is the strongest thing you can say. Most candidates oversell. Naming
your own limitation before you're asked signals seniority.

---

## 3. Numbers worth knowing cold

Don't memorise everything. Memorise these — the real-data block especially.

### Real data (USAID SCMS) — lead with these

| Number | What it is |
|---|---|
| **10,324** | Real shipments, 2006–2015 |
| **43 / 73 / 88** | Countries / vendors / manufacturing sites |
| **$1.63B** | Total commodity value moved |
| **88.5%** | Actual on-time delivery |
| **82.9%** | Worst performer — the internal RDC channel, on $1.09B of value |
| **$1.68 vs $10.02** | Ocean vs air freight, per kg (82.5% vs 90.4% on-time) |
| **44%** | Share of line items with a vendor PO date — the rest bypass vendor ordering |
| **61% / 27% / 11.5%** | Arrive exactly on schedule / early / late |
| **0.845** | Late-delivery model ROC AUC |
| **61% at top 20%** | Late deliveries caught by reviewing the riskiest fifth (3.0× lift) |

### Simulated manufacturing side

| Number | What it is |
|---|---|
| **63%** | End-to-end yield — units procured that reach a patient |
| **~$127M** | Modelled value lost |
| **8** | Funnel stages |
| **19 days** | Mean quality-testing dwell (~28 for failed batches needing retest) |
| **~4.7 pp** | Potency lost on a cold-chain excursion (p < 0.001) |
| **0.98 / 0.988** | Clinical model test accuracy / macro F1 |
| **0.745 / 0.702** | Batch risk test accuracy / macro F1 |
| **4** | Interventions A/B tested |
| **104** | Tests passing |

If you forget a number, say *"I'd need to check the exact figure, but the order of
magnitude was…"* That is a completely acceptable answer and far better than guessing.

---

## 4. The questions you will get

### "Walk me through the drug classification pipeline."

The most likely opening question. Have the sequence ready, and give a reason for
each step rather than just naming it.

> Seven steps.
>
> **Data quality first.** I profile for missing values, duplicates, invalid
> categories and impossible values. The published file is clean, so to actually
> *test* the cleaning code I inject a controlled set of realistic defects — blank
> fields, double-submitted records, `high` versus `HIGH`, a negative age — and check
> the pipeline catches every one. Since I know the ground truth, that doubles as a
> unit test.
>
> **Cleaning, in a specific order.** Standardise text *before* imputing, because
> otherwise the mode is computed over `high` and `HIGH` counted separately. Then
> range checks, deduplication, and finally median imputation for numerics and mode
> for categoricals. Median rather than mean because it's robust to the outliers I
> just nulled.
>
> **Feature engineering.** Ordinal scores for blood pressure and cholesterol —
> one-hot would throw away the ordering, which is real. A combined severity score.
> And a binary flag at the Na/K threshold I found in the exploratory analysis, which
> hands the model the known boundary instead of making it rediscover it.
>
> **Encoding and scaling inside one sklearn Pipeline**, so the scaler only ever sees
> training folds and the serialised model carries its own preprocessing.
>
> **Stratified split**, because the smallest class is 8% of the data and an
> unstratified split could leave barely any of it in the test set.
>
> **Decision Tree with balanced class weights.** A tree because the true boundary is
> axis-aligned — thresholds on Na/K, then splits on blood pressure — so it fits the
> problem exactly, and because you can read the rules and check them against clinical
> logic. Balanced weights so an error on the 8% class costs as much as one on the 45%
> class.
>
> **Evaluation on macro-averaged metrics**, not accuracy, plus a confusion matrix,
> per-class ROC/AUC, and error analysis on every individual mistake.

**If they ask "why not just use a Random Forest?"**

> I compared them. Random forest actually edges it out slightly on the test split,
> but they tie on cross-validated F1 — and CV is the criterion I committed to in
> advance. On a 200-row dataset a single test split is noisy, so switching to
> whichever model wins on it is exactly how leakage creeps in. The tree also gives me
> readable rules, which in a clinical context is worth more than a fractional gain.

### "What did the error analysis actually tell you?"

This is where you separate yourself. The answer is a *negative* result.

> One patient out of fifty was misclassified — Na/K of 14.6, just under the 15
> threshold, predicted DrugY when the actual was drugX. That's the expected place for
> a tree to fail: a hard threshold can't express "close to the boundary."
>
> But the important part is that it was wrong at **100% confidence**. A Decision Tree
> grown to pure leaves reports every single prediction as certain, so the
> probabilities are degenerate — they carry no information at all.
>
> That matters practically. The obvious safety net in a clinical setting is a
> confidence threshold: auto-accept above it, escalate to a pharmacist below it. I
> built the routing table to test that, and it's completely flat — every row sits at
> confidence 1.0, so no threshold separates anything. The fix is probability
> calibration, Platt scaling or isotonic regression, or constraining leaf size.
>
> I found that by doing the error analysis. Reporting 98% accuracy and stopping would
> have hidden it entirely.

### "Which parts are real and which are simulated?"

Answer this crisply and unprompted — volunteering it builds far more credibility than
being caught by it.

> Two of the three datasets are real. The **USAID SCMS delivery history is genuine
> operational data** — 10,324 actual shipments, published as US Government open data —
> and it drives all the procurement, vendor, logistics and freight analysis plus the
> late-delivery model. The **Kaggle drug200 clinical dataset** is real too.
>
> The **manufacturing side is simulated**: batch quality outcomes, storage temperature
> and humidity, inventory snapshots. That's because SCMS records procurement and
> logistics, not manufacturing — and no public dataset carries per-batch storage
> telemetry, because that data doesn't leave a pharma company.
>
> I had a choice there. I could have dropped the stability and inventory analytics
> entirely, or pretended SCMS covered something it doesn't. Instead I built a
> calibrated generator for exactly those gaps, and labelled every simulated figure as
> simulated — in the dashboard and in the README.

**If they push:** *"Doesn't the simulated half undermine it?"*

> It would if I blurred them together. There's a dedicated Real-World SCMS page, the
> models are labelled by data provenance, and the Home page opens by stating which is
> which. The simulation earns its place two ways: it covers domains that otherwise
> couldn't be shown at all, and because I know its ground truth I can *verify* the
> analytics — I planted four structural signals in the generator and the test suite
> asserts the analysis recovers all four. You can't do that with real data.

### "What surprised you?"

A great question to get. Have this ready — it is the best story in the project.

> I built the vendor scorecard expecting to find a bad manufacturer. The worst
> performer turned out to be **"SCMS from RDC"** — which isn't a supplier at all. It's
> their own regional distribution centre channel, running at 82.9% on-time across
> $1.09 billion of commodity value.
>
> That reframed the finding completely. A bad external supplier is a procurement
> problem that needs a contract renegotiation. A bad *internal* channel is an
> operations problem you already control.
>
> And the machine learning corroborated it from a different direction — the top
> feature in the late-delivery model is fulfilment route, ahead of transport mode or
> destination. The model found the same thing without being told to look.

### "Your late-delivery model has worse accuracy than doing nothing. Why ship it?"

If they spot this, they are testing whether you understand your own metrics.

> Because accuracy is the wrong metric, and I'd argue that's the most useful thing
> the project taught me. Only 11.5% of shipments are late, so always predicting
> "on time" scores 88.5%. My model gets 87.9% — marginally worse, and completely
> useless as a comparison.
>
> The metrics that matter are ROC AUC, which is 0.84, and the gains curve. If an
> expeditor has capacity to review a fifth of shipments, the model surfaces **61% of
> everything that will actually be late** — three times better than picking at random.
> That's a real operational tool.
>
> So the fix isn't a better model, it's a better decision rule: use the predicted
> probability to rank and triage, not to make a binary call at 0.5. I show the gains
> curve on the dashboard instead of leading with accuracy, precisely because leading
> with accuracy here would be misleading.

### "Your clinical model gets 98% accuracy. Isn't that suspicious?"

Excellent question to get, because the honest answer is impressive.

> It would be suspicious if I claimed it was hard. The drug200 dataset has 200 rows
> and an almost deterministic decision rule — if the sodium-to-potassium ratio is
> above about 15, the patient gets DrugY, nearly regardless of anything else. Below
> that, blood pressure and cholesterol separate the other four drugs.
>
> So high accuracy is *expected*, and I say so on the dashboard. I actually use it as
> a **correctness check**: my feature importances show Na/K carrying about half the
> decision weight, which is exactly the rule I found in exploratory analysis. If the
> model *hadn't* recovered that structure, I'd have a bug.
>
> The genuinely interesting model is the second one — batch risk — which gets a macro
> F1 of about 0.70. That reflects real irreducible noise, and I'd be suspicious of
> myself if it were much higher.

### "Walk me through how you avoided data leakage."

> Three things. First, everything — imputation, encoding, scaling — lives inside a
> single sklearn Pipeline, so the transformers only ever see training folds. You
> physically can't leak test statistics into the fit.
>
> Second, I select on **cross-validated** score, never on the test set. There's a nice
> example: on the clinical model, random forest actually scores slightly higher than
> the decision tree on my test split, but the decision tree wins on cross-validated
> F1. I ship the decision tree, and the dashboard explains why — picking whichever
> model wins on the test set is exactly how leakage creeps in.
>
> Third, the split is stratified and seeded, so class balance is preserved and the
> split is reproducible.

### "You used chi-square and a z-test. Why both?"

> They answer the same question from different angles, so agreement between them is a
> consistency check. The two-proportion z-test gives me a directional effect size with
> a confidence interval, which is what a business case needs. Chi-square tests
> independence between arm and outcome and gives me Cramér's V for association
> strength.
>
> For a 2×2 table, chi-square should equal the z-statistic squared. I have a test that
> asserts that. If they ever disagreed, I'd have an implementation bug.

### "How do you know an intervention is worth doing?"

> Statistical significance alone isn't enough, and that's the part people skip. With a
> big enough sample, almost any difference becomes significant. So I require three
> things.
>
> **Statistical significance** — the effect probably isn't noise.
> **Adequate power** — the test *could* have detected the effect if it were there. This
> matters because otherwise an inconclusive result gets misread as "no effect", when
> really it means "we couldn't tell".
> **Practical significance** — the lift is big enough to be worth the capital.
>
> Only interventions clearing all three get an ADOPT recommendation. And I report the
> value as *gross* benefit — it excludes implementation cost, so it's not an ROI, and I
> say that explicitly.

### "What was the hardest technical problem?"

Have a real answer. This one is genuinely good:

> Imputation, oddly. About 5% of storage temperature readings were missing — realistic,
> because cold-chain data loggers do drop out.
>
> The naive fix is to impute with the median. But the portfolio median is around 25 °C,
> because most products are stored at room temperature. Cold-chain products sit at
> 2–8 °C. So imputing globally would have invented a 20-degree temperature excursion
> on batches that never had one — and that fabricated excursion would then propagate
> into my stability model, my risk labels, and every conclusion downstream.
>
> So I impute **within group**: cold-chain temperatures from cold-chain batches,
> humidity by region, transit times by transport mode. And where the true value is
> actually knowable, I don't impute at all — supplier reliability is an attribute of
> the *supplier*, so I restore it by lookup from the dimension table. Repair beats
> guessing whenever the value is recoverable.
>
> I have tests for both. One asserts the cold-chain mean stays below 12 °C after
> cleaning; another asserts the repaired reliability values match the dimension exactly.

### "Why Streamlit and not a BI tool?"

> Because a chunk of what this platform does isn't expressible in a BI tool. The
> stability page fits an OLS degradation model and solves it for a shelf-life estimate.
> The A/B page runs power analysis. The simulation page propagates seven levers through
> an elasticity model live. Those need Python in the request path.
>
> If the deliverable were fixed reporting on a warehouse, I'd use Power BI or Tableau
> and it'd be the right call.

### "How would you productionise this?"

> Four things, roughly in order.
>
> **Orchestration** — the CLI scripts become scheduled tasks in Airflow or Dagster, with
> the data quality check as a gate that fails the run rather than a report someone
> reads later.
>
> **Real warehouse** — SQLite becomes Postgres or BigQuery. The SQL layer is already
> separated, so that's a connection change.
>
> **Model registry and monitoring** — MLflow for versioning, plus drift detection on
> input distributions. My batch risk model depends on storage conditions, and if the
> network changes its cold-chain policy the model silently goes stale.
>
> **Serving** — the prediction interface becomes a FastAPI endpoint. It's already a
> clean function returning a dict, so that's thin.

### "What would you do differently?"

> Two things.
>
> I'd reach for **survival analysis** on the shelf-life question instead of OLS on
> potency. What I actually care about is *time until a batch goes out of specification*,
> which is a time-to-event problem with censoring — plenty of batches never fail within
> the observation window. Cox regression handles that properly; my linear model doesn't.
>
> And I'd add a **discrete-event simulation** to sanity-check the main recommendation.
> My elasticity model says compressing quality testing improves throughput, but it has
> no concept of capacity or queueing — so it can't tell me whether I'd just be moving
> the bottleneck to packaging. That's a real weakness in the advice I'm giving.

### "How did you decide what to test?"

> I let the funnel analysis pick. Quality testing was the worst stage on both volume
> and time, so automated quality testing was the obvious first candidate. Cold-chain
> excursions showed a large, significant potency effect in the stability analysis, so
> IoT monitoring was second. The regional analysis showed one region well below OTIF
> target, which pointed at route optimisation.
>
> The observational analysis generates hypotheses; the experiments test them. Running
> experiments on interventions the data hadn't already flagged would be guessing.

---

## 5. Demo path (5 minutes)

Nine pages, but you only need five. Don't wander.

1. **Home** — "Three pieces of work, and I label which data is real." *Ten seconds.*
2. **ML Models → Drug Classification** — the headline. Show the confusion matrix,
   then the feature importance: "Na/K carries about half the decision weight, which
   is exactly the rule I found in EDA — the model recovering known structure is a
   correctness check."
3. **Funnel Analytics** — "Only 63% of procured units reach a patient." Show the
   funnel, then the bottleneck table: "quality testing is worst on both volume and
   time."
4. **Real-World Operations** — "This half is real: 10,324 actual USAID shipments."
   Show the vendor scorecard: "I expected a bad manufacturer. It's their own
   internal channel."
5. **A/B Testing** — "And this is how I'd decide whether a fix is worth the money."
   Show the confidence interval, then the power analysis.

If there's time, **Data Quality** is the best supporting page: the generic profiler
scores the real SCMS file 99.3% complete and grade A, and it's wrong.

Let questions redirect you. Curiosity is a buying signal.

## 6. Résumé bullets

Grounded in what the project actually does, leading with the real data:

> - Analysed **10,324 real USAID pharmaceutical shipments** (43 countries, 73 vendors,
>   $1.63B commodity value) using **Python (Pandas, Plotly)**, building a procurement
>   funnel and vendor scorecard that identified an internal distribution channel — not
>   an external supplier — as the weakest link at 82.9% on-time across $1.09B of product.
> - Built an end-to-end **machine learning pipeline** across three models — drug
>   classification, batch stability risk, and late-delivery prediction on real shipment
>   data — with group-wise imputation, leakage-controlled feature engineering and
>   grid-searched selection across Decision Tree, Random Forest and XGBoost under
>   stratified cross-validation (**ROC AUC 0.84**; top-20% risk targeting captures 61%
>   of late deliveries, a 3× lift).
> - Designed and executed **A/B experiments** on four supply chain interventions using
>   **Chi-Square tests, two-proportion z-tests and Welch's t-tests (SciPy)** with power
>   analysis and practical-significance gating, delivered through a 9-page interactive
>   **Streamlit** dashboard backed by 104 automated tests.

## 7. Things not to do

**Don't blur real and simulated.** Two datasets are genuinely real; the
manufacturing telemetry is not. Say which is which before you're asked — being caught
conflating them costs you far more than volunteering it ever could.

**Don't lead with accuracy on either model.** On the clinical model it's a 200-row dataset with a deterministic rule; on the
late-delivery model accuracy is literally worse than doing nothing. Lead with the RDC
finding and the gains curve.

**Don't list the tech stack unprompted.** "I used Pandas, NumPy, scikit-learn,
XGBoost…" tells them nothing. They can read the README. Talk about the *problem*.

**Don't hide the batch risk model's 0.70 F1.** Volunteer it and explain why a higher
number would worry you. Knowing when a model is *good enough* and when it's
suspiciously good is a senior skill.

**Don't claim ROI.** You have gross benefit, not net. Say "gross benefit, excluding
implementation cost" and you sound rigorous instead of hand-wavy.

---

## 8. If you're asked to go deeper

Places where you have real depth, if the conversation goes technical:

- **The degradation model** — simplified Arrhenius relationship, temperature excess above
  labelled storage condition, humidity threshold effect, interpretable coefficients with
  physically correct signs, solved for the specification crossing to get a shelf-life
  estimate. R² around 0.74.
- **`thermal_load`** — the engineered interaction (excess temperature × exposure days)
  that turns out to be the top predictor of batch risk, ahead of raw duration. Explains
  *why* you engineered it: degradation depends on temperature and time jointly.
- **The bronze/silver split** — six ordered cleaning steps, full remediation audit trail,
  measured quality uplift, and the reason profiling the *raw* layer matters (auditing the
  clean table scores your cleaning code, not the data).
- **Forecast backtesting** — why the naive moving-average baseline is included on purpose,
  and why bias is reported separately from MAPE (over-forecasting becomes expiry
  write-off, under-forecasting becomes stock-out; MAPE hides the direction).
- **Configuration design** — every threshold in one auditable YAML file, so sensitivity
  testing is a config edit and a reviewer can check every assumption in one place.
