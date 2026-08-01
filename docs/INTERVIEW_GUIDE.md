# Interview Guide

How to present this project, and how to answer the questions it invites.

The goal is not to recite features. It is to show that you can take an ambiguous
business problem, build something that answers it, and be honest about the limits of
your own analysis. That last part is what separates a strong candidate from a
competent one.

---

## 1. The 60-second version

Use this when someone says *"tell me about a project."*

> I built an analytics platform for a pharmaceutical supply chain. It tracks batches
> through eight stages — from buying the raw ingredient to a unit reaching a patient —
> and finds where product is lost along the way.
>
> The headline finding is that only about **63% of what you buy actually reaches a
> patient**, and the losses aren't spread evenly. Quality testing is the worst stage:
> it destroys the most volume *and* it's the slowest, taking around 19 days. So it's
> the bottleneck on two dimensions at once.
>
> But finding a bottleneck isn't the same as knowing what to do about it. So I ran
> **A/B tests on four operational interventions** — automated quality testing, warehouse
> automation, route optimisation and IoT cold-chain monitoring — and used chi-square
> and two-proportion z-tests to work out which ones produce a real, statistically
> significant improvement rather than noise.
>
> I also built **two machine learning models**: one that predicts which drug a patient
> should get from their clinical readings, and one that predicts whether a
> manufactured batch will go out of specification, based on storage conditions.
>
> It's all in a Streamlit dashboard with twelve pages, backed by 99 tests.

**Then stop.** Let them pick where to go. Whatever they ask about, you have depth.

---

## 2. The story arc

If you get five minutes instead of one, tell it as a narrative. Analysts who tell
stories get remembered; analysts who list technologies don't.

**Act 1 — The question.**
"A pharma company buys expensive raw ingredients and turns them into medicine. Along
the way, product gets lost — rejected at quality control, damaged in transit, expired
on a shelf. Nobody could tell me *where* the losses were concentrated, so nobody knew
where to spend money fixing it."

**Act 2 — Building the measurement.**
"First I had to be able to see the funnel at all. I built a data layer that tracks
every batch through all eight stages — units at each stage, the date it cleared, the
storage conditions it experienced. Then I had to trust it, so I wrote a data quality
module that scores five dimensions and a cleaning layer that fixes what it finds."

**Act 3 — The finding.**
"Quality testing was the bottleneck on both dimensions — most volume lost and slowest
stage. And when I looked at *why* batches were failing, the reasons clustered on assay
and dissolution problems, which trace back to the **incoming raw material**, not the
manufacturing process. Two suppliers had a 78% quality pass rate against 96% for the
best ones. So the quality problem was actually a sourcing problem."

**Act 4 — The proof.**
"That's a hypothesis, not a conclusion. So I set up controlled experiments. In-line
quality technology showed a statistically significant improvement in batch release
rate. But I didn't stop at the p-value — I checked whether the test was adequately
powered, and whether the effect was big enough to be worth the capital. Two of the
four interventions cleared all three bars."

**Act 5 — The honest bit.**
"The supply chain data is a simulation I built, not a real company's data — no public
dataset has batch-level funnel telemetry. I say that clearly in the README. What's
real is the method, and because I know the ground truth I could actually *verify* the
analytics work — I planted four findings in the generator and the tests assert the
analysis recovers all four."

That last act is the strongest thing you can say. Most candidates oversell. Naming
your own limitation before you're asked signals seniority.

---

## 3. Numbers worth knowing cold

Don't memorise everything. Memorise these.

| Number | What it is |
|---|---|
| **63%** | End-to-end yield — units procured that reach a patient |
| **~$127M** | Value lost across the modelled period |
| **8** | Funnel stages |
| **~2,400 / ~7,200** | Batches / shipment legs |
| **19 days** | Mean quality-testing dwell (~28 for failed batches needing retest) |
| **78% vs 96%** | Worst vs best supplier QA pass rate |
| **~4.7 pp** | Potency lost when a cold-chain excursion occurs (p < 0.001) |
| **89%** | Network on-time delivery, against a 95% target |
| **0.98 / 0.988** | Clinical model test accuracy / macro F1 |
| **0.745 / 0.702** | Batch risk test accuracy / macro F1 |
| **4** | Interventions A/B tested |
| **99** | Tests passing |

If you forget a number, say *"I'd need to check the exact figure, but the order of
magnitude was…"* That is a completely acceptable answer and far better than guessing.

---

## 4. The questions you will get

### "Why is the supply chain data synthetic?"

This is the question. Answer it directly and without apology.

> Because the alternative was worse. I needed batch-level funnel telemetry —
> per-stage timestamps, per-stage yields, storage conditions and shipment legs all on
> the same key. No public dataset has that. I could have stitched three or four
> incomplete extracts together and pretended the joins were sound, but the joins
> wouldn't have been sound and the analysis would have been built on a fiction I
> couldn't see.
>
> So I built a generator instead, with three properties. It's **reproducible** — same
> seed, byte-identical output, so every number in my README can be re-derived. It's
> **calibrated** — stage yields, QC release times and OTIF levels come from published
> industry benchmarks, all declared in a config file rather than buried in code. And
> it's **coupled to the real data** — the product mix comes from the actual prescription
> distribution in the Kaggle clinical dataset.
>
> There's a benefit I didn't expect: because I know the ground truth, I can *verify*
> the analytics. I planted four structural signals in the generator, and the test
> suite asserts the analysis discovers each one. You can't do that with real data.

**If they push:** *"So the findings aren't real?"*

> Correct, and I say so in the README. The findings demonstrate method, not facts
> about a company. What transfers to a real engagement is the pipeline, the
> statistical framework, and the discipline of separating what was measured from what
> was experimentally validated. Point the loader at a real extract and everything
> downstream works unchanged.

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

Don't wander. Follow this route.

1. **Home** — "Here's the whole business on one page. 63% yield, $127M lost, and the
   funnel shows you immediately where." *Point at the widest gap.*
2. **Funnel Analytics** → bottleneck table — "Quality testing is flagged on both
   drop-off and delay. The severity score combines them with value lost."
3. **Funnel Analytics** → QA Failure Reasons tab — "And the failure modes are assay and
   dissolution, which are raw-material problems. That's the link to sourcing."
4. **Shipments** → supplier scorecard — "There they are. 78% versus 96%. Same suppliers
   driving both quality failures and late delivery."
5. **A/B Testing** — "So here's the intervention, tested properly." *Show the confidence
   interval, then the power analysis.* "Significant, adequately powered, and the lift is
   commercially meaningful. That's an adopt."
6. **Data Quality** — "And underneath all of it, this is why I trust the numbers." *Show
   bronze vs silver.*
7. **Insights** — "Everything consolidated, with limitations stated."

If they interrupt with questions, let the demo go where they want. Curiosity is a
buying signal.

---

## 6. Résumé bullets

Grounded in what the project actually does:

> - Performed pharmaceutical supply chain funnel analysis across procurement,
>   manufacturing, quality assurance, warehousing, distribution and pharmacy delivery
>   using **Python (Pandas, Plotly)**, identifying quality testing as the primary
>   bottleneck driving ~37% end-to-end unit loss and quantifying $127M of value at risk.
> - Built an end-to-end **machine learning pipeline** for drug classification and batch
>   stability risk prediction — group-wise missing-value imputation, feature
>   engineering, grid-searched model selection across Decision Tree, Random Forest and
>   XGBoost under stratified cross-validation — reaching 0.99 macro F1 on clinical
>   classification with serialised **scikit-learn** pipelines.
> - Designed and executed **A/B experiments** on four supply chain interventions,
>   applying **Chi-Square tests, two-proportion z-tests and Welch's t-tests (SciPy)**
>   with power analysis and practical-significance thresholds, surfaced through a
>   12-page interactive **Streamlit** dashboard backed by 99 automated tests.

---

## 7. Things not to do

**Don't call the data real.** If it comes up later that you implied it, everything else
you said becomes suspect. Lead with it instead — it's a strength.

**Don't say "99% accuracy" as your headline.** On a 200-row dataset with a
deterministic rule, an experienced interviewer hears that as naivety. Lead with the
funnel finding and the experiments.

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
