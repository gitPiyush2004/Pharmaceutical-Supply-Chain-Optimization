# Interview Guide

How to present this project, and how to answer the questions it invites.

The goal is not to recite features. It is to show that you can take an ambiguous
business problem, build something that answers it, and be honest about the limits of
your own analysis. That last part is what separates a strong candidate from a
competent one — and it is what this project is built around.

---

## 1. The 60-second version

Use this when someone says *"tell me about a project."*

> I built a pharmaceutical analytics platform on two real public datasets — no
> simulated data anywhere.
>
> **The core of it** is 10,324 actual USAID shipments of HIV and malaria medicines to
> 43 countries. $1.63 billion of commodity value, of which **$259 million arrives
> late**. I traced where, then tested whether each difference was real.
>
> **The best finding is one I nearly got wrong.** The worst-performing fulfilment
> route is the programme's own distribution centre — 11.9 points behind direct drop.
> I almost reported that. Then I stratified by time period, and the gap is **1.9
> points before 2011 and 20.5 after**. The channel didn't start weak, it collapsed.
> Those two readings imply opposite actions — replace the channel, versus find out
> what changed in 2011.
>
> **Two ML pipelines.** Drug classification on the Kaggle clinical dataset, and
> late-delivery prediction on the real shipments — where the accuracy is *worse* than
> always guessing "on time", because only 11.5% are late. So I deployed it as a
> ranking instead: reviewing the riskiest 20% catches 63% of late deliveries.
>
> Streamlit dashboard, nine pages, 141 tests.

**Then stop.** Let them pick where to go. Whatever they ask about, you have depth.

> **Two things to get in early:** say **"all real data"**, and say **"I nearly got
> that wrong"**. The first differentiates you from every other portfolio project; the
> second is the thing interviewers remember.

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
"USAID publishes this as open data, and it's messy in ways that matter. Freight cost
is a text column containing things like `Freight Included in Commodity Cost` and
cross-references like `See DN-304 (ID#:10589)`. A standard completeness check scores
the file **99.3% complete and grade A** — and that's wrong. Over half the
purchase-order dates and 40% of freight costs are unusable strings sitting in text
columns. Perfectly non-null, completely unparseable.

So before any analysis I wrote a parser that classifies every ambiguous value and
records *why* it couldn't be used."

**Act 3 — The distinction that changed everything downstream.**
"The most important call was recognising that `N/A - From RDC` **isn't missing
data**. It correctly records that no vendor purchase order existed, because those
goods came from distribution centre stock. Five thousand four hundred rows.

I'd originally built a generic cleaning layer that mode-imputed every text column. On
this file that would have invented 5,404 purchase orders that never happened, and
every lead-time figure downstream would have been wrong. So I deleted the generic
layer and gave every unusable value a reason code instead — structural absence,
genuinely missing, or cross-referenced — and excluded structural absences from the
statistics rather than filling them.

Here's the part I like: **parsing the file correctly makes its quality score go
down**, by 1.70 points, because honest nulls score worse on completeness than
non-null garbage. There's a test asserting that negative sign. Any pipeline judged on
'did the quality score improve' would be incentivised to leave the garbage in."

**Act 4 — The finding, and catching myself getting it wrong.**
"I built the vendor scorecard expecting a bad manufacturer. The worst performer was
**their own regional distribution centre channel**, carrying $1.09 billion of
product. I nearly reported an 11.9-point gap against direct-drop fulfilment.

Then I stratified by time period: **1.9 points before 2011, 20.5 points after**. The
channel degraded — it wasn't always weak. The pooled number describes a permanent
structural gap that never existed, and it points at the wrong fix: this is a
degradation to investigate, not a channel to replace.

If they know their statistics, get the name right before they correct you: **this is
effect modification, not textbook Simpson's paradox.** Simpson's paradox needs the
difference to *reverse sign*, and direct drop is ahead in both eras — what changes is
the magnitude, by a factor of ten. My code distinguishes the two: `is_simpsons_paradox`
requires a sign flip and returns False here, while `interaction_detected` returns
True. Same practical consequence, different name, and being loose about it is exactly
the sort of thing a good interviewer will pick up.

I only caught it because I stratified. That's now a function in the codebase with a
test that checks it fires on this case *and stays quiet on a synthetic stable effect*
— a detector that fires on everything is worthless."

**Act 5 — Making the model useful rather than accurate.**
"I trained a model to predict late delivery. ROC AUC 0.85 — but its accuracy is
*below* the majority-class baseline, because only 11.5% of shipments are late.
Reported as a classifier it looks worse than a constant guess.

Rather than tune toward a vanity number I changed the decision rule: a gains curve
showing that reviewing the top 20% by predicted risk catches **63% of all late
deliveries**, a 3.2× lift. That's an expeditor's work queue, and it's deployable.
There's a test pinning both facts — that accuracy stays at or below baseline, and
that the gains beat random — so nobody later 'improves' it by optimising the wrong
metric."

**Act 6 — The honest bit: what I deleted.**
"An earlier version of this project had a third model predicting batch stability
risk, and a manufacturing funnel. Both ran on data I'd generated myself, because no
public dataset carries per-batch storage telemetry. I checked properly — Kaggle,
openFDA, data.gov.in, CDSCO, Mendeley, Zenodo — and the one promising candidate
turned out to be a simulation itself.

So I deleted them. About a thousand lines, two dashboard pages and a model. The
analysis was fine; the problem was that I could make the numbers say whatever I'd
calibrated them to say. Two real models beat three when one is circular."

That last act is the strongest thing you can say. Most candidates add. Being able to
name what you *removed*, and why, signals judgement.

---

## 3. Numbers worth knowing cold

Don't memorise everything. Memorise the top block.

### The delivery pipeline (USAID SCMS) — lead with these

| Number | What it is |
|---|---|
| **10,324** | Real shipments, 2006–2015 |
| **43 / 73 / 88** | Countries / vendors / manufacturing sites |
| **$1.63B / $259M** | Commodity value moved / value arriving late |
| **84.1% / 88.5%** | Value on time / shipments on time (they differ — late shipments skew large) |
| **93.4% → 73.9%** | RDC channel on-time, pre-2011 vs post-2010 — the collapse |
| **+1.9 → +20.5 pp** | RDC gap by era — pooled 11.9 pp is misleading (effect modification, *not* a sign reversal) |
| **61% / 27% / 11.5%** | Arrive exactly on schedule / early / late |
| **44%** | Line items with a vendor PO date — the rest bypass vendor ordering entirely |
| **99.3% vs 60%** | Freight column completeness score vs actually usable |

### Statistics

| Number | What it is |
|---|---|
| **p = 7.5e-80** | RDC vs direct drop, two-proportion z-test (z = −18.9) |
| **p = 0.812** | First-line designation — a genuine null |
| **1.82 pp** | Minimum detectable effect on that null (post-hoc power was 6% — the wrong tool) |
| **0.44 vs 6.0e-10** | Welch vs Mann-Whitney on freight ratio by product group, same data |
| **78** | Skewness of freight-as-share-of-value (mean 2,548%, median 10.6%) |

### Machine learning

| Number | What it is |
|---|---|
| **0.98 / 0.988 / 0.989** | Drug classification accuracy / macro F1 / AUC |
| **99.5%** | CV accuracy ceiling at depth ≥ 4 (depth 3 reaches only 88.5%) — one patient |
| **0.848** | Late-delivery ROC AUC |
| **0.881 vs 0.885** | Late-delivery accuracy vs majority baseline — *below* it |
| **63.3% at top 20%** | Late deliveries caught by the riskiest fifth (3.2× lift) |

### Product & pricing (same SCMS file)

| Number | What it is |
|---|---|
| **184 / 86 / 88** | Catalogue items / molecules / factories — SCMS is a catalogue too |
| **5.0× vs 2.5×** | Price spread for identical products, pooled vs within-year |
| **80%** | Efavirenz 600mg price decline 2006→2015 (why pooling misleads) |
| **2.1×** | Median branded premium over generic, same product, same year |
| **6.7×** | Nevirapine 200mg 2009: $0.050 generic vs $0.335 Viramune |
| **63% / 94%** | Share of priced value in the top 5 / top 15 of 92 products |
| **82.9%** | Line items with a usable unit price — the rest are excluded, not zeroed |

If you forget a number, say *"I'd need to check the exact figure, but the order of
magnitude was…"* That is a completely acceptable answer and far better than guessing.

---

## 4. The questions you will get

### "Walk me through the drug classification pipeline."

The most likely opening question. Have the sequence ready, and give a reason for each
step rather than just naming it.

> Seven steps.
>
> **Data quality first.** I profile for missing values, duplicates, invalid
> categories and impossible values. This dataset is genuinely clean — verified zero
> nulls, zero duplicates, nothing out of range — so I say that rather than dressing
> up a no-op as remediation.
>
> **Cleaning, in a specific order.** Standardise text *before* imputing, because
> otherwise the mode is computed with `high` and `HIGH` counted separately. Then
> range checks, then median imputation for numerics and mode for categoricals. Median
> rather than mean because it's robust to outliers.
>
> **Feature engineering.** Ordinal scores for blood pressure and cholesterol —
> one-hot would throw away the ordering, which is real. A combined severity score.
> And a binary flag at the Na/K threshold I found in exploratory analysis, which
> hands the model the known boundary instead of making it rediscover it.
>
> **Encoding and scaling inside one sklearn Pipeline**, so the scaler only ever sees
> training folds and the serialised model carries its own preprocessing — train/serve
> skew becomes structurally impossible rather than merely unlikely.
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

> I compared them. Random forest actually edges it out on the test split, but the
> decision tree wins on cross-validated macro F1 — and CV is the criterion I
> committed to in advance. On a 200-row dataset a single test split is noisy, so
> switching to whichever model wins on it is exactly how leakage creeps in. The tree
> also gives readable rules, which in a clinical context is worth more than a
> fractional gain.

### "Your clinical model gets 98% accuracy. Isn't that suspicious?"

Excellent question to get, because the honest answer is better than the number.

> It would be suspicious if I claimed it was hard. **The label is a pure function of
> the features** — I verified it, with zero exceptions in all 200 rows. If the
> sodium-to-potassium ratio is at or above 15.015 the patient gets DrugY, and below
> that threshold blood pressure, cholesterol and a single age cut at 50 resolve the
> remaining four exactly. There's a test asserting that property, because the claim
> depends on it.
>
> So 100% is *arithmetically attainable*, which means 98% isn't an achievement — it's
> slightly below what the problem permits. I say that on the dashboard.
>
> What's genuinely interesting is *why* it isn't 100%. Cross-validated accuracy
> plateaus at **99.5% for every depth from 4 upward**; depth 3 is too shallow to
> express the rule at all and only reaches 88.5%. On 200 rows, 99.5% is exactly one
> patient. The cause is sample size, not model capacity: the true boundary sits in
> the narrow gap between 14.642 — the highest non-DrugY ratio — and 15.015, and a
> tree fitted on 150 rows doesn't always place its split inside that gap. The single
> test error is at Na/K 14.64, the boundary value itself.
>
> I also use the high accuracy as a **correctness check**: feature importances show
> Na/K carrying about half the decision weight, which is exactly the rule I found in
> EDA. If the model *hadn't* recovered that structure, I'd have a bug.

### "What did the error analysis actually tell you?"

This is where you separate yourself. The answer is a *negative* result.

> One patient out of fifty, at the boundary — the expected place for a tree to fail,
> because a hard threshold can't express "close to the line."
>
> But the important part is that it was wrong at **100% confidence**. A Decision Tree
> grown to pure leaves reports every single prediction as certain, so the
> probabilities are degenerate — they carry no information at all.
>
> That matters practically. The obvious safety net in a clinical setting is a
> confidence threshold: auto-accept above it, escalate to a pharmacist below it. I
> built the routing table to test that idea, and it's completely flat — every row
> sits at confidence 1.0, so no threshold separates anything. The fix is probability
> calibration, Platt scaling or isotonic regression, or constraining leaf size.
>
> Reporting 98% accuracy and stopping would have hidden that entirely.

### "Your late-delivery model has worse accuracy than doing nothing. Why ship it?"

If they spot this, they're testing whether you understand your own metrics.

> Because accuracy is the wrong metric, and that's the most useful thing this part of
> the project taught me. Only 11.5% of shipments are late, so always predicting "on
> time" scores 88.5%. My model gets 88.1% — marginally worse, and useless as a
> comparison.
>
> The metrics that matter are ROC AUC, which is 0.85, and the gains curve. If an
> expeditor has capacity to review a fifth of shipments, the model surfaces **63% of
> everything that will actually be late** — 3.2× better than random.
>
> So the fix isn't a better model, it's a better decision rule: use the predicted
> probability to rank and triage, not to make a binary call at 0.5. The dashboard
> leads with the gains curve precisely because leading with accuracy here would
> mislead. And there's a test that pins the accuracy *below* baseline, so if someone
> later tunes toward accuracy the suite tells them they've broken the point.

### "How did you avoid data leakage?"

> Four things, and the fourth is the one specific to this dataset.
>
> First, everything — imputation, encoding, scaling — lives inside a single sklearn
> Pipeline, so the transformers only ever see training folds.
>
> Second, I select on **cross-validated** score, never on the test set. On the
> clinical model random forest scores slightly higher on my test split but the
> decision tree wins on CV; the decision tree ships and the dashboard explains why.
>
> Third, the split is stratified and seeded.
>
> Fourth — and this is where a supply chain dataset will bite you — **almost every
> lead-time column is computed from the delivery date**, which is the thing I'm
> predicting. Quote-to-delivery, PO-to-delivery, delay in days: all leak. The only
> safe one is the *scheduled* lead time, which is known at order time. I also
> excluded vendor identity and manufacturing site: with 73 vendors, several appearing
> a handful of times, the model would memorise suppliers instead of learning
> transferable structure — and it couldn't score a vendor it had never seen.
>
> There's a test that reads the persisted feature list and fails if any of those
> columns appear. Documenting leakage control in prose isn't the same as enforcing it.

### "Why is there no funnel of units through the supply chain?"

A great question, and the answer is the point.

> Because this dataset can't support one, and I checked every quantity column before
> concluding that. SCMS states `Line Item Quantity` once at order time and never
> restates it at delivery. There's no ordered-versus-received pair, no scrap
> quantity, no per-stage weight — and every line item in the file was ultimately
> delivered. A chart showing units draining between stages would have been invented,
> not measured.
>
> What *is* measurable is attrition in **timeliness**. So the funnel is value-based:
> $1.63B ordered, 84.1% arriving on or before schedule, and each band is a strictly
> tighter definition of "on time" than the one above it — which makes it monotone by
> construction rather than by luck. There's a test asserting that.
>
> The related decision I'm happier about: milestone coverage runs 74% → 44% → 100%,
> and I draw it as a **stacked bar rather than a funnel**. A funnel shape would imply
> shipments dropping out of the process, when actually a fulfilment route just has no
> vendor order to record. There's a test asserting that series is *non*-monotone, so
> if the data ever changed, the chart choice gets revisited instead of silently
> becoming a lie.

### "You reported a null result. How do you know the test wasn't just underpowered?"

If you get this, you're talking to someone good. Have the answer ready.

> By not using post-hoc power, which is the trap. Post-hoc power is computed at the
> **observed** effect size, so it's a deterministic function of the p-value — a
> genuine null mechanically returns low power no matter how large the sample. On my
> null it reads 6%, which looks damning and means nothing. Using it to judge a null
> is circular reasoning dressed up as a statistic.
>
> The right question is: given these sample sizes, **how large a gap would have shown
> up?** That's the minimum detectable effect. On the first-line-designation
> comparison — 3,294 against 7,030 shipments at an 88.5% baseline — it's 1.82
> percentage points. So I can say the measured 0.16-point gap rules out any
> difference larger than 1.82 points. That's a real, quotable bound.
>
> And it doesn't quite close the case, which I also say. A difference worth acting on
> at that baseline is 1.33 points, just *inside* the detectable limit. So the verdict
> is "no effect above 1.8 points" rather than a flat "no difference." The code has
> three distinct null verdicts for exactly that reason — collapsing them loses the
> useful part.

### "How do you choose between a t-test and a non-parametric test?"

> With a written rule rather than per-chart judgement, because on this data the
> choice changes the answer completely. Freight as a share of commodity value has a
> mean of 2,548% against a median of 10.6% — skewness of 78, driven by tiny line
> values against real shipping costs. Compared across product group, **Welch returns
> p = 0.44 and Mann-Whitney p = 6.0e-10 on the same data.** Welch is comparing means
> the outliers have rendered meaningless.
>
> So the threshold lives in the config file: absolute skew above 2 and the rank-based
> result is the one quoted. Both are always computed and shown.
>
> The subtlety I'd want to mention is that the disagreement runs in **both**
> directions. Delivery delay by era: Welch is significant at p = 8e-06, Mann-Whitney
> isn't at p = 0.45. Here *neither* is wrong. The mean delay moved from −5.0 to −7.5
> days while both medians are exactly 0, because 61% of deliveries land on their
> scheduled day. The change is in the tail, not the typical shipment. Calling Welch
> "an artefact of outliers" there would be wrong, so the code returns "both" and says
> the effect is a tail effect. That case is what convinced me a blanket rule needs a
> stated exception rather than a silent one.

### "You dropped a dataset. Why?"

> I had a third one — a 253,973-row Indian medicine catalogue — driving a
> market-structure page. I removed it, and I think that was the right call.
>
> It answered the same question worse. It carried *list* prices for products nobody in
> my dataset actually bought, and it couldn't be joined to anything. SCMS carries
> prices **actually paid**, on the same rows as the delivery performance, so I can ask
> "did the cheaper supplier also deliver on time?" and get an answer. That question is
> unanswerable with a separate catalogue.
>
> The honest cost is scale: the project went from about 264,000 rows to 10,500. If
> someone wants to see big-data handling, that's a fair criticism. My view is that two
> datasets that both earn their place beats three where one is decoration — and it made
> the whole thing explainable in 60 seconds instead of 90.

### "Everything is real data now. What did you lose by dropping the simulation?"

> Randomisation, and two analyses.
>
> The analyses were drug stability and batch-risk classification. Both needed
> per-batch storage temperature and potency, and no public dataset has that — it
> doesn't leave a pharma company. I checked Kaggle, openFDA, data.gov.in, CDSCO,
> Mendeley and Zenodo. The most promising candidate, a biopharmaceutical
> manufacturing dataset on Kaggle, turned out to be the IndPenSim simulation, so
> using it would have swapped one simulation for another.
>
> The bigger loss is randomisation. What I have now are **observational
> comparisons** — nobody assigned a shipment to a fulfilment route. So a difference
> between groups identifies where to look, not what caused it. Every comparison on
> the dashboard carries the confound that limits it, and adding a comparison
> dimension in the code *requires* a confound string — it's not optional.
>
> Five confounds are worth naming if they push: INCO term is perfectly collinear with
> fulfilment route, so they're the same split rather than two findings. Vendor lead
> time is structurally 100% missing for RDC, so no lead-time test is possible across
> that split. First-line designation perfectly predicts whether freight is available,
> so freight comparisons across it are undefined. The `Unknown` transport mode is a
> pre-2011 recording artefact with a 98.9% on-time rate — I exclude it, because
> leaving it in would make "Unknown" the best-performing mode on the dashboard. And
> transport mode is entangled with era, since Ocean is almost entirely post-2010.
>
> I'd rather explain a real confound than defend a synthetic effect.

### "Show me something you got wrong."

Have two answers. Both are true and both are better than a rehearsed weakness.

> **The one that shipped wrong numbers.** All five SCMS date columns look like dates,
> so I parsed them with one `dayfirst=True` rule. They're actually two formats:
> `%m/%d/%y` for the quote and purchase-order dates, `%d-%b-%y` for the three
> delivery dates. So `5/3/13` parsed as 3 May instead of 5 March. That produced 478
> negative lead times and 1,128 purchase orders that appeared to precede their own
> price quote. What caught it was the cross-field consistency check — not a null
> check, not a range check, because every date was populated and every date was
> valid. There are now regression tests naming both wrong figures.
>
> **The one where my premise was wrong.** I wrote a consistency rule asserting that a
> product's active-ingredient count can't exceed its pack quantity. It flagged 7,886
> Indian products as inconsistent — and every one was correct. A single vial of
> Augmentin contains amoxicillin *and* clavulanic acid. I deleted the rule rather
> than tuning its threshold, because the problem wasn't the threshold. That dataset
> now reports zero cross-field invariants, which is the honest answer for a flat
> catalogue with no derived quantities.

### "You used chi-square and a z-test. Why both?"

> They answer the same question from different angles, so agreement is a consistency
> check. The two-proportion z-test gives a directional effect size with a confidence
> interval, which is what a decision needs. Chi-square handles more than two groups —
> useful for the four transport modes — and gives Cramér's V for association
> strength.
>
> For a 2×2 table chi-square should equal the z-statistic squared, and I have a test
> asserting it. That test also guards a setting: SciPy applies Yates' continuity
> correction to 2×2 tables by default, which breaks the equality and is
> over-conservative at these sample sizes, so I disable it deliberately.
>
> I also report the minimum expected cell count on every chi-square. Below 5 the
> approximation is unreliable, and the page says so rather than quoting a p-value the
> reader will over-trust.

### "How do you know a difference is worth acting on?"

> Three bars, and the middle one is the one people skip.
>
> **Statistical significance** — the effect probably isn't noise.
> **Adequate sensitivity** — the test *could* have detected an effect worth acting on.
> Measured as a minimum detectable effect, not post-hoc power.
> **Practical significance** — the difference is big enough to be worth the cost of
> change.
>
> With a big enough sample almost any difference becomes significant, so the third
> bar does real work. There's a test asserting that a genuine 0.2-point gap on
> 500,000 rows returns **DO NOT ACT** — statistically real, too small to matter.
>
> On money: there is exactly **one assumed number** in the whole project, an SLA
> penalty per late shipment in the config file, because SCMS records no penalty or
> expediting cost. Everything it multiplies is measured. I isolated it there so it's
> obvious which figures depend on an assumption, and I don't dress it up with a net
> present value or a confidence interval — that would give an assumption the
> appearance of a measurement.

### "Why does a grade-A data quality score matter to you?"

> Because it's wrong, and the way it's wrong is instructive. Generic profiling asks
> whether cells are populated, unique and in range. It cannot ask whether a populated
> cell *means* anything. SCMS scores 99.3% complete and grade A while 55% of its
> purchase-order dates and 40% of its freight costs are unusable text.
>
> The demonstration I'm proudest of is that **parsing the file correctly lowers its
> score by 1.70 points**, because unparseable strings count as complete and the nulls
> that replace them don't. That negative number means the generic metric *rewards* a
> file for holding non-null garbage. There's a test asserting the sign stays negative
> — if it ever turned positive it would mean my parser had started imputing instead
> of nulling, which is a silent and serious regression.
>
> The Indian dataset moves the other way, which is the control: there parsing
> genuinely recovers information — a price column stored as text becomes a number,
> free-text pack labels become a form and a quantity — so the score goes up. And
> drug200 is flat at zero, because it's published clean and I didn't manufacture a
> difference to make the comparison look better.

### "Tell me about the pricing analysis."

This is the answer that shows you learned something and then reused it.

> SCMS is usually treated as a logistics file, but every line also carries the
> molecule, brand, dosage, factory and the price actually paid. So I asked whether the
> programme paid a consistent price for the same thing.
>
> Pooled across all ten years, identical products — same molecule, same strength, same
> dosage form — show a **5.0x** price spread. That looks like a scandal. It isn't.
> Antiretroviral prices collapsed over that decade: Efavirenz 600mg fell **80%**, from
> $0.56 a unit to $0.11. So a pooled comparison is measuring *when* you bought, not
> *who* from. Within a single year the spread is **2.5x** — pooling inflated it by
> exactly a factor of two.
>
> **That's the same mistake I'd already found in the delivery data**, where a pooled
> 11.9-point service gap was 1.9 points before 2011 and 20.5 after. Finding the
> identical trap twice on two unrelated questions is why every comparison in the
> project is stratified by year before it gets quoted.
>
> Then I checked whether the remaining 2.5x was just noise. It isn't — it's generic
> versus originator. Where both were bought in the same year, branded costs a median
> **2.1x** more across 41 product-years. Nevirapine 200mg is the clearest: three Indian
> factories at $0.050 a unit, Boehringer's Viramune at $0.335. **6.7x for the same
> molecule at the same strength in the same year.**

**The follow-up you should invite:** *"So they were overpaying?"*

> I can't say that, and this is the important part. The expensive supplier delivered
> **100% on time**; the cheapest was at 86%. The premium buys something measurable. And
> nothing in the file records freight terms, volume commitments, urgency, or whether a
> product was registered in the destination country — all of which legitimately move
> price.
>
> So the honest statement is: the choice exists, here's what each side costs on both
> price and service, and whether 6.7x is worth 14 points of reliability is a
> procurement judgement rather than something my analysis settles.

**Why it matters commercially:** 63% of priced spend sits in five products. So this
isn't a 92-product renegotiation — it's five conversations.

### "What was the hardest technical problem?"

> Deciding what a missing value *means*, before deciding what to do about it.
>
> The SCMS purchase-order date column has 5,404 blanks out of 10,324 — 52%. Every
> instinct and every tutorial says impute, and my own code did: I'd written a generic
> cleaning layer that mode-imputed every text column, for every table.
>
> But those aren't gaps. `N/A - From RDC` records that no purchase order *existed*,
> because the goods came from distribution centre stock. Imputing would have
> fabricated 5,404 purchase orders, and every lead-time statistic downstream would
> have been computed over shipments that never had the milestone being measured.
>
> The fix was structural rather than a tweak. I deleted the generic cleaning layer
> entirely and replaced it with per-dataset parsing that attaches a reason code to
> every value it can't use — `parsed`, `structural`, `missing`, `cross_reference`.
> Then every function that reports a lead time also reports the denominator it
> actually used. That's why the dashboard says "measured on 44% of line items" instead
> of quoting one number.
>
> The lesson that generalised: **a cleaning step that can't tell you why a value was
> absent shouldn't be allowed to fill it in.**

### "Why Streamlit and not a BI tool?"

> Because a lot of what this does isn't expressible in a BI tool. The statistical
> testing page runs power analysis and minimum-detectable-effect calculations live.
> The comparison engine chooses between a t-test and a rank test based on measured
> skew. The ML page renders calibration and gains curves from persisted model
> metadata. Those need Python in the request path.
>
> If the deliverable were fixed reporting on a warehouse, I'd use Power BI or Tableau
> and it'd be the right call.

### "How would you productionise this?"

> Four things, roughly in order.
>
> **Orchestration** — the CLI scripts become scheduled tasks in Airflow or Dagster,
> with the data quality check as a **gate that fails the run**, not a report someone
> reads later. That matters more here than usual: the date-parsing bug would have
> been caught by the consistency check on day one if it had been a gate.
>
> **A warehouse** — the CSV cache becomes Postgres or BigQuery. The loader is already
> the single entry point, so that's one module.
>
> **Model registry and monitoring** — MLflow for versioning, plus drift detection on
> input distributions. This matters specifically because my own analysis shows the
> network changed materially in 2011; a model trained across that boundary is
> averaging two regimes, and a fresh one would need the same check.
>
> **Serving** — the prediction interface becomes a FastAPI endpoint. It's already a
> clean function returning a dict, so that's thin.

### "What would you do differently?"

> Two things, and the second is a real weakness in what I'm claiming.
>
> **I'd resolve the freight cross-references instead of discarding them.** 2,445 rows
> carry freight as `See DN-304 (ID#:10589)` — and that embedded ID is a resolvable
> reference to another row in the same file. So those values are *recoverable* by a
> self-join, not lost. I classified them correctly as `cross_reference` and excluded
> them, which is honest but leaves 24% of the freight data on the table.
>
> **I'd try to identify what changed in 2011.** My headline finding localises a
> service collapse in time and I stop there, because the dataset can't explain it.
> What I'd want is a difference-in-differences design using the country mix — if the
> RDC channel's destination portfolio shifted after 2010, the "collapse" might partly
> be composition rather than degradation. I state that limitation rather than
> resolving it, and it's the single most valuable open question in the project.

### "How did you decide which comparisons to run?"

> I let the data pick, then constrained it. The pipeline analysis shows where value
> arrives late; each dimension that could explain it becomes a comparison. But two
> rules keep it from becoming p-hacking.
>
> First, a group needs at least 30 shipments to be reported at all — below that a
> single late delivery swings the rate by several points.
>
> Second, and more importantly, adding a comparison dimension in the code requires
> supplying its **confound**. It's a required field, not a comment. That's a
> deliberate design choice: it means nobody, including future me, can add a
> comparison to the dashboard without having thought about what limits it.
>
> The catalogue is sorted by effect size but the recommendations aren't. Destination
> region shows the widest gap — 12.7 points, Asia against West & Central Africa — and
> it's the least actionable, because the programme doesn't choose where medicines are
> needed. A dashboard that ranks findings purely by effect size points an
> organisation at things it can't change.

---

## 5. Demo path (5 minutes)

Nine pages, but you only need five or six. Don't wander.

1. **Home** — "Three real datasets, no simulation." *Ten seconds.*
2. **Delivery Pipeline** — "$259M arrives late. And there's no unit funnel here on
   purpose — the dataset states quantity once and never restates it, so an attrition
   chart would be invented." Show the value funnel and the lateness thresholds.
3. **Statistical Testing** — the centrepiece. Show the RDC comparison, then scroll to
   the stratified chart: "I nearly reported the pooled 11.9 points. Stratified by era
   it's 1.9 then 20.5 — effect modification, and the page is explicit that it isn't a
   sign reversal." Then the null-result section: "and here's how to report a
   non-result properly, using a minimum detectable effect rather than post-hoc
   power."
4. **ML Models → Late Delivery** — "Accuracy below baseline. Here's why I ship it
   anyway." Show the gains curve.
5. **Product & Pricing** — "Same file, different question. The pooled price spread is
   5x, the honest one is 2.5x, and the gap is a decade of falling prices — the same
   trap as slide 3." Then the Nevirapine factory table: $0.050 generic against $0.335
   branded, but the expensive one delivered 100% on time.
6. **Data Quality** — "The generic profiler grades this file an A, and it's wrong.
   Parsing it correctly makes the score go *down*."

If there's time, **Insights** is the best closer — every finding carries a "does not
establish" column.

Let questions redirect you. Curiosity is a buying signal.

## 6. Résumé bullets

Grounded in what the project actually does:

> - Analysed **10,324 real USAID pharmaceutical shipments** (43 countries, 73
>   vendors, $1.63B commodity value) in **Python (Pandas, NumPy, Plotly)**, building
>   an order-to-delivery pipeline analysis that quantified **$259M of commodity value
>   delivered late** and identified an internal distribution channel — not an external
>   supplier — as the weakest link; used **stratified analysis** to show the apparent
>   11.9pp service gap masked a post-2010 collapse from 93.4% to 73.9% on-time (a
>   ten-fold interaction with era that the pooled figure hid entirely).
> - Built two end-to-end **machine learning pipelines** on real data — drug
>   classification (Decision Tree, balanced class weights, 5 classes, 0.988 macro F1)
>   and late-delivery prediction (XGBoost, **ROC AUC 0.85**) — with leakage-controlled
>   feature engineering, grid-searched selection across Decision Tree, Random Forest
>   and XGBoost under stratified cross-validation, and gains-curve deployment
>   (**top-20% risk targeting captures 63% of late deliveries, 3.2× lift**) after
>   demonstrating that accuracy fell below the majority-class baseline.
> - Designed a **statistical testing framework** (SciPy, statsmodels) using
>   **two-proportion z-tests, chi-square, Welch's t-test and Mann-Whitney U** with
>   power analysis, minimum-detectable-effect bounds for null results, and
>   practical-significance gating.
> - Built a **procurement pricing analysis** on the same shipment data, showing the
>   pooled price spread for identical products (5.0×) was **double** the within-year
>   figure (2.5×) once a decade-long 80% price decline was controlled for, and
>   quantifying a **2.1× median generic-to-branded premium** across 41 product-years —
>   with 63% of spend concentrated in five products. Delivered through a **9-page
>   Streamlit dashboard** backed by **141 automated tests**.

## 7. Things not to do

**Don't lead with accuracy on either model.** On the clinical model the label is a
pure function of the features, so 98% is below what's attainable. On the
late-delivery model accuracy is literally worse than doing nothing. Lead with the
stratified RDC finding and the gains curve.

**Don't claim causation.** These are observational comparisons. Say "observational"
out loud once, early — it costs you nothing and buys a lot of credibility with anyone
who was about to test you on it.

**Don't call the RDC finding Simpson's paradox.** It's effect modification — the gap
never reverses sign. The distinction is small, the code already makes it, and getting
it right in conversation is free. Overstating it is the one place in this project
where a sharp interviewer could catch you out.

**Don't list the tech stack unprompted.** "I used Pandas, NumPy, scikit-learn,
XGBoost…" tells them nothing. They can read the README. Talk about the *problem*.

**Don't oversell the data quality score.** The interesting thing is that the score is
*wrong*, not that it's high. If you present grade A as an achievement you've missed
your own finding.

**Don't claim ROI.** You have gross benefit against one assumed penalty rate. Say
"gross benefit, one assumed SLA rate, everything else measured" and you sound
rigorous instead of hand-wavy.

---

## 8. If you're asked to go deeper

Places where you have real depth, if the conversation goes technical:

- **Reason-coded provenance** — the four codes (`parsed`, `structural`, `missing`,
  `cross_reference`), why structural absence is categorically different from a gap,
  and how every downstream statistic reports its own denominator as a result.
- **Minimum detectable effect vs post-hoc power** — why the latter is circular on a
  null, how the MDE inverts the arcsine transform around the baseline rate to express
  a detectable Cohen's h as a rate difference, and the three distinct null verdicts
  the code produces.
- **Effect modification versus Simpson's paradox** — why a sign flip is the defining
  criterion for the latter, why both make a pooled average misleading, and how the
  detector distinguishes them (`sign_flip` versus a gap range exceeding half the
  pooled gap).
- **Both directions of Welch/Mann-Whitney disagreement** — outlier masking versus a
  tail-only shift, and why a blanket "use the rank test when skewed" rule needs a
  stated exception rather than a silent one.
- **Gains and lift as the deployment metric** — why a ranking metric is the right
  frame for a 88.5/11.5 split, how the curve is computed on the reproduced test split
  rather than all rows (that was a leakage bug I fixed), and what capacity assumption
  the 20% figure encodes.
- **Probability calibration** — why a tree grown to pure leaves produces degenerate
  probabilities, what that breaks operationally, and the two standard fixes.
- **Why the 80% Pareto line is drawn on one chart and omitted elsewhere** — on SCMS
  product value the cumulative curve genuinely reaches 94% by the fifteenth product,
  so the reference line is meaningful; on a flat distribution it sits far above the
  curve and makes the chart look broken. The builder takes it as an optional argument
  for exactly that reason.
- **Max-over-min as a spread measure** — why it is deliberately crude, why the median
  across products is quoted rather than any single ratio, and why the factory-level
  view matters (a spread is only actionable if it maps onto suppliers you can choose
  between).
- **Configuration design** — every threshold in one auditable YAML file, including the
  two that decide *how a statistic gets reported* (`skew_limit`, `min_group_size`), so
  that choice is written down once instead of made per chart.
