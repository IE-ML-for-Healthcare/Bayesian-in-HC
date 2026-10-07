# Codex brief: align Naive Bayes opioid notebook with threshold 0.22 and updated findings

## Objective

Update `04_naive_bayes_opioid_demo_v2.ipynb` so the full notebook is internally consistent with the decisions already made in class and in the latest repository version

Keep the notebook simple, concise, clinically intuitive, and appropriate for a bachelor-level AI for Healthcare class

Do not expand the scope beyond the alignment and teaching improvements described here

## Repository state audited

- Repository: `IE-ML-for-Healthcare/Bayesian-in-HC`
- Branch: `main`
- Audited HEAD: `11319d8192c7d4dac2e11f2cde88cf19b8a465a5`
- Target notebook: `04_naive_bayes_opioid_demo_v2.ipynb`
- No `AGENTS.md` or `CLAUDE.md` exists in the GitHub repository at the audited state, but re-check the local working tree before editing
- If local `PROFESSOR_NOTES_04_naive_bayes.md` exists, treat it as private teaching notes and keep it untracked

## Decisions that are already settled

- Use the previously established clinical classification threshold of `0.22`
- Do not re-optimize or re-teach threshold selection in this notebook
- Keep `fit_prior=True` in the first Naive Bayes model so it learns the approximately 18% training prevalence
- Keep the external 5% prior comparison later in the dedicated prior section
- Student-facing terminology must call `average_precision_score` **PR AUC**, consistent with the previous exercise
- The early `Low_inc` baseline uses the raw binary `Low_inc` flag as a score and does not fit a separate model
- Section 9 does fit a separate one-feature Bernoulli Naive Bayes model using `Low_inc` only for repeated cross-validation
- Do not change the dataset, split, random seeds, model family, or core teaching sequence unless required for consistency

## Current validated Section 3 results

After the latest code changes, the intended evaluation results are

```text
--- Naive Bayes (all 20 features) ---
ROC AUC: 0.711  (95% CI 0.62 to 0.80)
PR AUC:  0.353  (no-skill baseline = 0.180)
At threshold 0.22 -> precision 0.407, recall 0.733

--- Baseline: Low_inc flag alone ---
ROC AUC: 0.745
PR AUC:  0.340  (no-skill baseline = 0.180)
```

At threshold 0.22 this corresponds to

- 33 true positives
- 12 false negatives
- 48 false positives
- 157 true negatives
- 81 patients flagged

## Required fixes

### 1. Re-run and save the notebook outputs

The evaluation cell has been edited but its stored output is stale

The current saved output does not yet show the new `Low_inc` PR AUC line even though the code calculates it

Run the notebook top to bottom in a clean kernel after all edits and save the fresh outputs

### 2. Align Section 3 metric terminology

The Section 3 Markdown still introduces `Average precision (AP, the area under the precision-recall curve)`

Change the student-facing wording to **PR AUC** and explain briefly that this notebook reports it using `average_precision_score`

Keep the explanation simple

Recommended framing

> **PR AUC**: focuses on how well the model identifies OD patients when OD is the minority class. Here we report it using `average_precision_score`. The no-skill baseline equals the OD prevalence, about 18%

Do not display `AP`, `Average precision`, or `Avg precision` as the student-facing metric name elsewhere in the notebook

Internal scikit-learn scoring names such as `average_precision` may remain in code where required

### 3. Replace the current Critical Reading interpretation

The current statement that the 20-feature model simply does not beat `Low_inc` is too strong because the metrics split

Use this interpretation or a very close concise equivalent

```markdown
**Critical reading**
- The full 20-feature model does **not clearly outperform** the single `Low_inc` flag. `Low_inc` does slightly better on ROC AUC (0.745 vs 0.711), while the full model does slightly better on PR AUC (0.353 vs 0.340). The other 19 features therefore add little clear value on this test split. We test this more reliably with cross-validation later in section 9
- Both PR AUC scores are well above the **0.18 no-skill baseline**, so both contain useful signal. However, much of the model's predictive power appears to come from income alone
- At our established **0.22 threshold**, the full model catches about **73% of OD patients**. It still misses 12 of 45 OD cases, and only about **41% of flagged patients actually have OD**, so better case detection comes with many false alarms
```

Do not say that most features `add noise` unless the repeated cross-validation results directly support that stronger claim

Prefer `add little useful signal` or `add little clear value`

### 4. Fix the Section 4 prior comparison, which is currently logically inconsistent

This is the highest-priority code bug in the current notebook

Current problem

```python
threshold = 0.22
y_pred = (y_proba >= threshold).astype(int)
...
ext_pred = nb_external_prior.predict(X_test)
...
print("--- At threshold 0.5 (changed by the prior) ---")
for name, pred in [("Learned prior", y_pred), ("5% prior", ext_pred)]:
```

`y_pred` uses threshold 0.22 but `ext_pred = .predict()` uses the classifier default 0.5 threshold, while the output header says both use 0.5

Fix it so the learned-prior and 5% prior models are compared at the **same fixed threshold of 0.22**

Preferred implementation pattern

```python
ext_pred = (ext_proba >= threshold).astype(int)
```

Use the existing `threshold` variable rather than hardcoding another value

Change the output header accordingly

```python
print(f"\n--- At threshold {threshold:.2f} (changed by the prior) ---")
```

Then re-run and update the following Markdown with the actual new counts, precision, and recall

Do not preserve the old statement that only 2 patients crossed 0.5 because 0.5 is no longer the comparison threshold

### 5. Align PR AUC terminology in the prior section

Current code prints

```python
print(f"AP       learned: ...")
```

Change the displayed label to `PR AUC`

Current Markdown says `AUC/AP only depend on the order`

Change this to `ROC AUC and PR AUC` or similarly explicit wording

The teaching point remains unchanged: changing only the prior changes absolute probabilities but not patient ranking

### 6. Keep the prior-shift demonstration, but align it with 0.22 where useful

Section 4.2 is about the mechanics of prior shift, not choosing a new clinical threshold, so it can remain

If the threshold-mapping examples currently use `[0.05, 0.10, 0.20]`, consider replacing `0.20` with `0.22` so the established course threshold appears in the demonstration

Do not turn this into another threshold-optimization exercise

### 7. Update Section 9 model comparison to report both ROC AUC and PR AUC consistently

The current repeated cross-validation table displays `Avg precision`

Rename the displayed column to **PR AUC** while keeping `average_precision` internally for scikit-learn scoring

The takeaways must compare both ROC AUC and PR AUC, not only generic `AUC`

Do not declare a winner from one metric or one split

The intended high-level conclusion is

- The 20-feature model does not show a clear overall improvement over `Low_inc` alone
- The extra clinical features add little predictive value beyond income in this dataset
- Differences among the models are small relative to cross-validation variability

After re-running, use the actual cross-validation values to support that wording

### 8. Refactor Section 10 so it does not re-teach threshold selection

The current Section 10 reintroduces the harm-ratio threshold formula, a table of possible thresholds, natural frequencies at several thresholds, and a decision curve across many thresholds

This conflicts with the settled teaching decision that threshold selection was already covered in the previous notebook and this notebook should use `0.22`

Refactor Section 10 around the question

> What does our already chosen 0.22 threshold mean in practice?

Recommended changes

- Replace the threshold-selection derivation and table with one brief reminder that `0.22` was established in the previous exercise
- Do not optimize the threshold again here
- Show natural frequencies at `0.22` as the main operational interpretation
- If the decision curve is retained, frame it as a **clinical utility check**, not as a method for selecting a new threshold
- Add a vertical reference at 22% to the decision curve if retained
- Update the accompanying interpretation to focus on whether the model adds value at the established 22% threshold
- Avoid statements that direct students to choose another threshold

Preserve useful decision-curve content if it still supports the teaching goal, but keep it secondary to the fixed 0.22 decision rule

### 9. Align the equity section with the established threshold

The current equity code hardcodes

```python
t = 0.20
```

Replace this with the existing notebook threshold variable

```python
t = threshold
```

Then re-run the table and update all narrative percentages and counts that currently describe results at 20%

Do not leave any prose saying `at a 20% threshold` after the code uses 0.22

### 10. Align the Summary and student questions

Current Summary issues

- `Performance | AUC ≈ 0.70, no better than income alone` is too vague because ROC AUC and PR AUC tell slightly different stories
- `Decisions | Threshold comes from the harm ratio, not 0.5` reopens threshold selection even though 0.22 is already inherited from the previous exercise

Update the Summary so it reflects

- ROC AUC and PR AUC both show useful signal, but the 20-feature model adds little clear value beyond income
- The notebook uses the previously established 0.22 clinical threshold
- Calibration and clinical utility remain separate questions from ranking

Review the student questions as well

- If Question 3 asks what happens to `AUC`, specify whether students should inspect ROC AUC, PR AUC, or both
- Reframe the deployment question around performance and utility at the established 0.22 threshold rather than asking students to choose another threshold

### 11. Add a concise `How could we improve this model?` section near the end

Add this immediately before the final Summary so it follows naturally from the model comparison, calibration, and equity findings

Keep it short and practical

Suggested content

```markdown
## How could we improve this model?

1. **Use better clinical features**
   Add predictors with stronger clinical information, such as dose, duration, previous overdose, mental-health history, and concurrent sedative use. The current non-income features contain little predictive signal

2. **Reduce redundant or weak features**
   Correlated features can make Naive Bayes double-count evidence. Remove or combine features only when cross-validation shows that the change improves out-of-sample performance

3. **Use better-calibrated probabilities**
   Logistic regression already performs similarly and is better calibrated here. Naive Bayes can also be recalibrated, but recalibration improves the probabilities, not the ranking

**Key lesson:** When better algorithms do not improve performance, the problem is often the data and features, not the model
```

Keep this section conceptual

Do not add a new feature-engineering experiment, hyperparameter search, or optimization workflow

### 12. Update the private professor notes if they exist locally

`PROFESSOR_NOTES_04_naive_bayes.md` is intentionally not tracked in Git

If it exists in the local project, align at least these items with the final notebook

- Cell 10 evaluation outputs and terminology
- Cell 11 Critical Reading notes
- Section 4 prior comparison outputs at threshold 0.22
- Any remaining references to AP instead of PR AUC
- Section 9 model-comparison wording if PR AUC changes the interpretation
- Section 10 teaching notes so threshold selection is not re-taught
- Equity results after switching from 0.20 to 0.22
- Final section numbering after adding `How could we improve this model?`

Do not commit the professor notes unless the repository policy has changed

## Validation requirements

After editing

1. Restart the kernel and execute the notebook from top to bottom
2. Save all fresh outputs in the notebook
3. Confirm every code cell runs without error
4. Confirm the initial evaluation reproduces approximately
   - ROC AUC 0.711 with 95% CI 0.62 to 0.80
   - PR AUC 0.353 for all 20 features
   - Precision 0.407 and recall 0.733 at threshold 0.22
   - `Low_inc` ROC AUC 0.745
   - `Low_inc` PR AUC 0.340
5. Confirm the confusion matrix at 0.22 contains TP 33, FN 12, FP 48, TN 157
6. Confirm the learned-prior and 5% prior comparison uses the same threshold variable on both models
7. Recompute and update every dependent number after that prior-section fix
8. Confirm Section 9 displays `PR AUC`, not `Avg precision`
9. Confirm the equity section uses `threshold`, not a hardcoded `0.20`
10. Search the notebook for `AP`, `Average precision`, `Avg precision`, `threshold 0.5`, `at 20%`, `t = 0.20`, and similar stale wording, then inspect each remaining match manually
11. Confirm any remaining 0.5 or alternative thresholds are intentional analytical references, not accidental prediction defaults or claims about the chosen clinical threshold
12. Confirm all section references and numbering remain correct after inserting the improvement section
13. Keep the notebook concise and bachelor-level, with no unnecessary mathematical or modelling detours

## Definition of done

The notebook should tell one consistent story

- Learn the approximately 18% prior from the training cohort
- Evaluate the model using ROC AUC, PR AUC, and the inherited 0.22 classification threshold
- Show that `Low_inc` alone captures much of the available predictive signal without overstating which model wins
- Demonstrate how a different deployment prior shifts probabilities while comparing models fairly at the same threshold
- Show that calibration, model choice, clinical utility, and equity are separate questions
- Use 0.22 consistently for operational and equity interpretations
- Explain that meaningful improvement is more likely to come from better features, less redundancy, and better calibration than from simply trying another algorithm
