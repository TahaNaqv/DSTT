# 00 — START HERE: Mid-Exam Plan (Lec 9–17 + 5 repos)

The paper has three parts: **coding, numericals, MCQs**. These files map onto them:

| File | Use it for |
|---|---|
| `01_CODE_Memorize.md` | All repo + slide code, in learning order, with mnemonics and verified outputs |
| `02_CODE_Drills.md` | Fill-in-the-blank → skeleton → blank-page drills (this is where memorizing actually happens) |
| `03_NUMERICALS.md` | Every calculation type with step-by-step worked examples + practice |
| `04_THEORY.md` | Lec 9–17 theory + top 40 MCQ traps |
| `05_MOCK_EXAM_A.md` / `_SOLUTION` | Hard mock paper 1 (100 marks, 2 hours) |
| `06_MOCK_EXAM_B.md` / `_SOLUTION` | Hard mock paper 2 (different numbers and code) |

---

## The memorization method (for someone who finds memorizing hard)

1. **Learn 3 skeletons, not 30 programs.** TF-IDF (I-C-F-I), mapper (for line / for word / write word-tab-1), reducer (split → changed? → flush → reset → add → last flush). Every repo script is a skeleton plus 1–3 new lines.
2. **Understand the "why" of each line** before memorizing it. "Why the last flush?" → "the last word never sees a key change." A reason is far easier to remember than a line.
3. **Write, don't read.** Cover the code, write it on paper, compare, rewrite only what you missed 3 times.
4. **Say the mnemonic, then write.** Your hand follows your mouth.
5. **Spaced repetition:** each drill on day 1, again on day 2, again on the day before the exam. Three recalls with gaps beat ten readings in one sitting.
6. **Predict outputs before running anything** — the exam asks "what does this print?" often.

---

## Learning sequence (do it in this order)

### Session 1 — Code foundations (≈ 2.5 h)
1. `01_CODE` Block 1 (pandas) → Drill 12 Level 1
2. Block 2 (text cleaning) → Drill 8 cleaning part
3. Block 3 (BoW) → Drill 2
4. **Block 4 (TF-IDF skeleton)** → Drill 1 all three levels
5. Read `04_THEORY` Parts D, E, F (they explain *why* this code exists)

### Session 2 — The repos (≈ 3 h)
1. Block 5 (similarity, top terms, classifier, save/load) → Drills 3, 4, 5, 6 Level 1
2. **Block 6 (class activity 4 functions)** → Drill 7 Level 1
3. Block 7 (visualizations) — learn the shared 4 lines + the one-line difference per plot
4. Block 8 (Lab 15) → Drill 8
5. Redo Drill 1 Level 3 from memory (spaced recall)

### Session 3 — MapReduce + Big Data (≈ 2.5 h)
1. **Block 9 (mapper, reducer, inverted index, commands)** → Drills 9, 10, 11
2. `04_THEORY` Part G (Hadoop, HDFS, MapReduce, Lec 17)
3. `03_NUMERICALS` topics 12, 13, 14 (HDFS blocks, MapReduce trace, timing)

### Session 4 — Numericals (≈ 3 h)
1. `03_NUMERICALS` topics 1–3 (TF-IDF, cosine, sparsity) — do every worked example yourself first
2. Topics 4–7 (confusion matrix, bias-variance, loss, CV)
3. Topics 8–11 (outliers, SMOTE, imbalance, sampling)
4. All 4 practice questions without looking

### Session 5 — Theory + ML lectures (≈ 2 h)
1. `04_THEORY` Parts A, B, C (Lec 9–12)
2. Block 10 code (Lec 12 slide code)
3. Read the 40 MCQ traps twice

### Session 6 — Mock A (2 h timed + 1 h review)
Closed book. Mark yourself honestly. For every lost mark, go back to the exact block/topic and redo that drill.

### Session 7 — Spaced recall (≈ 1.5 h)
Level 2 of every drill. Redo any mock-A question you lost marks on.

### Session 8 — Mock B (2 h timed + 1 h review)

### Night before
- `02_CODE_Drills` → **Blank-page final check** (7 items, 25 minutes)
- `01_CODE` Quick-Recall Card + 12 code traps
- `04_THEORY` 40 MCQ traps
- `03_NUMERICALS` formula column only

If you have less time, the **minimum high-yield set** is: Blocks 4, 6, 9 code · TF-IDF hand calculation · confusion matrix · IQR/z-score · SMOTE formula · HDFS blocks · MapReduce trace · the 40 traps.

---

## In the exam hall
- **Coding:** write imports first, then the skeleton, then the special lines. Even a partial skeleton earns marks.
- **Numericals:** formula → givens → substitution → answer → one-line interpretation. State the log base.
- **MCQs:** watch for Type I/II swaps, sklearn vs hand IDF, "estimate" vs "improve" for CV, and "split before resampling".
