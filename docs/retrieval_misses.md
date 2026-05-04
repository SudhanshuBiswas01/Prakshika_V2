# Retrieval Misses — Stage 2 Diagnosis

## Evaluation Summary

| # | Question | Top-1 Chunk | Score | Contains Answer? |
|---|----------|-------------|-------|------------------|
| 1 | Newton's universal law of gravitation? | ch1_p11_0002 | 0.549 | PARTIAL — lists topics but doesn't state the law |
| 2 | Value of gravitational constant G? | ch1_p18_0025 | 0.590 | NO — just asks "what if G was twice as large" |
| 3 | Acceleration due to gravity vs height? | ch1_p17_0018 | 0.534 | NO — discusses velocity of stone, not g vs height |
| 4 | Difference between mass and weight? | ch1_p19_0033 | 0.622 | YES |
| 5 | Kepler's laws of planetary motion? | ch1_p13_0006 | 0.777 | YES |
| 6 | Mendeleev's periodic table basis? | ch2_p28_0064 | 0.773 | YES |
| 7 | Merits of modern periodic table? | ch2_p29_0065 | 0.664 | NO — talks about Mendeleev's table merits, not modern |
| 8 | Atomic radius across a period? | ch2_p35_0101 | 0.724 | YES |
| 9 | Electronic config of group 1? | ch2_p32_0080 | 0.816 | YES |
| 10 | Law of octaves (Newlands)? | ch2_p27_0061 | 0.677 | YES |

**Hit rate: 6/10 YES, 1 PARTIAL, 3 NO**

---

## Miss Diagnoses

### Miss 1: "Value of gravitational constant G" → ch1_p18_0025
The top-1 chunk is a **question** about G, not the passage that defines its value. The actual value (G = 6.67 × 10⁻¹¹ Nm²/kg²) is in ch1_p14 (the derivation page). The question chunk mentions "G" heavily and gets a high embedding similarity, but has no factual content. **Root cause: question_or_exercise chunks pollute retrieval with keyword-dense but answer-poor content.**

### Miss 2: "How does acceleration due to gravity change with height" → ch1_p17_0018
Top-1 discusses velocity of a stone and free fall — tangentially related but not the right chunk. The actual content about g varying with height is on page 18-19 (the g = GM/R² derivation and the table of g at different altitudes). **Root cause: semantic similarity captures "gravity" and "change" but not the specific concept of g-vs-height. The correct chunk uses formulas and a table, which embeds poorly as plain text.**

### Miss 3: "Merits of modern periodic table" → ch2_p29_0065
Top-1 chunk discusses merits of **Mendeleev's** table, not the **modern** one. The word "periodic table" appears in both contexts and the embedding can't distinguish between them. **Root cause: synonym overlap between Mendeleev's and modern periodic table contexts — the model sees "periodic table" + "merits" and picks the first match regardless of "modern" qualifier.**
