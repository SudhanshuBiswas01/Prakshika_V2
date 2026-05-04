# Stage 5 Targeted Fix: Metadata Filtering

## The Problem
In the Stage 4 evaluation, Question 2 ("What is the value of the gravitational constant G?") was falsely refused by the LLM. 

**Diagnosis**: The LLM returned "I don't have that in my study materials." Looking at the retrieval logs, the system fetched chunk `ch1_p18_0025`, which was a `question_or_exercise` chunk that mentioned G ("What would happen if the value of G was twice as large?"). 

Because question chunks contain very dense keywords (like "value", "G", "gravitational", "constant"), they often have high semantic similarity to the user's query. However, they almost never contain the *answer* to the question. The actual definition and value of G was in a `prose` chunk on page 14, but it was crowded out of the top-k results by these question chunks.

**Classification**: `synonym_mismatch` (the model matched "value of G" to a question asking about it, rather than the definition).

## The Fix
I implemented a metadata filter in `retrieve_v2`. The strategy:
1. Retrieve `k + 5` chunks initially to cast a wider net.
2. Check the `content_type` metadata of each retrieved chunk.
3. Apply a penalty multiplier (`0.7x`) to the cosine similarity score of any chunk marked as `question_or_exercise`.
4. Re-sort the list by the adjusted score and return the top `k` chunks.

## The Result
This effectively deprioritizes question chunks, allowing the dense `prose` chunks containing actual facts and definitions to rise to the top. This resolves the false refusal for Question 2 without needing to change the prompt or re-embed the entire dataset.
