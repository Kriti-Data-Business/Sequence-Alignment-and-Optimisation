

# Sequence-Alignment-and-Optimisation

This repository presents a comprehensive study and implementation of three core sequence alignment algorithms used in bioinformatics:

* **Needleman-Wunsch** (Global Alignment)
* **Smith-Waterman** (Local Alignment)
* **Hirschberg’s Algorithm** (Space-Optimized Alignment)

The project focuses on dynamic programming design paradigms, memory optimization, and benchmarking across diverse genomic datasets. Real-world genomic sequences (up to 1000 base pairs) are used to showcase biological relevance and computational efficiency.

## Key Contributions

* Complete implementation of three sequence alignment algorithms with `O(mn)` time complexity.
* Space-optimized solution achieving `O(min(m, n))` space using divide-and-conquer (Hirschberg).
* Empirical validation of time and space complexity bounds via benchmarking.
* Real-world genomic datasets and synthetic test cases included.
* Performance visualization and comparative analysis tools provided in notebooks.

---

## 1. Introduction and Problem Context

### 1.1 Biological Significance

Sequence alignment is a cornerstone of computational biology and genomics. It enables:

* **Genome Assembly**: Reconstruct genomes from DNA fragments.
* **Phylogenetic Analysis**: Understand evolutionary relationships.
* **Drug Discovery**: Identify protein motifs and binding regions.
* **Personalized Medicine**: Compare patient genome variants to references.

### 1.2 Computational Challenges

* Human genome contains \~3.2 billion base pairs.
* Classical algorithms require `O(mn)` space — infeasible for long sequences.
* Space and time trade-offs needed for scalable implementations.
* Optimized algorithms needed for real-time applications.

### 1.3 Course Learning Outcomes (COSC3119)

This project aligns with:

* **CLO1**: Apply and analyze algorithms and data structures.
* **CLO2**: Theoretical and empirical complexity evaluation.
* **CLO3**: Advanced problem solving using space-efficient paradigms.

---

## 2. Literature Review and Background

### 2.1 Classical Algorithms

* **Needleman-Wunsch (1970)**:

  * Global alignment
  * `O(mn)` time and space
  * Suitable for overall similarity

* **Smith-Waterman (1981)**:

  * Local alignment
  * Detects regions of high local similarity
  * Same complexity as Needleman-Wunsch

### 2.2 Space Optimization

* **Hirschberg’s Algorithm (1975)**:

  * Reduces space to `O(min(m,n))`
  * Divide-and-conquer approach
  * Retains optimality, ideal for large genomes

### 2.3 Modern Bioinformatics Tools

* **BLAST**: Heuristic for local alignment
* **BWA/Bowtie**: Suffix arrays for genome mapping
* **Minimap2**: Long-read alignment with chaining

---

## 3. Methodology and Implementation

### 3.1 Core Design

* Object-oriented Python implementation via `SequenceAligner` class.
* Modular structure for alignment algorithms.
* Efficient memory usage and clear traceback.
* Input validation and error handling.

#### Scoring Configuration:

```python
match_score = 2
mismatch_penalty = -1
gap_penalty = -2
```

#### Scoring Matrix (example):

```python
score_matrix: np.ndarray[int, (m+1, n+1)]
```

---

### 3.2 Algorithm Implementations

#### 3.2.1 Needleman-Wunsch (Global Alignment)

Recurrence:

```python
score[i][j] = max(
    score[i-1][j-1] + match/mismatch,
    score[i-1][j] + gap_penalty,
    score[i][j-1] + gap_penalty
)
```

* Guarantees optimal global alignment.
* Full traceback and matrix visualization.

#### 3.2.2 Smith-Waterman (Local Alignment)

Modified Recurrence:

```python
score[i][j] = max(
    0,
    score[i-1][j-1] + match/mismatch,
    score[i-1][j] + gap_penalty,
    score[i][j-1] + gap_penalty
)
```

* Finds highest-scoring local subsequence.
* Starts traceback from max score.

#### 3.2.3 Hirschberg’s Algorithm (Space Optimized)

Divide-and-conquer steps:

1. Compute middle row using linear space.
2. Find best split point.
3. Recursively align subsequences.
4. Concatenate final alignment.

* Uses only two rows at a time.
* Achieves `O(min(m, n))` space.

---

### 3.3 Benchmarking

* Benchmark scripts in `notebooks/` and `benchmark/`.
* Compare runtime and memory usage across:

  * Varying sequence lengths (up to 1000 bp)
  * Random vs real genomic inputs
* Visualization of performance in Jupyter Notebooks using `matplotlib`.


