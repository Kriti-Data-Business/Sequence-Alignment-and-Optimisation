import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc
import pandas as pd
from typing import Tuple, List
import random
import string

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SequenceAligner:
    """
    Comprehensive implementation of sequence alignment algorithms
    Demonstrates CLO1: Applying and analyzing algorithms and data structures
    """
    
    def __init__(self, match_score=2, mismatch_penalty=-1, gap_penalty=-2):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
    
    def needleman_wunsch(self, seq1: str, seq2: str) -> Tuple[int, str, str, np.ndarray]:
        """
        Global sequence alignment using Needleman-Wunsch algorithm
        Time Complexity: O(m*n), Space Complexity: O(m*n)
        """
        m, n = len(seq1), len(seq2)
        
        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize first row and column
        for i in range(m + 1):
            score_matrix[i][0] = i * self.gap_penalty
        for j in range(n + 1):
            score_matrix[0][j] = j * self.gap_penalty
        
        # Fill the scoring matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i-1][j-1] + (
                    self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty
                )
                delete = score_matrix[i-1][j] + self.gap_penalty
                insert = score_matrix[i][j-1] + self.gap_penalty
                score_matrix[i][j] = max(match, delete, insert)
        
        # Traceback to find alignment
        aligned_seq1, aligned_seq2 = self._traceback_global(seq1, seq2, score_matrix)
        
        return score_matrix[m][n], aligned_seq1, aligned_seq2, score_matrix
    
    def smith_waterman(self, seq1: str, seq2: str) -> Tuple[int, str, str, np.ndarray]:
        """
        Local sequence alignment using Smith-Waterman algorithm
        Time Complexity: O(m*n), Space Complexity: O(m*n)
        """
        m, n = len(seq1), len(seq2)
        
        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=int)
        max_score = 0
        max_pos = (0, 0)
        
        # Fill the scoring matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i-1][j-1] + (
                    self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty
                )
                delete = score_matrix[i-1][j] + self.gap_penalty
                insert = score_matrix[i][j-1] + self.gap_penalty
                score_matrix[i][j] = max(0, match, delete, insert)
                
                if score_matrix[i][j] > max_score:
                    max_score = score_matrix[i][j]
                    max_pos = (i, j)
        
        # Traceback for local alignment
        aligned_seq1, aligned_seq2 = self._traceback_local(seq1, seq2, score_matrix, max_pos)
        
        return max_score, aligned_seq1, aligned_seq2, score_matrix
    
    def hirschberg_algorithm(self, seq1: str, seq2: str) -> Tuple[int, str, str]:
        """
        Space-optimized global alignment using Hirschberg's algorithm
        Time Complexity: O(m*n), Space Complexity: O(min(m,n))
        Demonstrates CLO3: Problem solving using advanced paradigms
        """
        def nw_score_only(s1, s2):
            """Compute only the last row of scoring matrix"""
            m, n = len(s1), len(s2)
            prev_row = [i * self.gap_penalty for i in range(n + 1)]
            
            for i in range(1, m + 1):
                curr_row = [i * self.gap_penalty]
                for j in range(1, n + 1):
                    match = prev_row[j-1] + (
                        self.match_score if s1[i-1] == s2[j-1] else self.mismatch_penalty
                    )
                    delete = prev_row[j] + self.gap_penalty
                    insert = curr_row[j-1] + self.gap_penalty
                    curr_row.append(max(match, delete, insert))
                prev_row = curr_row
            
            return prev_row
        
        def hirschberg_rec(s1, s2):
            if len(s1) == 0:
                return '-' * len(s2), s2
            elif len(s2) == 0:
                return s1, '-' * len(s1)
            elif len(s1) == 1 or len(s2) == 1:
                # Base case: use standard algorithm
                _, align1, align2, _ = self.needleman_wunsch(s1, s2)
                return align1, align2
            else:
                # Divide
                mid = len(s1) // 2
                
                # Conquer: compute scores for both halves
                scores_left = nw_score_only(s1[:mid], s2)
                scores_right = nw_score_only(s1[mid:][::-1], s2[::-1])
                scores_right = scores_right[::-1]
                
                # Find optimal split point
                max_score = float('-inf')
                split_point = 0
                for j in range(len(s2) + 1):
                    total_score = scores_left[j] + scores_right[j]
                    if total_score > max_score:
                        max_score = total_score
                        split_point = j
                
                # Recursive calls
                left_align1, left_align2 = hirschberg_rec(s1[:mid], s2[:split_point])
                right_align1, right_align2 = hirschberg_rec(s1[mid:], s2[split_point:])
                
                return left_align1 + right_align1, left_align2 + right_align2
        
        aligned_seq1, aligned_seq2 = hirschberg_rec(seq1, seq2)
        
        # Calculate final score
        score = 0
        for i in range(len(aligned_seq1)):
            if aligned_seq1[i] == '-' or aligned_seq2[i] == '-':
                score += self.gap_penalty
            elif aligned_seq1[i] == aligned_seq2[i]:
                score += self.match_score
            else:
                score += self.mismatch_penalty
        
        return score, aligned_seq1, aligned_seq2
    
    def _traceback_global(self, seq1, seq2, score_matrix):
        """Traceback for global alignment"""
        aligned_seq1, aligned_seq2 = "", ""
        i, j = len(seq1), len(seq2)
        
        while i > 0 or j > 0:
            current_score = score_matrix[i][j]
            
            if i > 0 and j > 0:
                diagonal_score = score_matrix[i-1][j-1] + (
                    self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty
                )
                if current_score == diagonal_score:
                    aligned_seq1 = seq1[i-1] + aligned_seq1
                    aligned_seq2 = seq2[j-1] + aligned_seq2
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0 and current_score == score_matrix[i-1][j] + self.gap_penalty:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                j -= 1
        
        return aligned_seq1, aligned_seq2
    
    def _traceback_local(self, seq1, seq2, score_matrix, max_pos):
        """Traceback for local alignment"""
        aligned_seq1, aligned_seq2 = "", ""
        i, j = max_pos
        
        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            current_score = score_matrix[i][j]
            
            diagonal_score = score_matrix[i-1][j-1] + (
                self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty
            )
            
            if current_score == diagonal_score:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                i -= 1
                j -= 1
            elif current_score == score_matrix[i-1][j] + self.gap_penalty:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                j -= 1
        
        return aligned_seq1, aligned_seq2

class PerformanceBenchmark:
    """
    Benchmarking suite for algorithm performance analysis
    Demonstrates CLO2: Theoretical and empirical complexity comparison
    """
    
    def __init__(self):
        self.aligner = SequenceAligner()
        self.results = []
    
    def generate_random_sequence(self, length: int) -> str:
        """Generate random DNA sequence"""
        return ''.join(random.choices('ATGC', k=length))
    
    def measure_performance(self, seq1: str, seq2: str, algorithm: str) -> dict:
        """Measure time and memory usage for an algorithm"""
        tracemalloc.start()
        start_time = time.time()
        
        if algorithm == "needleman_wunsch":
            score, align1, align2, matrix = self.aligner.needleman_wunsch(seq1, seq2)
        elif algorithm == "smith_waterman":
            score, align1, align2, matrix = self.aligner.smith_waterman(seq1, seq2)
        elif algorithm == "hirschberg":
            score, align1, align2 = self.aligner.hirschberg_algorithm(seq1, seq2)
            matrix = None
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'algorithm': algorithm,
            'seq1_length': len(seq1),
            'seq2_length': len(seq2),
            'score': score,
            'execution_time': end_time - start_time,
            'memory_usage': peak,
            'aligned_seq1': align1,
            'aligned_seq2': align2,
            'score_matrix': matrix
        }
    
    def run_benchmark_suite(self, sizes: List[int]) -> pd.DataFrame:
        """Run comprehensive benchmark across different sequence sizes"""
        algorithms = ["needleman_wunsch", "smith_waterman", "hirschberg"]
        results = []
        
        for size in sizes:
            print(f"Benchmarking size {size}...")
            seq1 = self.generate_random_sequence(size)
            seq2 = self.generate_random_sequence(size)
            
            for algorithm in algorithms:
                try:
                    result = self.measure_performance(seq1, seq2, algorithm)
                    results.append(result)
                except Exception as e:
                    print(f"Error with {algorithm} at size {size}: {e}")
        
        return pd.DataFrame(results)

class VisualizationSuite:
    """
    Comprehensive visualization for algorithm analysis
    Demonstrates CLO1: Analyzing algorithms through visual representation
    """
    
    @staticmethod
    def plot_score_matrix(score_matrix: np.ndarray, seq1: str, seq2: str, title: str):
        """Visualize scoring matrix as heatmap"""
        plt.figure(figsize=(max(8, len(seq2)), max(6, len(seq1))))
        
        # Create labels
        x_labels = ['-'] + list(seq2)
        y_labels = ['-'] + list(seq1)
        
        sns.heatmap(score_matrix, annot=True, fmt='d', cmap='viridis',
                   xticklabels=x_labels, yticklabels=y_labels)
        plt.title(f'{title} - Scoring Matrix')
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_alignment(seq1: str, seq2: str, title: str):
        """Visualize sequence alignment"""
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Sequence 1: {seq1}")
        print(f"Sequence 2: {seq2}")
        
        # Show matches, mismatches, and gaps
        alignment_chars = []
        for i in range(len(seq1)):
            if seq1[i] == seq2[i]:
                alignment_chars.append('|')  # Match
            elif seq1[i] == '-' or seq2[i] == '-':
                alignment_chars.append(' ')  # Gap
            else:
                alignment_chars.append('.')  # Mismatch
        
        print(f"Alignment:  {''.join(alignment_chars)}")
        print()
    
    @staticmethod
    def plot_performance_comparison(df: pd.DataFrame):
        """Create comprehensive performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution Time Comparison
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm]
            ax1.plot(data['seq1_length'], data['execution_time'], 
                    marker='o', label=algorithm, linewidth=2)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage Comparison
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm]
            ax2.plot(data['seq1_length'], data['memory_usage'] / (1024*1024), 
                    marker='s', label=algorithm, linewidth=2)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Time Complexity Analysis
        sizes = sorted(df['seq1_length'].unique())
        theoretical_n2 = [s*s for s in sizes]
        theoretical_n2 = [t/theoretical_n2[0] * df[df['seq1_length']==sizes[0]]['execution_time'].iloc[0] 
                         for t in theoretical_n2]
        
        ax3.plot(sizes, theoretical_n2, '--', label='O(n²) theoretical', alpha=0.7, linewidth=2)
        
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm].sort_values('seq1_length')
            ax3.plot(data['seq1_length'], data['execution_time'], 
                    marker='o', label=f'{algorithm} empirical', linewidth=2)
        
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Theoretical vs Empirical Complexity')
        ax3.legend()
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Algorithm Score Comparison
        score_comparison = df.groupby(['algorithm', 'seq1_length'])['score'].mean().unstack(level=0)
        score_comparison.plot(kind='bar', ax=ax4)
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Average Alignment Score')
        ax4.set_title('Average Alignment Scores by Algorithm')
        ax4.legend(title='Algorithm')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_complexity_summary():
        """Create theoretical complexity comparison table"""
        complexity_data = {
            'Algorithm': ['Needleman-Wunsch', 'Smith-Waterman', 'Hirschberg'],
            'Time Complexity': ['O(mn)', 'O(mn)', 'O(mn)'],
            'Space Complexity': ['O(mn)', 'O(mn)', 'O(min(m,n))'],
            'Alignment Type': ['Global', 'Local', 'Global'],
            'Key Advantage': ['Optimal global alignment', 'Finds local similarities', 'Space-efficient']
        }
        
        df = pd.DataFrame(complexity_data)
        print("\nTHEORETICAL COMPLEXITY ANALYSIS")
        print("=" * 50)
        print(df.to_string(index=False))
        print()
        
        return df

# Demonstration and Testing Suite
def run_comprehensive_demo():
    """
    Main demonstration function showcasing all COSC3119 CLOs
    """
    print("🧬 SEQUENCE ALIGNMENT ALGORITHMS DEMONSTRATION")
    print("COSC3119 - Advanced Data Structures & Algorithms")
    print("=" * 60)
    
    # Initialize components
    aligner = SequenceAligner()
    benchmark = PerformanceBenchmark()
    visualizer = VisualizationSuite()
    
    # CLO1 & CLO3: Algorithm Implementation and Analysis
    print("\n📊 CLO1 & CLO3: ALGORITHM IMPLEMENTATION AND ANALYSIS")
    print("-" * 55)
    
    # Test sequences
    seq1 = "AGCTGAC"
    seq2 = "AGCTGTC"
    
    print(f"Test Sequences:")
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    
    # Demonstrate each algorithm
    algorithms = [
        ("Needleman-Wunsch (Global)", aligner.needleman_wunsch),
        ("Smith-Waterman (Local)", aligner.smith_waterman),
        ("Hirschberg (Space-Optimized)", aligner.hirschberg_algorithm)
    ]
    
    results = {}
    for name, func in algorithms:
        print(f"\n{name}:")
        start_time = time.time()
        
        if name.startswith("Hirschberg"):
            score, align1, align2 = func(seq1, seq2)
            matrix = None
        else:
            score, align1, align2, matrix = func(seq1, seq2)
        
        end_time = time.time()
        
        print(f"  Score: {score}")
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        
        results[name] = (score, align1, align2, matrix)
        
        # Visualize alignment
        visualizer.plot_alignment(align1, align2, f"{name} Alignment")
    
    # Visualize scoring matrices for Needleman-Wunsch and Smith-Waterman
    if results["Needleman-Wunsch (Global)"][3] is not None:
        visualizer.plot_score_matrix(
            results["Needleman-Wunsch (Global)"][3], 
            seq1, seq2, 
            "Needleman-Wunsch"
        )
    
    if results["Smith-Waterman (Local)"][3] is not None:
        visualizer.plot_score_matrix(
            results["Smith-Waterman (Local)"][3], 
            seq1, seq2, 
            "Smith-Waterman"
        )
    
    # CLO2: Theoretical and Empirical Complexity Comparison
    print("\n📈 CLO2: COMPLEXITY ANALYSIS AND BENCHMARKING")
    print("-" * 50)
    
    # Show theoretical complexity
    complexity_df = visualizer.create_complexity_summary()
    
    # Run empirical benchmarks
    print("Running empirical performance benchmarks...")
    sizes = [10, 20, 50, 100, 200]  # Adjust based on computational resources
    benchmark_df = benchmark.run_benchmark_suite(sizes)
    
    # Display benchmark results
    print("\nEMPIRICAL BENCHMARK RESULTS:")
    print(benchmark_df.groupby(['algorithm', 'seq1_length'])[['execution_time', 'memory_usage']].mean())
    
    # Create comprehensive visualization
    visualizer.plot_performance_comparison(benchmark_df)
    
    # Advanced Analysis: Space-Time Tradeoff
    print("\n🔍 ADVANCED ANALYSIS: SPACE-TIME TRADEOFF")
    print("-" * 45)
    
    large_seq1 = benchmark.generate_random_sequence(500)
    large_seq2 = benchmark.generate_random_sequence(500)
    
    print("Comparing algorithms on large sequences (500bp each):")
    
    for algorithm in ["needleman_wunsch", "hirschberg"]:
        try:
            result = benchmark.measure_performance(large_seq1, large_seq2, algorithm)
            print(f"\n{algorithm.replace('_', ' ').title()}:")
            print(f"  Execution time: {result['execution_time']:.4f} seconds")
            print(f"  Memory usage: {result['memory_usage'] / (1024*1024):.2f} MB")
            print(f"  Score: {result['score']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Research Applications
    print("\n🔬 REAL-WORLD APPLICATIONS")
    print("-" * 30)
    print("1. Genome Assembly: Hirschberg's algorithm enables alignment of large chromosomes")
    print("2. Phylogenetic Analysis: Smith-Waterman finds conserved regions across species")
    print("3. Drug Discovery: Protein sequence alignment for target identification")
    print("4. Personalized Medicine: Comparing patient genomes to reference sequences")
    
    print("\n✅ DEMONSTRATION COMPLETE")
    print("All COSC3119 CLOs successfully demonstrated:")
    print("  ✓ CLO1: Algorithm implementation and analysis")
    print("  ✓ CLO2: Theoretical vs empirical complexity comparison")
    print("  ✓ CLO3: Advanced problem-solving paradigms (divide-and-conquer)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run the comprehensive demonstration
    run_comprehensive_demo()
    
    print("\n📚 NEXT STEPS FOR PORTFOLIO:")
    print("1. Upload this notebook to GitHub with detailed README")
    print("2. Include performance benchmarks on real genomic datasets")
    print("3. Add comparative analysis with existing bioinformatics tools")
    print("4. Document theoretical insights and practical applications")
    print("5. Include peer review or supervisor feedback")
