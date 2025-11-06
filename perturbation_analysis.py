"""
Analysis utilities for gene perturbation results.

Provides functions for analyzing, visualizing, and ranking perturbation effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from gene_perturbation import PerturbationResult
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# UMAP is optional
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class PerturbationAnalyzer:
    """Analyze and visualize perturbation results."""
    
    def __init__(self, results: Dict[str, PerturbationResult]):
        """
        Initialize analyzer with perturbation results.
        
        Args:
            results: Dictionary mapping gene names to PerturbationResult objects
        """
        self.results = results
    
    def rank_by_effect_size(
        self,
        metric: str = 'euclidean',
        top_k: Optional[int] = None,
        cell_indices: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Rank genes by their perturbation effect size.
        
        Args:
            metric: Distance metric ('euclidean' or 'cosine')
            top_k: Return only top k genes (None for all)
        
        Returns:
            DataFrame with genes ranked by effect size
        """
        rankings = []
        
        if isinstance(self.results, dict):

            for gene_name, result in self.results.items():
                distances = result.compute_distances(metric, cell_indices=cell_indices)
                rankings.append({
                'gene': gene_name,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'median_distance': np.median(distances),
                'max_distance': np.max(distances),
                'n_cells': len(distances)
                })
        
        else:
            gene_name = self.results.gene_names
            distances = self.results.compute_distances(metric, cell_indices=cell_indices)
            rankings.append({
                'gene': gene_name,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'median_distance': np.median(distances),
                'max_distance': np.max(distances),
                'n_cells': len(distances)
                })

        df = pd.DataFrame(rankings)
        df = df.sort_values('mean_distance', ascending=False)
        
        if top_k is not None:
            df = df.head(top_k)
        
        return df.reset_index(drop=True)
    
    def compute_embedding_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for all perturbations.
        
        Returns:
            DataFrame with statistics for each perturbation
        """
        stats_list = []

        if isinstance(self.results, dict):

            for gene_name, result in self.results.items():
                stats = result.get_summary_stats()
                stats['gene'] = gene_name
                stats['perturbation_type'] = result.perturbation_type.value
                stats['n_genes_perturbed'] = len(result.gene_names)
                stats_list.append(stats)
        
        else:
            gene_name = self.results.gene_names
            result = self.results
            stats = result.get_summary_stats()
            stats['gene'] = gene_name
            stats['perturbation_type'] = result.perturbation_type.value
            stats['n_genes_perturbed'] = len(result.gene_names)
            stats_list.append(stats)
            
        return pd.DataFrame(stats_list)
    
    def plot_effect_distribution(
        self,
        genes: Optional[List[str]] = None,
        metric: str = 'euclidean',
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distribution of perturbation effects.
        
        Args:
            genes: List of genes to plot (None for all)
            metric: Distance metric
            figsize: Figure size
        """
        if genes is None:
            if isinstance(self.results, dict):
                genes = list(self.results.keys())
            else:
                genes = []
                genes.append(self.results.gene_names)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        data = []
        labels = []
        for gene in genes:
            if isinstance(self.results, dict):
                if gene in self.results:
                    distances = self.results[gene].compute_distances(metric)
                    data.append(distances)
                    labels.append(gene)
            else:
                distances = self.results.compute_distances(metric)
                data.append(distances)
                labels.append(', '.join(gene) if len(gene)>1 else gene)
        
        axes[0].boxplot(data, labels=labels)
        axes[0].set_xlabel('Gene')
        axes[0].set_ylabel(f'{metric.capitalize()} Distance')
        axes[0].set_title('Perturbation Effect Distribution')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Histogram
        for gene, distances in zip(labels, data):
            axes[1].hist(distances, alpha=0.5, label=gene, bins=30)
        
        axes[1].set_xlabel(f'{metric.capitalize()} Distance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Effect Size Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_embedding_shift(
        self,
        gene: str,
        method: str = 'pca',
        color: list = None,
        cmap: str = 'tab20b',
        n_components: int = 2,
        sample_size: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize embedding shifts using dimensionality reduction.
        
        Args:
            gene: Gene name to visualize
            method: Dimensionality reduction method ('pca' or 'umap')
            color: List of numeric values. Color by another feature like cluster ID, embeddings shown as different shapes 
            cmap: if color is not None, this palette is used to color points by feature
            n_components: Number of components (2 or 3)
            sample_size: Subsample cells for visualization
            figsize: Figure size
        """
    
        if isinstance(self.results, dict):
            if gene not in self.results:
                raise ValueError(f"Gene {gene} not found in results")
            result = self.results[gene]
        else:
            if not gene == self.results.gene_names:
                raise ValueError(f"Gene {gene} not found in results")
            result = self.results
        
        # Combine original and perturbed embeddings
        all_embeddings = np.vstack([
            result.original_embeddings,
            result.perturbed_embeddings
        ])
        
        # Subsample if requested
        if sample_size and sample_size < len(result.original_embeddings):
            indices = np.random.choice(len(result.original_embeddings), sample_size, replace=False)
            all_embeddings = np.vstack([
                result.original_embeddings[indices],
                result.perturbed_embeddings[indices]
            ])
        else:
            indices = np.arange(len(result.original_embeddings))
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Install it with: pip install umap-learn")
            reducer = UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(all_embeddings)
        
        # Split back into original and perturbed
        n_cells = len(indices)
        reduced_orig = reduced[:n_cells]
        reduced_pert = reduced[n_cells:]
            
        
        # Plotting
        if n_components == 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(reduced_orig[:, 0], reduced_orig[:, 1], 
                       c=color if ( color is not None ) else 'tab:orange',
                       marker='o', cmap=cmap if color is not None else None,
                      alpha=0.5, label='Original', s=20)
            ax.scatter(reduced_pert[:, 0], reduced_pert[:, 1], 
                       c=color if ( color is not None ) else 'tab:blue',
                       marker='x' if color is not None else 'o',
                       cmap=cmap if color is not None else None,
                      alpha=0.5, label='Perturbed', s=20)
            
            # Draw arrows for shifts
            for i in range(min(100, n_cells)):  # Limit arrows for clarity
                ax.arrow(reduced_orig[i, 0], reduced_orig[i, 1],
                        reduced_pert[i, 0] - reduced_orig[i, 0],
                        reduced_pert[i, 1] - reduced_orig[i, 1],
                        alpha=0.2, head_width=0.05, color='gray')
            
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_title(f'Embedding Shift for {gene} Perturbation')
            ax.legend()
            
        elif n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_orig[:, 0], reduced_orig[:, 1], reduced_orig[:, 2],
                       c=color if ( color is not None ) else 'tab:orange',
                       marker='o',
                       cmap=cmap if color is not None else None,
                      alpha=0.5, label='Original', s=20)
            ax.scatter(reduced_pert[:, 0], reduced_pert[:, 1], reduced_pert[:, 2],
                       c=color if ( color is not None ) else 'tab:blue',
                       marker='x' if color is not None else 'o',
                       cmap=cmap if color is not None else None,
                      alpha=0.5, label='Perturbed', s=20)
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_zlabel(f'{method.upper()} 3')
            ax.set_title(f'Embedding Shift for {gene} Perturbation')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def compare_perturbations(
        self,
        genes: List[str],
        metric: str = 'euclidean',
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Compare perturbation effects across multiple genes.
        
        Args:
            genes: List of genes to compare
            metric: Distance metric
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        mean_distances = []
        std_distances = []
        
        for gene in genes:
            if isinstance(self.results, dict):
                if gene not in self.results:
                    continue
                distances = self.results[gene].compute_distances(metric)
            else:
                if not gene == self.results.gene_names:
                    continue
                distances = self.results.compute_distances(metric)
            mean_distances.append(np.mean(distances))
            std_distances.append(np.std(distances))
        
        x_pos = np.arange(len(genes))
        ax.bar(x_pos, mean_distances, yerr=std_distances, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(genes, rotation=45, ha='right')
        ax.set_ylabel(f'Mean {metric.capitalize()} Distance')
        ax.set_title('Comparison of Perturbation Effects')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compute_perturbation_signature(
        self,
        gene: str,
        aggregate_method: str = 'mean'
    ) -> np.ndarray:
        """
        Compute an aggregate perturbation signature in embedding space.
        
        Args:
            gene: Gene name
            aggregate_method: How to aggregate across cells ('mean', 'median')
        
        Returns:
            Perturbation signature vector
        """
        if isinstance(self.results, dict):
            if gene not in self.results:
                raise ValueError(f"Gene {gene} not found in results")
            shifts = self.results[gene].embedding_shifts
        else:
            if not gene == self.results.gene_names:
                raise ValueError(f"Gene {gene} not found in results")
            shifts = self.results.embedding_shifts    
        
        if aggregate_method == 'mean':
            return np.mean(shifts, axis=0)
        elif aggregate_method == 'median':
            return np.median(shifts, axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
    
    
    def export_results(self, output_path: str):
        """
        Export all results to a CSV file.
        
        Args:
            output_path: Path to save CSV file
        """
        stats_df = self.compute_embedding_statistics()
        stats_df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
