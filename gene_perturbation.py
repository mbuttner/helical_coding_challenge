"""
Gene Perturbation Framework for Single-Cell Foundation Models

This module provides tools to perform in-silico gene perturbations
and analyze their effects in the embedding space of a foundation model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings


class PerturbationType(Enum):
    """Types of gene perturbations."""
    KNOCKOUT = "knockout"  # Set gene expression to 0
    OVEREXPRESSION = "overexpression"  # Increase gene expression
    KNOCKDOWN = "knockdown"  # Reduce gene expression


@dataclass
class PerturbationResult:
    """Container for perturbation results."""
    gene_names: List[str]
    perturbation_type: PerturbationType
    original_embeddings: np.ndarray
    perturbed_embeddings: np.ndarray
    embedding_shifts: np.ndarray
    original_expressions: Optional[np.ndarray] = None
    perturbed_expressions: Optional[np.ndarray] = None
    
    def compute_distances(self, metric: str = 'cosine', cell_indices = None) -> np.ndarray:
        """Compute distances between original and perturbed embeddings."""
        if metric == 'euclidean':
            if cell_indices is not None:
                return np.linalg.norm(self.embedding_shifts[cell_indices], axis=1)
            else:
                return np.linalg.norm(self.embedding_shifts, axis=1)
        elif metric == 'cosine':
            if cell_indices is not None:
                original_emb_sub = self.original_embeddings[cell_indices]
                perturb_emb_sub = self.perturbed_embeddings[cell_indices]
                orig_norm = np.linalg.norm(original_emb_sub, axis=1, keepdims=True)
                pert_norm = np.linalg.norm(perturb_emb_sub, axis=1, keepdims=True)
                cosine_sim = np.sum(original_emb_sub * perturb_emb_sub, axis=1) / (orig_norm.squeeze() * pert_norm.squeeze())
            else: 
                orig_norm = np.linalg.norm(self.original_embeddings, axis=1, keepdims=True)
                pert_norm = np.linalg.norm(self.perturbed_embeddings, axis=1, keepdims=True)
                cosine_sim = np.sum(self.original_embeddings * self.perturbed_embeddings, axis=1) / (orig_norm.squeeze() * pert_norm.squeeze())
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of the perturbation effect."""
        distances = self.compute_distances('euclidean')
        return {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'max_distance': float(np.max(distances)),
            'min_distance': float(np.min(distances))
        }
    
    def compute_knn_neighborhoods(
        self,
        k: int = 30,
        metric: str = 'euclidean'
    ) -> Dict[str, np.ndarray]:
        """
        Compute k-nearest neighbors in joint embedding space.
        
        This creates a joint embedding space by stacking original and perturbed
        embeddings, then finds the k-nearest neighbors for each cell.
        
        Args:
            k: Number of nearest neighbors to find
            metric: Distance metric ('euclidean' or 'cosine')
        
        Returns:
            Dictionary containing:
                - 'indices': KNN indices for each cell (shape: n_cells, k)
                - 'distances': KNN distances for each cell (shape: n_cells, k)
                - 'neighbor_types': Type of each neighbor (0=original, 1=perturbed)
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_cells = len(self.original_embeddings)
        
        # Create joint embedding space
        # Stack: [original_cells, perturbed_cells]
        joint_embeddings = np.vstack([
            self.original_embeddings,
            self.perturbed_embeddings
        ])
        
        # Fit KNN model
        if metric == 'cosine':
            # For cosine, use normalized embeddings
            joint_norm = joint_embeddings / (np.linalg.norm(joint_embeddings, axis=1, keepdims=True) + 1e-8)
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            nbrs.fit(joint_norm)
            distances, indices = nbrs.kneighbors(joint_norm)
        else:
            nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric)
            nbrs.fit(joint_embeddings)
            distances, indices = nbrs.kneighbors(joint_embeddings)
        
        # Remove self from neighbors (first neighbor is always self with distance 0)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Determine neighbor types (0 = original, 1 = perturbed)
        # Original cells are indices 0:n_cells, perturbed are n_cells:2*n_cells
        neighbor_types = (indices >= n_cells).astype(int)
        
        return {
            'indices': indices,
            'distances': distances,
            'neighbor_types': neighbor_types,
            'n_cells': n_cells
        }
    
    def test_neighborhood_bias(
        self,
        k: int = 30,
        metric: str = 'euclidean',
        return_all_cells: bool = False
    ) -> Dict:
        """
        Test if cell neighborhoods are biased towards original or perturbed cells.
        
        Uses chi-square test to determine if the proportion of original vs perturbed
        neighbors differs significantly from the expected 50/50 distribution in the
        joint embedding space.
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric ('euclidean' or 'cosine')
            return_all_cells: If True, return per-cell statistics
        
        Returns:
            Dictionary containing:
                - 'chi_square_stats': Chi-square statistic for each cell
                - 'p_values': P-value for each cell
                - 'bias_scores': Bias score (-1 to 1, negative = original-biased, positive = perturbed-biased)
                - 'significant_cells': Indices of cells with significant bias (p < 0.05)
                - 'n_original_biased': Number of cells biased towards original
                - 'n_perturbed_biased': Number of cells biased towards perturbed
                - 'n_mixed': Number of cells with mixed neighborhoods
                - 'summary': Overall summary statistics
        """
        from scipy.stats import chi2_contingency, chisquare
        
        # Get KNN neighborhoods
        knn_result = self.compute_knn_neighborhoods(k=k, metric=metric)
        neighbor_types = knn_result['neighbor_types']
        n_cells = knn_result['n_cells']
        
        # Analyze neighborhoods for all cells (both original and perturbed)
        chi_square_stats = []
        p_values = []
        bias_scores = []
        
        for i in range(2 * n_cells):
            neighbors = neighbor_types[i]
            
            # Count original vs perturbed neighbors
            n_original = np.sum(neighbors == 0)
            n_perturbed = np.sum(neighbors == 1)
            
            # Expected: 50/50 split in joint space
            observed = np.array([n_original, n_perturbed])
            expected = np.array([k/2, k/2])
            
            # Chi-square test
            chi2, p = chisquare(observed, expected)
            chi_square_stats.append(chi2)
            p_values.append(p)
            
            # Bias score: -1 (all original) to +1 (all perturbed)
            bias_score = (n_perturbed - n_original) / k
            bias_scores.append(bias_score)
        
        chi_square_stats = np.array(chi_square_stats)
        p_values = np.array(p_values)
        bias_scores = np.array(bias_scores)
        
        # Find significant cells (p < 0.05)
        significant_mask = p_values < 0.05
        significant_indices = np.where(significant_mask)[0]
        
        # Categorize bias for significant cells
        significant_bias = bias_scores[significant_mask]
        n_original_biased = np.sum(significant_bias < -0.1)  # More than 10% bias towards original
        n_perturbed_biased = np.sum(significant_bias > 0.1)  # More than 10% bias towards perturbed
        n_mixed = np.sum(np.abs(significant_bias) <= 0.1)    # Mixed neighborhood
        
        # Create summary
        summary = {
            'total_cells': 2 * n_cells,
            'k_neighbors': k,
            'n_significant': len(significant_indices),
            'pct_significant': 100 * len(significant_indices) / (2 * n_cells),
            'n_original_biased': int(n_original_biased),
            'n_perturbed_biased': int(n_perturbed_biased),
            'n_mixed': int(n_mixed),
            'mean_bias_score': float(np.mean(bias_scores)),
            'mean_bias_original_cells': float(np.mean(bias_scores[:n_cells])),
            'mean_bias_perturbed_cells': float(np.mean(bias_scores[n_cells:])),
            'mean_p_value': float(np.mean(p_values)),
            'median_p_value': float(np.median(p_values))
        }
        
        result = {
            'summary': summary,
            'significant_cells': significant_indices,
            'n_original_biased': int(n_original_biased),
            'n_perturbed_biased': int(n_perturbed_biased),
            'n_mixed': int(n_mixed)
        }
        
        if return_all_cells:
            result.update({
                'chi_square_stats': chi_square_stats,
                'p_values': p_values,
                'bias_scores': bias_scores,
                'neighbor_types': neighbor_types
            })
        
        return result
    
    def compute_mixing_score(
        self,
        k: int = 30,
        metric: str = 'euclidean'
    ) -> float:
        """
        Compute a mixing score between original and perturbed embeddings.
        
        A score of 1.0 indicates perfect mixing (neighborhoods have 50/50 original/perturbed).
        A score of 0.0 indicates complete separation (neighborhoods are homogeneous).
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric
        
        Returns:
            Mixing score between 0 and 1
        """
        knn_result = self.compute_knn_neighborhoods(k=k, metric=metric)
        neighbor_types = knn_result['neighbor_types']
        
        # For each cell, compute how balanced the neighborhood is
        mixing_scores = []
        for neighbors in neighbor_types:
            n_original = np.sum(neighbors == 0)
            n_perturbed = np.sum(neighbors == 1)
            
            # Perfect mixing: 50/50 split
            # Complete separation: all same type
            balance = min(n_original, n_perturbed) / (k / 2)
            mixing_scores.append(balance)
        
        # Average mixing score across all cells
        return float(np.mean(mixing_scores))



class GenePerturbationModel:
    """
    A framework for performing gene perturbations on single-cell data
    using a foundation model's embedding space.
    """
    
    def __init__(
        self,
        foundation_model: Callable,
        gene_names: List[str],
        normalize_embeddings: bool = False
    ):
        """
        Initialize the perturbation model.
        
        Args:
            foundation_model: A callable that takes expression data and returns embeddings
                             Should have signature: model(expression_data) -> embeddings
            gene_names: List of gene names corresponding to expression columns
            normalize_embeddings: Whether to normalize embeddings to unit length
        """
        self.foundation_model = foundation_model
        self.gene_names = np.array(gene_names)
        self.normalize_embeddings = normalize_embeddings
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    def _get_gene_indices(self, genes: Union[str, List[str]]) -> np.ndarray:
        """Convert gene names to indices."""
        if isinstance(genes, str):
            genes = [genes]
        
        indices = []
        for gene in genes:
            if gene not in self.gene_to_idx:
                warnings.warn(f"Gene {gene} not found in gene list. Skipping.")
                continue
            indices.append(self.gene_to_idx[gene])
        
        return np.array(indices)
    
    def _embed_data(self, expression_data: np.ndarray) -> np.ndarray:
        """Embed expression data using the foundation model."""
        embeddings = self.foundation_model(expression_data)
        
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def knockout(
        self,
        expression_data: np.ndarray,
        genes: Union[str, List[str]],
        return_expressions: bool = False
    ) -> PerturbationResult:
        """
        Perform gene knockout by setting expression to 0.
        
        Args:
            expression_data: Expression matrix (n_cells, n_genes)
            genes: Gene name(s) to knockout
            return_expressions: Whether to return expression matrices
        
        Returns:
            PerturbationResult object
        """
        gene_indices = self._get_gene_indices(genes)
        
        # Create perturbed data
        perturbed_data = expression_data.copy()
        perturbed_data[:, gene_indices] = 0
        
        # Get embeddings
        original_embeddings = self._embed_data(expression_data)
        perturbed_embeddings = self._embed_data(perturbed_data)
        
        return PerturbationResult(
            gene_names=genes if isinstance(genes, list) else [genes],
            perturbation_type=PerturbationType.KNOCKOUT,
            original_embeddings=original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - original_embeddings,
            original_expressions=expression_data if return_expressions else None,
            perturbed_expressions=perturbed_data if return_expressions else None
        )
    
    def overexpression(
        self,
        expression_data: np.ndarray,
        genes: Union[str, List[str]],
        fold_change: float = 2.0,
        return_expressions: bool = False
    ) -> PerturbationResult:
        """
        Perform gene overexpression by increasing expression.
        
        Args:
            expression_data: Expression matrix (n_cells, n_genes)
            genes: Gene name(s) to overexpress
            fold_change: Factor by which to increase expression
            return_expressions: Whether to return expression matrices
        
        Returns:
            PerturbationResult object
        """
        gene_indices = self._get_gene_indices(genes)
        
        # Create perturbed data
        perturbed_data = expression_data.copy()
        perturbed_data[:, gene_indices] = perturbed_data[:, gene_indices] * fold_change
        
        # Get embeddings
        original_embeddings = self._embed_data(expression_data)
        perturbed_embeddings = self._embed_data(perturbed_data)
        
        return PerturbationResult(
            gene_names=genes if isinstance(genes, list) else [genes],
            perturbation_type=PerturbationType.OVEREXPRESSION,
            original_embeddings=original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - original_embeddings,
            original_expressions=expression_data if return_expressions else None,
            perturbed_expressions=perturbed_data if return_expressions else None
        )
    
    def knockdown(
        self,
        expression_data: np.ndarray,
        genes: Union[str, List[str]],
        reduction_factor: float = 0.5,
        return_expressions: bool = False
    ) -> PerturbationResult:
        """
        Perform gene knockdown by reducing expression.
        
        Args:
            expression_data: Expression matrix (n_cells, n_genes)
            genes: Gene name(s) to knockdown
            reduction_factor: Factor by which to reduce expression (0-1)
            return_expressions: Whether to return expression matrices
        
        Returns:
            PerturbationResult object
        """
        if not 0 <= reduction_factor <= 1:
            raise ValueError("reduction_factor must be between 0 and 1")
        
        gene_indices = self._get_gene_indices(genes)
        
        # Create perturbed data
        perturbed_data = expression_data.copy()
        perturbed_data[:, gene_indices] = perturbed_data[:, gene_indices] * reduction_factor
        
        # Get embeddings
        original_embeddings = self._embed_data(expression_data)
        perturbed_embeddings = self._embed_data(perturbed_data)
        
        return PerturbationResult(
            gene_names=genes if isinstance(genes, list) else [genes],
            perturbation_type=PerturbationType.KNOCKDOWN,
            original_embeddings=original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - original_embeddings,
            original_expressions=expression_data if return_expressions else None,
            perturbed_expressions=perturbed_data if return_expressions else None
        )
    
    def batch_perturbation(
        self,
        expression_data: np.ndarray,
        gene_list: List[str],
        perturbation_type: str = 'knockout',
        **kwargs
    ) -> Dict[str, PerturbationResult]:
        """
        Perform perturbations on multiple genes independently.
        
        Args:
            expression_data: Expression matrix (n_cells, n_genes)
            gene_list: List of genes to perturb individually
            perturbation_type: Type of perturbation ('knockout', 'overexpression', 'knockdown')
            **kwargs: Additional arguments for the perturbation method
        
        Returns:
            Dictionary mapping gene names to PerturbationResult objects
        """
        results = {}
        
        perturbation_method = {
            'knockout': self.knockout,
            'overexpression': self.overexpression,
            'knockdown': self.knockdown
        }.get(perturbation_type)
        
        if perturbation_method is None:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        for gene in gene_list:
            try:
                result = perturbation_method(expression_data, gene, **kwargs)
                results[gene] = result
            except Exception as e:
                warnings.warn(f"Failed to perturb {gene}: {str(e)}")
        
        return results
    
    def combinatorial_perturbation(
        self,
        expression_data: np.ndarray,
        gene_combinations: List[List[str]],
        perturbation_type: str = 'knockout',
        **kwargs
    ) -> Dict[str, PerturbationResult]:
        """
        Perform perturbations on combinations of genes.
        
        Args:
            expression_data: Expression matrix (n_cells, n_genes)
            gene_combinations: List of gene combinations to perturb together
            perturbation_type: Type of perturbation
            **kwargs: Additional arguments for the perturbation method
        
        Returns:
            Dictionary mapping gene combination strings to PerturbationResult objects
        """
        results = {}
        
        perturbation_method = {
            'knockout': self.knockout,
            'overexpression': self.overexpression,
            'knockdown': self.knockdown
        }.get(perturbation_type)
        
        if perturbation_method is None:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        for gene_combo in gene_combinations:
            combo_key = "+".join(gene_combo)
            try:
                result = perturbation_method(expression_data, gene_combo, **kwargs)
                results[combo_key] = result
            except Exception as e:
                warnings.warn(f"Failed to perturb {combo_key}: {str(e)}")
        
        return results
