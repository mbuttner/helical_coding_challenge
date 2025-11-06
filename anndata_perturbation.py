"""
AnnData Perturbation Adapter for Foundation Models with Tokenization

This adapter works with foundation models that have:
1. process_data(adata) -> tokenized HF Dataset
2. get_embeddings(dataset) -> embeddings as numpy array

Input: AnnData object
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Optional, Union, Callable
import warnings
from gene_perturbation import PerturbationResult, PerturbationType

try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    warnings.warn("AnnData not installed. Install with: pip install anndata")


class AnnDataPerturbationModel:
    """
    Gene perturbation model for foundation models with tokenization.
    
    Works with models that have:
    - process_data(adata) -> tokenized HF Dataset
    - get_embeddings(dataset) -> numpy array of embeddings
    """
    
    def __init__(
        self,
        foundation_model,
        adata: 'anndata.AnnData',
        normalize_embeddings: bool = False
    ):
        """
        Initialize the perturbation model.
        
        Args:
            foundation_model: Model with process_data() and get_embeddings() methods
            adata: AnnData object with expression data
            normalize_embeddings: Whether to normalize embeddings to unit length
        """
        if not ANNDATA_AVAILABLE:
            raise ImportError("AnnData is required. Install with: pip install anndata")
        
        # Verify model has required methods
        if not hasattr(foundation_model, 'process_data'):
            raise AttributeError("Model must have 'process_data' method")
        if not hasattr(foundation_model, 'get_embeddings'):
            raise AttributeError("Model must have 'get_embeddings' method")
        
        self.model = foundation_model
        self.adata = adata.copy()  # Work with a copy to avoid modifying original
        self.normalize_embeddings = normalize_embeddings
        
        # Extract gene names
        self.gene_names = np.array(adata.var_names.tolist())
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
        
        # Cache original tokenized dataset and embeddings
        print("Tokenizing original data...")
        self.original_dataset = self.model.process_data(self.adata)
        print("Computing original embeddings...")
        self.original_embeddings = self._get_embeddings(self.original_dataset)
        print(f"Original embeddings shape: {self.original_embeddings.shape}")
    
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
    
    def _get_embeddings(self, dataset) -> np.ndarray:
        """
        Get embeddings from dataset using model's get_embeddings method.
        
        Args:
            dataset: Tokenized HF Dataset
        
        Returns:
            Embeddings as numpy array
        """
        embeddings = self.model.get_embeddings(dataset)
        
        # Ensure numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def _create_perturbed_adata(
        self,
        gene_indices: np.ndarray,
        cell_indices: Optional[np.ndarray],
        perturbation_fn: Callable[[np.ndarray], np.ndarray]
    ) -> 'anndata.AnnData':
        """
        Create a perturbed AnnData object.
        
        Args:
            gene_indices: Indices of genes to perturb
            cell_indices: Indices of cells to perturb (None for all)
            perturbation_fn: Function to apply to expression values
        
        Returns:
            Perturbed AnnData object
        """
        # Create copy of AnnData
        perturbed_adata = self.adata.copy()
        
        # Determine which cells to perturb
        if cell_indices is None:
            cell_indices = np.arange(perturbed_adata.n_obs)
        
        # Handle sparse matrices - convert to dense for perturbation  
        is_sparse = sp.issparse(perturbed_adata.X)
        
        if is_sparse:
            # Convert to dense for modification
            X_dense = perturbed_adata.X.toarray()
        else:
            X_dense = perturbed_adata.X.copy()
        
        # Apply perturbation
        for gene_idx in gene_indices:
            current_expr = X_dense[cell_indices, gene_idx]
            X_dense[cell_indices, gene_idx] = perturbation_fn(current_expr)
        
        # Store back (keep as dense - perturbations often make data less sparse)
        perturbed_adata.X = sp.csr_matrix(X_dense)
        
        return perturbed_adata
    
    def knockout(
        self,
        genes: Union[str, List[str]],
        cell_indices: Optional[np.ndarray] = None,
        return_adata: bool = False
    ) -> PerturbationResult:
        """
        Perform gene knockout by setting expression to 0.
        
        Args:
            genes: Gene name(s) to knockout
            cell_indices: Optional cell indices to perturb (None for all cells)
            return_adata: Whether to return perturbed AnnData object
        
        Returns:
            PerturbationResult object
        """
        gene_indices = self._get_gene_indices(genes)
        genes_list = genes if isinstance(genes, list) else [genes]
        
        # Determine cells to analyze
        if cell_indices is None:
            cell_indices = np.arange(self.adata.n_obs)
        
        # Create perturbation function
        def knockout_fn(expr):
            result = np.zeros_like(expr)
            return result
        
        # Create perturbed AnnData
        print(f"Creating perturbed AnnData (knockout: {', '.join(genes_list)})...")
        perturbed_adata = self._create_perturbed_adata(
            gene_indices, cell_indices, knockout_fn
        )

        # Tokenize perturbed data
        print("Tokenizing perturbed data...")
        # Note: adata.X has to be sparse
        perturbed_dataset = self.model.process_data(perturbed_adata)
        
        # Get embeddings
        print("Computing perturbed embeddings...")
        perturbed_embeddings = self._get_embeddings(perturbed_dataset)
        
        result = PerturbationResult(
            gene_names=genes_list,
            perturbation_type=PerturbationType.KNOCKOUT,
            original_embeddings=self.original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - self.original_embeddings,
            original_expressions=None,
            perturbed_expressions=None
        )
        
        if return_adata:
            result.perturbed_adata = perturbed_adata
            result.perturbed_dataset = perturbed_dataset
        
        return result
    
    def overexpression(
        self,
        genes: Union[str, List[str]],
        fold_change: float = 2.0,
        cell_indices: Optional[np.ndarray] = None,
        return_adata: bool = False
    ) -> PerturbationResult:
        """
        Perform gene overexpression by increasing expression.
        
        Args:
            genes: Gene name(s) to overexpress
            fold_change: Factor by which to increase expression
            cell_indices: Optional cell indices to perturb
            return_adata: Whether to return perturbed AnnData
        
        Returns:
            PerturbationResult object
        """
        gene_indices = self._get_gene_indices(genes)
        genes_list = genes if isinstance(genes, list) else [genes]
        
        if cell_indices is None:
            cell_indices = np.arange(self.adata.n_obs)
        
        # Create perturbation function
        def overexpress_fn(expr):
            return expr * fold_change
        
        # Create perturbed AnnData
        print(f"Creating perturbed AnnData (overexpression: {', '.join(genes_list)}, FC={fold_change})...")
        perturbed_adata = self._create_perturbed_adata(
            gene_indices, cell_indices, overexpress_fn
        )

        

        # Tokenize and get embeddings
        print("Tokenizing perturbed data...")
        perturbed_dataset = self.model.process_data(perturbed_adata, use_raw_counts=False)
        print("Computing perturbed embeddings...")
        perturbed_embeddings = self._get_embeddings(perturbed_dataset)
        
        result = PerturbationResult(
            gene_names=genes_list,
            perturbation_type=PerturbationType.OVEREXPRESSION,
            original_embeddings=self.original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - self.original_embeddings
        )
        
        if return_adata:
            result.perturbed_adata = perturbed_adata
            result.perturbed_dataset = perturbed_dataset
        
        return result
    
    def knockdown(
        self,
        genes: Union[str, List[str]],
        reduction_factor: float = 0.5,
        cell_indices: Optional[np.ndarray] = None,
        return_adata: bool = False
    ) -> PerturbationResult:
        """
        Perform gene knockdown by reducing expression.
        
        Args:
            genes: Gene name(s) to knockdown
            reduction_factor: Factor by which to reduce expression (0-1)
            cell_indices: Optional cell indices to perturb
            return_adata: Whether to return perturbed AnnData
        
        Returns:
            PerturbationResult object
        """
        if not 0 <= reduction_factor <= 1:
            raise ValueError("reduction_factor must be between 0 and 1")
        
        gene_indices = self._get_gene_indices(genes)
        genes_list = genes if isinstance(genes, list) else [genes]
        
        if cell_indices is None:
            cell_indices = np.arange(self.adata.n_obs)
        
        # Create perturbation function
        def knockdown_fn(expr):
            return expr * reduction_factor
        
        # Create perturbed AnnData
        print(f"Creating perturbed AnnData (knockdown: {', '.join(genes_list)}, factor={reduction_factor})...")
        perturbed_adata = self._create_perturbed_adata(
            gene_indices, cell_indices, knockdown_fn
        )
        
        # Tokenize and get embeddings
        print("Tokenizing perturbed data...")
        perturbed_dataset = self.model.process_data(perturbed_adata)
        print("Computing perturbed embeddings...")
        perturbed_embeddings = self._get_embeddings(perturbed_dataset)
        
        result = PerturbationResult(
            gene_names=genes_list,
            perturbation_type=PerturbationType.KNOCKDOWN,
            original_embeddings=self.original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - self.original_embeddings
        )
        
        if return_adata:
            result.perturbed_adata = perturbed_adata
            result.perturbed_dataset = perturbed_dataset
        
        return result
    
    def custom_perturbation(
        self,
        genes: Union[str, List[str]],
        perturbation_fn: Callable[[np.ndarray], np.ndarray],
        cell_indices: Optional[np.ndarray] = None,
        return_adata: bool = False
    ) -> PerturbationResult:
        """
        Apply a custom perturbation function to genes.
        
        Args:
            genes: Gene name(s) to perturb
            perturbation_fn: Function that takes and returns expression values
            cell_indices: Optional cell indices to perturb
            return_adata: Whether to return perturbed AnnData
        
        Returns:
            PerturbationResult object
        """
        gene_indices = self._get_gene_indices(genes)
        genes_list = genes if isinstance(genes, list) else [genes]
        
        if cell_indices is None:
            cell_indices = np.arange(self.adata.n_obs)
        
        # Create perturbed AnnData
        print(f"Creating perturbed AnnData (custom: {', '.join(genes_list)})...")
        perturbed_adata = self._create_perturbed_adata(
            gene_indices, cell_indices, perturbation_fn
        )
        
        # Tokenize and get embeddings
        print("Tokenizing perturbed data...")
        perturbed_dataset = self.model.process_data(perturbed_adata)
        print("Computing perturbed embeddings...")
        perturbed_embeddings = self._get_embeddings(perturbed_dataset)
        
        result = PerturbationResult(
            gene_names=genes_list,
            perturbation_type=PerturbationType.CUSTOM,
            original_embeddings=self.original_embeddings,
            perturbed_embeddings=perturbed_embeddings,
            embedding_shifts=perturbed_embeddings - self.original_embeddings
        )
        
        if return_adata:
            result.perturbed_adata = perturbed_adata
            result.perturbed_dataset = perturbed_dataset
        
        return result
    
    def batch_perturbation(
        self,
        gene_list: List[str],
        perturbation_type: str = 'knockout',
        cell_indices: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, PerturbationResult]:
        """
        Perform perturbations on multiple genes independently.
        
        Args:
            gene_list: List of genes to perturb individually
            perturbation_type: Type of perturbation ('knockout', 'overexpression', 'knockdown')
            cell_indices: Optional cell indices to perturb
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
        
        total = len(gene_list)
        for i, gene in enumerate(gene_list, 1):
            print(f"\n[{i}/{total}] Processing {gene}...")
            try:
                result = perturbation_method(gene, cell_indices=cell_indices, **kwargs)
                results[gene] = result
            except Exception as e:
                warnings.warn(f"Failed to perturb {gene}: {str(e)}")
        
        return results
    
    def combinatorial_perturbation(
        self,
        gene_combinations: List[List[str]],
        perturbation_type: str = 'knockout',
        cell_indices: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, PerturbationResult]:
        """
        Perform perturbations on combinations of genes.
        
        Args:
            gene_combinations: List of gene combinations to perturb together
            perturbation_type: Type of perturbation
            cell_indices: Optional cell indices to perturb
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
        
        total = len(gene_combinations)
        for i, gene_combo in enumerate(gene_combinations, 1):
            combo_key = "+".join(gene_combo)
            print(f"\n[{i}/{total}] Processing combination: {combo_key}...")
            try:
                result = perturbation_method(gene_combo, cell_indices=cell_indices, **kwargs)
                results[combo_key] = result
            except Exception as e:
                warnings.warn(f"Failed to perturb {combo_key}: {str(e)}")
        
        return results
