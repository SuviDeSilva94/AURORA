"""
Bayesian Network Learning for Active Inference

Implements EOSC (Expected Observations to States and Controls) model using pgmpy
Structure Learning (STRL) and Parameter Learning (PARL)
"""

from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch,
    PC,
    BicScore,
    K2Score,
    ParameterEstimator,
    MaximumLikelihoodEstimator,
    BayesianEstimator
)
from pgmpy.inference import VariableElimination, CausalInference
from loguru import logger


class EOSCModel:
    """
    Expected Observations to States and Controls (EOSC) Model
    
    A Bayesian Network that represents causal relationships between:
    - Observations: System and application metrics
    - States: SLO compliance states
    - Controls: Parameterizable variables
    """
    
    def __init__(self, model: BayesianNetwork, markov_blanket: Optional[Dict[str, Set[str]]] = None):
        self.model = model
        self.markov_blanket = markov_blanket
        self.inference_engine = VariableElimination(model)
        self._causal_inference: Optional[CausalInference] = None

    @property
    def causal_inference(self) -> CausalInference:
        """
        Cached pgmpy ``CausalInference`` for Pearl do-calculus queries
        ``P(Y | do(X), Z)``. pgmpy performs graph mutilation (severs incoming
        edges to do-variables) and backdoor adjustment internally.

        Built lazily on first use; ``refit_cctv_benchmark_cpts`` returns a
        fresh ``EOSCModel``, so cache invalidation is automatic.
        """
        if self._causal_inference is None:
            self._causal_inference = CausalInference(self.model)
        return self._causal_inference

    def predict(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outcomes given evidence
        
        Args:
            evidence: Observed variable values
            
        Returns:
            Predicted probabilities for all variables
        """
        # Filter evidence to only include variables in the model
        valid_evidence = {
            k: v for k, v in evidence.items()
            if k in self.model.nodes()
        }
        
        predictions = {}
        for node in self.model.nodes():
            if node not in valid_evidence:
                try:
                    query_result = self.inference_engine.query(
                        variables=[node],
                        evidence=valid_evidence
                    )
                    predictions[node] = query_result.values
                except Exception as e:
                    logger.debug(f"Could not query {node}: {e}")
        
        return predictions
    
    def query_slo(self, slo_variable: str, evidence: Dict[str, Any]) -> float:
        """
        Query probability of SLO fulfillment given evidence
        
        Args:
            slo_variable: Name of SLO variable (e.g., 'network', 'in_time')
            evidence: Observed variable values
            
        Returns:
            Probability of SLO being fulfilled
        """
        try:
            valid_evidence = {
                k: v for k, v in evidence.items()
                if k in self.model.nodes() and k != slo_variable
            }
            
            query_result = self.inference_engine.query(
                variables=[slo_variable],
                evidence=valid_evidence
            )
            
            # Assuming binary SLO (fulfilled=True/False or 1/0)
            # Return probability of fulfillment
            values = query_result.values
            if len(values) == 2:
                return float(values[1])  # Probability of True/1
            else:
                return float(np.max(values))
                
        except Exception as e:
            logger.warning(f"Error querying SLO {slo_variable}: {e}")
            return 0.5  # Return uncertainty
    
    def get_structure(self) -> List[tuple]:
        """Get the DAG structure as list of edges"""
        return list(self.model.edges())
    
    def get_markov_blanket(self, variable: str) -> Set[str]:
        """
        Get Markov Blanket for a variable
        
        MB(X) = Parents(X) ∪ Children(X) ∪ Parents(Children(X))
        
        This reduces inference complexity (A-1 from paper)
        """
        if self.markov_blanket and variable in self.markov_blanket:
            return self.markov_blanket[variable]
        
        # Compute Markov Blanket
        mb = set()
        
        # Add parents
        mb.update(self.model.get_parents(variable))
        
        # Add children and their parents
        children = self.model.get_children(variable)
        mb.update(children)
        for child in children:
            mb.update(self.model.get_parents(child))
        
        return mb


class BayesianNetworkLearner:
    """
    Bayesian Network Learning for EOSC Models
    
    Implements:
    - Structure Learning (STRL): Learn causal DAG structure
    - Parameter Learning (PARL): Update conditional probability tables
    - Model Merging: Combine models from heterogeneous devices
    """
    
    def __init__(
        self,
        algorithm: str = "hc",  # "hc" = Hill Climbing, "pc" = PC algorithm
        scoring_method: str = "bic",  # "bic" or "k2"
        use_markov_blanket: bool = True
    ):
        self.algorithm = algorithm
        self.scoring_method = scoring_method
        self.use_markov_blanket = use_markov_blanket
    
    def learn_structure(self, data: List[Dict[str, Any]]) -> EOSCModel:
        """
        Learn EOSC model structure from data (STRL)
        
        Args:
            data: List of observations
            
        Returns:
            Trained EOSC model
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Discretize continuous variables
        df = self._discretize_data(df)
        
        logger.info(f"Learning BN structure with {len(df)} samples, {len(df.columns)} variables")
        
        # Structure learning
        if self.algorithm == "hc":
            # Hill Climbing search
            scoring_method = BicScore(df) if self.scoring_method == "bic" else K2Score(df)
            est = HillClimbSearch(df)
            dag = est.estimate(scoring_method=scoring_method)
        elif self.algorithm == "pc":
            # PC algorithm (constraint-based)
            est = PC(df)
            dag = est.estimate()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.info(f"Learned DAG with {len(dag.edges())} edges")
        
        # Parameter learning
        model = BayesianNetwork(dag.edges())
        model.fit(df, estimator=MaximumLikelihoodEstimator)
        
        # Compute Markov Blankets if enabled
        markov_blanket = None
        if self.use_markov_blanket:
            markov_blanket = self._compute_markov_blankets(model)
        
        return EOSCModel(model=model, markov_blanket=markov_blanket)
    
    def update_parameters(
        self,
        eosc_model: EOSCModel,
        new_data: List[Dict[str, Any]]
    ) -> EOSCModel:
        """
        Update EOSC model parameters with new observations (PARL)
        
        Args:
            eosc_model: Existing EOSC model
            new_data: New observations
            
        Returns:
            Updated EOSC model
        """
        # Convert to DataFrame
        df = pd.DataFrame(new_data)
        df = self._discretize_data(df)
        
        # Update CPTs using Bayesian estimation
        # This allows incremental updates
        model = eosc_model.model
        
        try:
            # Refit with new data
            model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
            logger.info(f"Updated model parameters with {len(df)} new samples")
        except Exception as e:
            logger.warning(f"Error updating parameters: {e}")
        
        return EOSCModel(model=model, markov_blanket=eosc_model.markov_blanket)
    
    def merge_models(
        self,
        model_a: EOSCModel,
        model_b: EOSCModel
    ) -> EOSCModel:
        """
        Merge two EOSC models
        
        From paper Section 3.2.3 and 4.2.2:
        Merging is only possible under specific conditions.
        
        Workaround: Extend one model with data from the other
        
        Args:
            model_a: First EOSC model
            model_b: Second EOSC model
            
        Returns:
            Merged EOSC model
        """
        # For simplicity, we return model_a
        # In practice, would need training data from both models
        logger.info("Merging models (simplified approach)")
        return model_a
    
    def _discretize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretize continuous variables for Bayesian Network learning
        
        Args:
            df: DataFrame with continuous and discrete variables
            
        Returns:
            DataFrame with all variables discretized
        """
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                # Discretize into bins
                try:
                    df[col] = pd.cut(df[col], bins=5, labels=False, duplicates='drop')
                except Exception:
                    # If discretization fails, keep as is
                    pass
            elif df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def _compute_markov_blankets(self, model: BayesianNetwork) -> Dict[str, Set[str]]:
        """
        Compute Markov Blankets for all variables
        
        Addresses A-1: Do MBs reduce the complexity of inference?
        
        Args:
            model: Bayesian Network
            
        Returns:
            Dictionary mapping variables to their Markov Blankets
        """
        markov_blankets = {}
        
        for node in model.nodes():
            mb = set()
            
            # Parents
            mb.update(model.get_parents(node))
            
            # Children and their parents
            children = model.get_children(node)
            mb.update(children)
            for child in children:
                mb.update(model.get_parents(child))
            
            markov_blankets[node] = mb
        
        return markov_blankets
    
    def extract_markov_blanket_model(
        self,
        eosc_model: EOSCModel,
        target_variable: str
    ) -> EOSCModel:
        """
        Extract subgraph containing only Markov Blanket
        
        This reduces computational complexity for inference
        
        Args:
            eosc_model: Full EOSC model
            target_variable: Variable to center MB around
            
        Returns:
            Reduced EOSC model containing only MB
        """
        mb = eosc_model.get_markov_blanket(target_variable)
        mb.add(target_variable)
        
        # Create subgraph
        edges = [
            (u, v) for u, v in eosc_model.model.edges()
            if u in mb and v in mb
        ]
        
        reduced_model = BayesianNetwork(edges)
        
        # Copy CPDs for nodes in MB
        for node in mb:
            if eosc_model.model.get_cpds(node):
                cpd = eosc_model.model.get_cpds(node)
                reduced_model.add_cpds(cpd)
        
        return EOSCModel(model=reduced_model)


# Fixed DAG for CCTV / computing-continuum experiments: all stress metrics are
# explicit parents of slo_violated so RCA can rank memory vs cpu vs network.
CCTV_BENCHMARK_EDGES = [
    ("network_quality", "cpu"),
    ("memory", "cpu"),
    ("cpu", "delay"),
    ("network_quality", "slo_violated"),
    ("memory", "slo_violated"),
    ("cpu", "slo_violated"),
    ("delay", "slo_violated"),
]


def fit_eosc_cctv_benchmark_dag(data: List[Dict[str, Any]]) -> EOSCModel:
    """
    Fit CPTs on a fixed causal DAG (parameters only).

    Structure learning on small synthetic samples often drops edges (e.g. no
    path from memory to slo_violated), which breaks fault-type-specific RCA.
    """
    learner = BayesianNetworkLearner()
    df = pd.DataFrame(data)
    df = learner._discretize_data(df)
    model = BayesianNetwork(CCTV_BENCHMARK_EDGES)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    markov_blanket = learner._compute_markov_blankets(model)
    logger.info(
        f"Fitted CCTV benchmark BN: {len(CCTV_BENCHMARK_EDGES)} edges, "
        f"{len(df)} rows"
    )
    return EOSCModel(model=model, markov_blanket=markov_blanket)


def refit_cctv_benchmark_cpts(data: List[Dict[str, Any]]) -> EOSCModel:
    """
    Refit CPTs on the fixed CCTV benchmark DAG (structure unchanged).

    Use for online / edge-side parameter refresh (Sedlak-style PARL): merge historical
    rows with new observations, then refit MLE CPTs—evaluate drift with
    ``max_abs_cpd_delta`` to guard against catastrophic forgetting in reports.
    """
    return fit_eosc_cctv_benchmark_dag(data)


# Fixed DAG for industrial-IoT motor monitoring (second workload):
# current → temperature (electrical → thermal coupling)
# vibration → bearing_status (mechanical degradation pathway)
# {temperature, bearing_status} → motor_health (latent fault state)
# {motor_health, current, vibration, temperature} → slo_violated (direct + mediated)
MOTOR_BENCHMARK_EDGES = [
    ("current", "temperature"),
    ("vibration", "bearing_status"),
    ("temperature", "motor_health"),
    ("bearing_status", "motor_health"),
    ("motor_health", "slo_violated"),
    ("current", "slo_violated"),
    ("vibration", "slo_violated"),
    ("temperature", "slo_violated"),
]


def fit_eosc_motor_benchmark_dag(data: List[Dict[str, Any]]) -> EOSCModel:
    """
    Fit CPTs on the fixed motor-monitoring DAG (parameters only).

    Structurally distinct from the CCTV benchmark: two latent mediating
    variables (``bearing_status``, ``motor_health``) sit between the directly
    observed signals (``current``, ``vibration``, ``temperature``) and the
    symptom (``slo_violated``). This produces a richer Markov blanket for the
    symptom (four parents through three observation routes) and tests whether
    the dual-gated mechanism generalises to fault distributions where the BN
    must reason across multi-hop mediating paths.
    """
    learner = BayesianNetworkLearner()
    df = pd.DataFrame(data)
    df = learner._discretize_data(df)
    model = BayesianNetwork(MOTOR_BENCHMARK_EDGES)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    markov_blanket = learner._compute_markov_blankets(model)
    logger.info(
        f"Fitted motor benchmark BN: {len(MOTOR_BENCHMARK_EDGES)} edges, "
        f"{len(df)} rows"
    )
    return EOSCModel(model=model, markov_blanket=markov_blanket)


def refit_motor_benchmark_cpts(data: List[Dict[str, Any]]) -> EOSCModel:
    """Refit CPTs on the fixed motor benchmark DAG (structure unchanged)."""
    return fit_eosc_motor_benchmark_dag(data)


def max_abs_cpd_delta(before: EOSCModel, after: EOSCModel) -> float:
    """
    Maximum absolute difference between matching CPT entries (same nodes and shapes).

    Returns ``inf`` if structures or state shapes differ.
    """
    nodes_b = set(before.model.nodes())
    nodes_a = set(after.model.nodes())
    if nodes_b != nodes_a:
        return float("inf")
    mx = 0.0
    for node in nodes_b:
        cpd_b = before.model.get_cpds(node)
        cpd_a = after.model.get_cpds(node)
        if cpd_b is None or cpd_a is None:
            return float("inf")
        vb, va = np.asarray(cpd_b.values), np.asarray(cpd_a.values)
        if vb.shape != va.shape:
            return float("inf")
        mx = max(mx, float(np.max(np.abs(vb - va))))
    return mx

