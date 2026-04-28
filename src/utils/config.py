"""
Configuration management
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for AIF Agent"""
    vfe_threshold: float = 2.0
    aif_iteration_ms: int = 1000
    bn_learning_algorithm: str = "hc"  # "hc" or "pc"
    mb_enabled: bool = True
    learning_batch_size: int = 10


@dataclass
class FogNodeConfig:
    """Configuration for Fog Node"""
    bn_learning_algorithm: str = "hc"
    coordination_interval_ms: int = 5000
    knowledge_transfer_interval: int = 10
    knowledge_transfer_threshold: float = 0.7
    offloading_enabled: bool = True
    offloading_threshold: float = 0.6
    cluster_model_enabled: bool = False


@dataclass
class SLOConfig:
    """SLO thresholds"""
    network_throughput_mb: float = 1.6
    processing_delay_ms: int = 33  # ~30fps
    success_rate: float = 0.95
    distance_threshold: int = 50


@dataclass
class KubernetesConfig:
    """Kubernetes configuration"""
    namespace: str = "default"
    context: str = "minikube"
    prometheus_url: str = "http://localhost:9090"


@dataclass
class Config:
    """Main configuration"""
    agent: AgentConfig
    fognode: FogNodeConfig
    slo: SLOConfig
    kubernetes: KubernetesConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            agent=AgentConfig(**data.get('active_inference', {})),
            fognode=FogNodeConfig(**data.get('fognode', {})),
            slo=SLOConfig(**data.get('slos', {})),
            kubernetes=KubernetesConfig(**data.get('kubernetes', {}))
        )
    
    @classmethod
    def default(cls) -> 'Config':
        """Get default configuration"""
        return cls(
            agent=AgentConfig(),
            fognode=FogNodeConfig(),
            slo=SLOConfig(),
            kubernetes=KubernetesConfig()
        )


