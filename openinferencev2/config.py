"""
Configuration management for OpenInferencev2
"""
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

class Config:
    """OpenInferencev2 configuration management"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.defaults = {
            # Hardware Configuration
            'num_gpus': 1,
            'max_batch_size': 32,
            'max_sequence_length': 2048,
            'max_queue_size': 1000,
            
            # Optimization Settings
            'use_fp16': True,
            'use_flash_attention': True,
            'use_cuda_graphs': True,
            'use_tensorrt': False,
            'use_torch_compile': False,
            'quantization': None,  # Options: None, 'int8', 'int4'
            
            # Parallelism Configuration
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1,
            'moe_parallel_size': 1,
            
            # Memory Management
            'kv_cache_size_gb': 8.0,
            'memory_pool_size_gb': 16.0,
            'max_memory_usage': 0.9,
            
            # Performance Tuning
            'request_timeout': 30.0,
            'batch_timeout': 0.1,
            'warmup_steps': 3,
            
            # Monitoring
            'enable_monitoring': True,
            'metrics_port': 9090,
            'log_level': 'INFO',
            
            # Distributed Backend
            'distributed_backend': 'nccl',
            'master_addr': 'localhost',
            'master_port': '12355',
            
            # Model Configuration
            'trust_remote_code': True,
            'revision': 'main',
            'cache_dir': None,
        }
        
        # Initialize with defaults
        self._config = self.defaults.copy()
        
        # Override with provided config
        if config_dict:
            self._config.update(config_dict)
            
    def __getattr__(self, name: str) -> Any:
        """Allow access to config values as attributes"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration has no attribute '{name}'")
        
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting config values as attributes"""
        if name.startswith('_') or name in ['defaults']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, '_config'):
                self._config[name] = value
            else:
                super().__setattr__(name, value)
                
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._config[key]
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting"""
        self._config[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._config.get(key, default)
        
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self._config.update(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()
        
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON or YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
        self.update(config_data)
        
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON or YAML file"""
        config_path = Path(config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Hardware validation
        if self.num_gpus < 1:
            errors.append("num_gpus must be at least 1")
            
        if self.max_batch_size < 1:
            errors.append("max_batch_size must be at least 1")
            
        if self.max_sequence_length < 1:
            errors.append("max_sequence_length must be at least 1")
            
        # Parallelism validation
        if self.tensor_parallel_size < 1:
            errors.append("tensor_parallel_size must be at least 1")
            
        if self.pipeline_parallel_size < 1:
            errors.append("pipeline_parallel_size must be at least 1")
            
        total_parallel = self.tensor_parallel_size * self.pipeline_parallel_size
        if total_parallel > self.num_gpus:
            errors.append(f"Total parallelism ({total_parallel}) cannot exceed number of GPUs ({self.num_gpus})")
            
        # Memory validation
        if self.kv_cache_size_gb <= 0:
            errors.append("kv_cache_size_gb must be positive")
            
        if self.memory_pool_size_gb <= 0:
            errors.append("memory_pool_size_gb must be positive")
            
        if not (0.0 < self.max_memory_usage <= 1.0):
            errors.append("max_memory_usage must be between 0.0 and 1.0")
            
        # Timeout validation
        if self.request_timeout <= 0:
            errors.append("request_timeout must be positive")
            
        if self.batch_timeout <= 0:
            errors.append("batch_timeout must be positive")
            
        # Quantization validation
        if self.quantization and self.quantization not in ['int8', 'int4']:
            errors.append("quantization must be None, 'int8', or 'int4'")
            
        return errors
        
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0
        
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"Config({self._config})"
        
    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = ["OpenInferencev2 Configuration:"]
        
        sections = {
            "Hardware": ['num_gpus', 'max_batch_size', 'max_sequence_length'],
            "Optimization": ['use_fp16', 'use_flash_attention', 'use_cuda_graphs', 'quantization'],
            "Parallelism": ['tensor_parallel_size', 'pipeline_parallel_size', 'moe_parallel_size'],
            "Memory": ['kv_cache_size_gb', 'memory_pool_size_gb', 'max_memory_usage'],
            "Performance": ['request_timeout', 'batch_timeout', 'warmup_steps'],
        }
        
        for section_name, keys in sections.items():
            lines.append(f"\n{section_name}:")
            for key in keys:
                if key in self._config:
                    lines.append(f"  {key}: {self._config[key]}")
                    
        return "\n".join(lines)
        
def create_default_config() -> Config:
    """Create a default configuration"""
    return Config()
    
def create_production_config(num_gpus: int = 4) -> Config:
    """Create a production-ready configuration"""
    return Config({
        'num_gpus': num_gpus,
        'max_batch_size': 64,
        'max_sequence_length': 4096,
        'use_fp16': True,
        'use_flash_attention': True,
        'use_cuda_graphs': True,
        'tensor_parallel_size': min(num_gpus, 4),
        'kv_cache_size_gb': 16.0,
        'memory_pool_size_gb': 32.0,
        'enable_monitoring': True,
        'log_level': 'INFO'
    })
    
def create_development_config() -> Config:
    """Create a development configuration"""
    return Config({
        'num_gpus': 1,
        'max_batch_size': 8,
        'max_sequence_length': 2048,
        'use_fp16': False,  # Easier debugging
        'use_cuda_graphs': False,  # Faster startup
        'kv_cache_size_gb': 4.0,
        'memory_pool_size_gb': 8.0,
        'enable_monitoring': True,
        'log_level': 'DEBUG'
    })
