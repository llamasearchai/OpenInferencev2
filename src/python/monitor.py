"""
Advanced Performance Monitoring System
Real-time metrics collection and analysis
"""
import time
import threading
import psutil
import GPUtil
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import json
import logging
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    utilization: float
    memory_used: float
    memory_total: float
    temperature: float
    power_usage: float
    clock_speed: int
@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
@dataclass
class InferenceMetrics:
    """Inference-specific metrics"""
    request_id: str
    latency: float
    tokens_generated: int
    tokens_per_second: float
    batch_size: int
    gpu_memory_peak: float
    timestamp: float = field(default_factory=time.time)
class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, history_size: int = 1000, sample_interval: float = 1.0):
        self.history_size = history_size
        self.sample_interval = sample_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics storage
        self.gpu_metrics_history = deque(maxlen=history_size)
        self.system_metrics_history = deque(maxlen=history_size)
        self.inference_metrics_history = deque(maxlen=history_size)
        
        # Real-time metrics
        self.current_gpu_metrics = []
        self.current_system_metrics = SystemMetrics(0, 0, {}, {})
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = defaultdict(list)
        
        # Performance thresholds
        self.thresholds = {
            'gpu_utilization_high': 90.0,
            'gpu_memory_high': 85.0,
            'gpu_temperature_high': 85.0,
            'cpu_usage_high': 80.0,
            'memory_usage_high': 90.0,
            'latency_high': 5.0,  # seconds
            'tokens_per_second_low': 10.0
        }
        
        # Initialize GPU monitoring
        self._initialize_gpu_monitoring()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities"""
        try:
            if NVML_AVAILABLE:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = []
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                self.logger.info(f"Initialized NVML monitoring for {self.gpu_count} GPUs")
            else:
                # Fallback to GPUtil
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)
                self.logger.info(f"Using GPUtil for {self.gpu_count} GPUs")
                
        except Exception as e:
            self.logger.warning(f"GPU monitoring initialization failed: {e}")
            self.gpu_count = 0
            
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                gpu_metrics = self._collect_gpu_metrics()
                system_metrics = self._collect_system_metrics()
                
                # Store metrics
                if gpu_metrics:
                    self.gpu_metrics_history.append(gpu_metrics)
                    self.current_gpu_metrics = gpu_metrics
                    
                self.system_metrics_history.append(system_metrics)
                self.current_system_metrics = system_metrics
                
                # Check thresholds and trigger callbacks
                self._check_thresholds(gpu_metrics, system_metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.sample_interval)
                
    def _collect_gpu_metrics(self) -> Optional[List[GPUMetrics]]:
        """Collect GPU performance metrics"""
        if self.gpu_count == 0:
            return None
            
        gpu_metrics = []
        
        try:
            if NVML_AVAILABLE:
                for i, handle in enumerate(self.gpu_handles):
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = 0.0
                        
                    # Get clock speeds
                    try:
                        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    except:
                        clock = 0
                        
                    metrics = GPUMetrics(
                        utilization=util.gpu,
                        memory_used=mem_info.used / 1024**3,  # GB
                        memory_total=mem_info.total / 1024**3,  # GB
                        temperature=temp,
                        power_usage=power,
                        clock_speed=clock
                    )
                    gpu_metrics.append(metrics)
                    
            else:
                # Fallback to GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics = GPUMetrics(
                        utilization=gpu.load * 100,
                        memory_used=gpu.memoryUsed / 1024,  # GB
                        memory_total=gpu.memoryTotal / 1024,  # GB
                        temperature=gpu.temperature,
                        power_usage=0.0,  # Not available in GPUtil
                        clock_speed=0  # Not available in GPUtil
                    )
                    gpu_metrics.append(metrics)
                    
        except Exception as e:
            self.logger.error(f"GPU metrics collection failed: {e}")
            return None
            
        return gpu_metrics
        
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes_per_sec': disk_io.read_bytes if disk_io else 0,
                'write_bytes_per_sec': disk_io.write_bytes if disk_io else 0
            }
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_metrics = {
                'bytes_sent_per_sec': net_io.bytes_sent if net_io else 0,
                'bytes_recv_per_sec': net_io.bytes_recv if net_io else 0
            }
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_metrics,
                network_io=net_metrics
            )
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return SystemMetrics(0, 0, {}, {})
            
    def record_inference_metrics(self, metrics: InferenceMetrics):
        """Record inference-specific metrics"""
        self.inference_metrics_history.append(metrics)
        
        # Check inference-specific thresholds
        self._check_inference_thresholds(metrics)
        
    def _check_thresholds(self, gpu_metrics: Optional[List[GPUMetrics]], 
                         system_metrics: SystemMetrics):
        """Check performance thresholds and trigger alerts"""
        alerts = []
        
        # GPU threshold checks
        if gpu_metrics:
            for i, gpu in enumerate(gpu_metrics):
                if gpu.utilization > self.thresholds['gpu_utilization_high']:
                    alerts.append(f"GPU {i} utilization high: {gpu.utilization:.1f}%")
                    
                memory_percent = (gpu.memory_used / gpu.memory_total) * 100
                if memory_percent > self.thresholds['gpu_memory_high']:
                    alerts.append(f"GPU {i} memory high: {memory_percent:.1f}%")
                    
                if gpu.temperature > self.thresholds['gpu_temperature_high']:
                    alerts.append(f"GPU {i} temperature high: {gpu.temperature}°C")
                    
        # System threshold checks
        if system_metrics.cpu_usage > self.thresholds['cpu_usage_high']:
            alerts.append(f"CPU usage high: {system_metrics.cpu_usage:.1f}%")
            
        if system_metrics.memory_usage > self.thresholds['memory_usage_high']:
            alerts.append(f"Memory usage high: {system_metrics.memory_usage:.1f}%")
            
        # Trigger callbacks
        for alert in alerts:
            self._trigger_callbacks('threshold_alert', alert)
            
    def _check_inference_thresholds(self, metrics: InferenceMetrics):
        """Check inference-specific thresholds"""
        alerts = []
        
        if metrics.latency > self.thresholds['latency_high']:
            alerts.append(f"High latency: {metrics.latency:.3f}s for request {metrics.request_id}")
            
        if metrics.tokens_per_second < self.thresholds['tokens_per_second_low']:
            alerts.append(f"Low throughput: {metrics.tokens_per_second:.1f} tokens/s for request {metrics.request_id}")
            
        for alert in alerts:
            self._trigger_callbacks('inference_alert', alert)
            
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific events"""
        self.callbacks[event_type].append(callback)
        
    def _trigger_callbacks(self, event_type: str, data):
        """Trigger callbacks for specific event type"""
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
                
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'system': {
                'cpu_usage': self.current_system_metrics.cpu_usage,
                'memory_usage': self.current_system_metrics.memory_usage,
                'disk_io': self.current_system_metrics.disk_io,
                'network_io': self.current_system_metrics.network_io
            }
        }
        
        if self.current_gpu_metrics:
            metrics['gpus'] = []
            for i, gpu in enumerate(self.current_gpu_metrics):
                metrics['gpus'].append({
                    'id': i,
                    'utilization': gpu.utilization,
                    'memory_used_gb': gpu.memory_used,
                    'memory_total_gb': gpu.memory_total,
                    'memory_percent': (gpu.memory_used / gpu.memory_total) * 100,
                    'temperature': gpu.temperature,
                    'power_usage': gpu.power_usage,
                    'clock_speed': gpu.clock_speed
                })
                
        return metrics
        
    def get_inference_statistics(self, time_window: Optional[float] = None) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_metrics_history:
            return {}
            
        # Filter by time window if specified
        current_time = time.time()
        if time_window:
            filtered_metrics = [
                m for m in self.inference_metrics_history 
                if current_time - m.timestamp <= time_window
            ]
        else:
            filtered_metrics = list(self.inference_metrics_history)
            
        if not filtered_metrics:
            return {}
            
        # Calculate statistics
        latencies = [m.latency for m in filtered_metrics]
        throughputs = [m.tokens_per_second for m in filtered_metrics]
        batch_sizes = [m.batch_size for m in filtered_metrics]
        
        return {
            'total_requests': len(filtered_metrics),
            'avg_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)],
            'avg_throughput': sum(throughputs) / len(throughputs),
            'max_throughput': max(throughputs),
            'avg_batch_size': sum(batch_sizes) / len(batch_sizes),
            'time_window': time_window or 'all_time'
        }
        
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export collected metrics to file"""
        data = {
            'timestamp': time.time(),
            'gpu_metrics': [
                {
                    'timestamp': i * self.sample_interval,
                    'gpus': [
                        {
                            'utilization': gpu.utilization,
                            'memory_used': gpu.memory_used,
                            'memory_total': gpu.memory_total,
                            'temperature': gpu.temperature,
                            'power_usage': gpu.power_usage,
                            'clock_speed': gpu.clock_speed
                        }
                        for gpu in metrics
                    ] if metrics else []
                }
                for i, metrics in enumerate(self.gpu_metrics_history)
            ],
            'system_metrics': [
                {
                    'timestamp': metrics.timestamp,
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'disk_io': metrics.disk_io,
                    'network_io': metrics.network_io
                }
                for metrics in self.system_metrics_history
            ],
            'inference_metrics': [
                {
                    'request_id': metrics.request_id,
                    'timestamp': metrics.timestamp,
                    'latency': metrics.latency,
                    'tokens_generated': metrics.tokens_generated,
                    'tokens_per_second': metrics.tokens_per_second,
                    'batch_size': metrics.batch_size,
                    'gpu_memory_peak': metrics.gpu_memory_peak
                }
                for metrics in self.inference_metrics_history
            ]
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Metrics exported to {filepath}")
        
    def get_gpu_temperature(self, gpu_id: int = 0) -> float:
        """Get current GPU temperature"""
        if self.current_gpu_metrics and gpu_id < len(self.current_gpu_metrics):
            return self.current_gpu_metrics[gpu_id].temperature
        return 0.0