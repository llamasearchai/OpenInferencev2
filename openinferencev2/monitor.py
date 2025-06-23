"""
Performance monitoring for OpenInferencev2
Real-time metrics collection and system monitoring
"""
import asyncio
import time
import psutil
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import json

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Metrics for a single inference"""
    request_id: str
    latency: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage: float
    gpu_utilization: float
    timestamp: float
    success: bool
    error_message: str = ""

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: List[float]
    gpu_memory: List[float]
    disk_io: Dict[str, float]
    network_io: Dict[str, float]

class PerformanceMonitor:
    """Real-time performance monitoring and metrics collection"""
    
    def __init__(self, history_size: int = 1000, collection_interval: float = 1.0):
        self.history_size = history_size
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics storage
        self.inference_metrics = deque(maxlen=history_size)
        self.system_metrics = deque(maxlen=history_size)
        
        # Real-time statistics
        self.current_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0,
            'avg_tokens_per_second': 0.0,
            'total_tokens_generated': 0,
            'requests_per_second': 0.0,
            'system_load': 0.0,
            'memory_usage_percent': 0.0,
            'gpu_utilization': 0.0,
            'gpu_memory_usage': 0.0
        }
        
        # Alerts and thresholds
        self.alert_thresholds = {
            'high_latency': 5.0,  # seconds
            'low_throughput': 10.0,  # tokens/second
            'high_memory': 90.0,  # percent
            'high_gpu_memory': 90.0,  # percent
            'high_cpu': 80.0,  # percent
            'high_error_rate': 0.1  # 10%
        }
        
        self.active_alerts = set()
        self.alert_callbacks = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        
        # Performance windows for trend analysis
        self.short_window = deque(maxlen=60)  # 1 minute
        self.medium_window = deque(maxlen=300)  # 5 minutes
        self.long_window = deque(maxlen=3600)  # 1 hour
        
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        self.executor.shutdown(wait=True)
        self.logger.info("Performance monitoring stopped")
        
    async def record_inference(self, response) -> None:
        """Record metrics for a completed inference"""
        try:
            # Get system metrics at time of inference
            gpu_util, gpu_mem = self._get_gpu_metrics()
            memory_usage = psutil.virtual_memory().percent
            
            metrics = InferenceMetrics(
                request_id=response.id,
                latency=response.latency,
                tokens_generated=len(response.tokens) if response.tokens else 0,
                tokens_per_second=response.tokens_per_second,
                memory_usage=memory_usage,
                gpu_utilization=gpu_util[0] if gpu_util else 0.0,
                timestamp=time.time(),
                success=response.success,
                error_message=response.error_message
            )
            
            with self._lock:
                self.inference_metrics.append(metrics)
                self._update_stats(metrics)
                self._check_alerts(metrics)
                
        except Exception as e:
            self.logger.error(f"Failed to record inference metrics: {e}")
            
    def _update_stats(self, metrics: InferenceMetrics):
        """Update running statistics"""
        self.current_stats['total_requests'] += 1
        
        if metrics.success:
            self.current_stats['successful_requests'] += 1
            self.current_stats['total_tokens_generated'] += metrics.tokens_generated
            
            # Update average latency
            total_successful = self.current_stats['successful_requests']
            current_avg = self.current_stats['avg_latency']
            self.current_stats['avg_latency'] = (
                (current_avg * (total_successful - 1) + metrics.latency) / total_successful
            )
            
            # Update average tokens per second
            current_tps = self.current_stats['avg_tokens_per_second']
            self.current_stats['avg_tokens_per_second'] = (
                (current_tps * (total_successful - 1) + metrics.tokens_per_second) / total_successful
            )
        else:
            self.current_stats['failed_requests'] += 1
            
        # Update requests per second (based on last minute)
        current_time = time.time()
        self.short_window.append((current_time, 1))  # (timestamp, request_count)
        
        # Remove old entries
        while self.short_window and current_time - self.short_window[0][0] > 60:
            self.short_window.popleft()
            
        if self.short_window:
            total_requests_in_window = sum(req[1] for req in self.short_window)
            time_span = current_time - self.short_window[0][0]
            self.current_stats['requests_per_second'] = total_requests_in_window / max(time_span, 1.0)
            
    def _check_alerts(self, metrics: InferenceMetrics):
        """Check for alert conditions"""
        alerts_to_trigger = set()
        
        # Check latency
        if metrics.latency > self.alert_thresholds['high_latency']:
            alerts_to_trigger.add('high_latency')
            
        # Check throughput
        if metrics.tokens_per_second < self.alert_thresholds['low_throughput']:
            alerts_to_trigger.add('low_throughput')
            
        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds['high_memory']:
            alerts_to_trigger.add('high_memory')
            
        # Check GPU memory
        if metrics.gpu_utilization > self.alert_thresholds['high_gpu_memory']:
            alerts_to_trigger.add('high_gpu_memory')
            
        # Check error rate
        total_requests = self.current_stats['total_requests']
        failed_requests = self.current_stats['failed_requests']
        if total_requests > 0:
            error_rate = failed_requests / total_requests
            if error_rate > self.alert_thresholds['high_error_rate']:
                alerts_to_trigger.add('high_error_rate')
                
        # Trigger new alerts
        new_alerts = alerts_to_trigger - self.active_alerts
        for alert in new_alerts:
            self._trigger_alert(alert, metrics)
            
        # Clear resolved alerts
        resolved_alerts = self.active_alerts - alerts_to_trigger
        for alert in resolved_alerts:
            self._clear_alert(alert)
            
        self.active_alerts = alerts_to_trigger
        
    def _trigger_alert(self, alert_type: str, metrics: InferenceMetrics):
        """Trigger an alert"""
        alert_data = {
            'type': alert_type,
            'timestamp': time.time(),
            'metrics': asdict(metrics),
            'current_stats': self.current_stats.copy()
        }
        
        self.logger.warning(f"Alert triggered: {alert_type}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
                
    def _clear_alert(self, alert_type: str):
        """Clear a resolved alert"""
        self.logger.info(f"Alert cleared: {alert_type}")
        
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                with self._lock:
                    self.system_metrics.append(system_metrics)
                    self._update_system_stats(system_metrics)
                    
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
                
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive operations in thread pool
        cpu_usage = await loop.run_in_executor(self.executor, psutil.cpu_percent, 1.0)
        memory = await loop.run_in_executor(self.executor, psutil.virtual_memory)
        
        # Get GPU metrics
        gpu_usage, gpu_memory = self._get_gpu_metrics()
        
        # Get disk I/O
        disk_io = psutil.disk_io_counters()
        disk_stats = {
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        net_stats = {
            'bytes_sent': net_io.bytes_sent if net_io else 0,
            'bytes_recv': net_io.bytes_recv if net_io else 0
        }
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            disk_io=disk_stats,
            network_io=net_stats
        )
        
    def _get_gpu_metrics(self) -> Tuple[List[float], List[float]]:
        """Get GPU utilization and memory usage"""
        gpu_usage = []
        gpu_memory = []
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage.append(gpu.load * 100)
                    gpu_memory.append(gpu.memoryUtil * 100)
            except Exception as e:
                self.logger.debug(f"Failed to get GPU metrics: {e}")
                
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    # PyTorch doesn't directly provide utilization, so we estimate
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    
                    gpu_usage.append(50.0)  # Placeholder - actual utilization hard to get
                    gpu_memory.append((memory_reserved / memory_total) * 100)
            except Exception as e:
                self.logger.debug(f"Failed to get GPU metrics via PyTorch: {e}")
                
        return gpu_usage, gpu_memory
        
    def _update_system_stats(self, metrics: SystemMetrics):
        """Update system-level statistics"""
        self.current_stats['system_load'] = metrics.cpu_usage
        self.current_stats['memory_usage_percent'] = metrics.memory_usage
        
        if metrics.gpu_usage:
            self.current_stats['gpu_utilization'] = sum(metrics.gpu_usage) / len(metrics.gpu_usage)
            
        if metrics.gpu_memory:
            self.current_stats['gpu_memory_usage'] = sum(metrics.gpu_memory) / len(metrics.gpu_memory)
            
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self._lock:
            return self.current_stats.copy()
            
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics over specified time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            # Filter metrics within time window
            recent_metrics = [
                m for m in self.inference_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {'message': 'No metrics available for specified time window'}
                
            # Calculate summary statistics
            successful_metrics = [m for m in recent_metrics if m.success]
            
            summary = {
                'time_window_minutes': window_minutes,
                'total_requests': len(recent_metrics),
                'successful_requests': len(successful_metrics),
                'failed_requests': len(recent_metrics) - len(successful_metrics),
                'success_rate': len(successful_metrics) / len(recent_metrics) if recent_metrics else 0,
            }
            
            if successful_metrics:
                latencies = [m.latency for m in successful_metrics]
                throughputs = [m.tokens_per_second for m in successful_metrics]
                
                summary.update({
                    'avg_latency': sum(latencies) / len(latencies),
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'p50_latency': sorted(latencies)[len(latencies) // 2],
                    'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)],
                    'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)],
                    'avg_throughput': sum(throughputs) / len(throughputs),
                    'total_tokens': sum(m.tokens_generated for m in successful_metrics)
                })
                
            return summary
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        stats = self.get_current_stats()
        
        # Determine health status
        health_issues = []
        
        if stats['memory_usage_percent'] > 90:
            health_issues.append('High memory usage')
            
        if stats['gpu_memory_usage'] > 90:
            health_issues.append('High GPU memory usage')
            
        if stats['system_load'] > 80:
            health_issues.append('High CPU usage')
            
        if stats['total_requests'] > 0:
            error_rate = stats['failed_requests'] / stats['total_requests']
            if error_rate > 0.1:
                health_issues.append('High error rate')
                
        if stats['avg_latency'] > 5.0:
            health_issues.append('High latency')
            
        # Overall health status
        if not health_issues:
            status = 'healthy'
        elif len(health_issues) <= 2:
            status = 'warning'
        else:
            status = 'critical'
            
        return {
            'status': status,
            'issues': health_issues,
            'active_alerts': list(self.active_alerts),
            'uptime_seconds': time.time() - (self.system_metrics[0].timestamp if self.system_metrics else time.time()),
            'metrics_collected': len(self.inference_metrics),
            'last_update': time.time()
        }
        
    def register_alert_callback(self, callback):
        """Register a callback for alert notifications"""
        self.alert_callbacks.append(callback)
        
    def set_alert_threshold(self, alert_type: str, threshold: float):
        """Set custom alert threshold"""
        if alert_type in self.alert_thresholds:
            self.alert_thresholds[alert_type] = threshold
        else:
            raise ValueError(f"Unknown alert type: {alert_type}")
            
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        with self._lock:
            data = {
                'current_stats': self.current_stats,
                'inference_metrics': [asdict(m) for m in self.inference_metrics],
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'alert_thresholds': self.alert_thresholds,
                'export_timestamp': time.time()
            }
            
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def reset_metrics(self):
        """Reset all collected metrics and statistics"""
        with self._lock:
            self.inference_metrics.clear()
            self.system_metrics.clear()
            self.short_window.clear()
            self.medium_window.clear()
            self.long_window.clear()
            
            self.current_stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_latency': 0.0,
                'avg_tokens_per_second': 0.0,
                'total_tokens_generated': 0,
                'requests_per_second': 0.0,
                'system_load': 0.0,
                'memory_usage_percent': 0.0,
                'gpu_utilization': 0.0,
                'gpu_memory_usage': 0.0
            }
            
        self.logger.info("Performance metrics reset")
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform monitor health check"""
        return {
            'monitoring_active': self.monitoring_active,
            'metrics_collected': len(self.inference_metrics),
            'system_metrics_collected': len(self.system_metrics),
            'active_alerts': list(self.active_alerts),
            'gpu_available': GPU_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'last_metric_time': self.inference_metrics[-1].timestamp if self.inference_metrics else None
        }