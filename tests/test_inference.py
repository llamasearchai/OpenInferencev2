"""
Comprehensive test suite for OpenInferencev2
Production-ready testing framework
"""
import asyncio
import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
from openinferencev2 import OpenInferencev2Engine, InferenceRequest, InferenceResponse
from openinferencev2.scheduler import RequestScheduler
from openinferencev2.monitor import PerformanceMonitor
from openinferencev2.config import Config

class TestOpenInferencev2Engine:
    """Test suite for the main inference engine"""
    
    @pytest.fixture
    async def engine(self):
        """Create test engine instance"""
        config = {
            'num_gpus': 1,
            'max_batch_size': 4,
            'max_sequence_length': 2048,
            'use_fp16': True,
            'use_flash_attention': True,
            'kv_cache_size_gb': 2
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model directory
            model_path = Path(temp_dir) / 'test_model'
            model_path.mkdir()
            
            # Create minimal config
            config_dict = {
                'vocab_size': 32000,
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'intermediate_size': 11008
            }
            
            with open(model_path / 'config.json', 'w') as f:
                json.dump(config_dict, f)
                
            engine = OpenInferencev2Engine(str(model_path), config)
            
            # Mock the model loading for testing
            with patch.object(engine, '_load_pytorch_model'):
                await engine.load_model()
                
            yield engine
            
            await engine.shutdown()
            
    @pytest.mark.asyncio
    async def test_single_inference(self, engine):
        """Test single inference request"""
        request = InferenceRequest(
            id="test_001",
            prompt="What is the capital of France?",
            max_tokens=50,
            temperature=0.7
        )
        
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_response = InferenceResponse(
                id="test_001",
                text="The capital of France is Paris.",
                tokens=[1, 2, 3, 4, 5],
                latency=0.1,
                tokens_per_second=50.0,
                finish_reason="length"
            )
            mock_generate.return_value = mock_response
            
            response = await engine.generate(request)
            
            assert response.id == "test_001"
            assert response.success == True
            assert len(response.text) > 0
            assert response.latency > 0
            assert response.tokens_per_second > 0
            
    @pytest.mark.asyncio
    async def test_batch_inference(self, engine):
        """Test batch inference processing"""
        requests = [
            InferenceRequest(id=f"batch_{i}", prompt=f"Test prompt {i}", max_tokens=20)
            for i in range(4)
        ]
        
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_responses = [
                InferenceResponse(
                    id=req.id,
                    text=f"Response for {req.prompt}",
                    tokens=[1, 2, 3],
                    latency=0.1,
                    tokens_per_second=30.0,
                    finish_reason="length"
                )
                for req in requests
            ]
            mock_generate.side_effect = mock_responses
            
            # Test through scheduler
            scheduler = RequestScheduler(engine, max_batch_size=4)
            responses = await scheduler.process_batch(requests)
            
            assert len(responses) == 4
            for i, response in enumerate(responses):
                assert response.id == f"batch_{i}"
                assert response.success == True
                
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, engine):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Generate some test requests
            request = InferenceRequest(
                id="perf_test",
                prompt="Performance test prompt",
                max_tokens=10
            )
            
            with patch.object(engine, '_generate_pytorch') as mock_generate:
                mock_generate.return_value = InferenceResponse(
                    id="perf_test",
                    text="Test response",
                    tokens=[1, 2, 3],
                    latency=0.05,
                    tokens_per_second=60.0,
                    finish_reason="length"
                )
                
                await engine.generate(request)
                
            # Check that metrics were collected
            stats = engine.get_performance_stats()
            assert stats['total_requests'] > 0
            assert stats['avg_latency'] > 0
            
        finally:
            monitor.stop_monitoring()
            
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = {
            'num_gpus': 1,
            'max_batch_size': 8,
            'tensor_parallel_size': 1,
            'pipeline_parallel_size': 1
        }
        
        config = Config(valid_config)
        assert config.num_gpus == 1
        assert config.max_batch_size == 8
        
        # Invalid configuration
        with pytest.raises(ValueError):
            invalid_config = {
                'num_gpus': 0,  # Invalid
                'max_batch_size': -1  # Invalid
            }
            Config(invalid_config)
            
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling and recovery"""
        request = InferenceRequest(
            id="error_test",
            prompt="Test error handling",
            max_tokens=10
        )
        
        # Simulate inference error
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_generate.side_effect = RuntimeError("Simulated error")
            
            response = await engine.generate(request)
            
            assert response.success == False
            assert "error" in response.error_message.lower()
            assert response.id == "error_test"
            
    @pytest.mark.asyncio
    async def test_streaming_inference(self, engine):
        """Test streaming inference capabilities"""
        request = InferenceRequest(
            id="stream_test",
            prompt="Test streaming",
            max_tokens=20
        )
        
        with patch.object(engine, 'generate') as mock_generate:
            mock_generate.return_value = InferenceResponse(
                id="stream_test",
                text="This is a streaming test response",
                tokens=[1, 2, 3, 4, 5, 6, 7],
                latency=0.1,
                tokens_per_second=70.0,
                finish_reason="length"
            )
            
            chunks = []
            async for chunk in engine.generate_stream(request):
                chunks.append(chunk)
                
            full_text = "".join(chunks)
            assert len(full_text) > 0
            assert len(chunks) > 1  # Should be multiple chunks

class TestRequestScheduler:
    """Test suite for request scheduling"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock engine for testing"""
        engine = Mock()
        engine.generate = Mock()
        return engine
        
    @pytest.fixture
    def scheduler(self, mock_engine):
        """Create scheduler instance"""
        return RequestScheduler(mock_engine, max_batch_size=4)
        
    @pytest.mark.asyncio
    async def test_request_prioritization(self, scheduler, mock_engine):
        """Test request prioritization logic"""
        # Set up mock responses
        mock_engine.generate.return_value = InferenceResponse(
            id="test",
            text="response",
            tokens=[1, 2, 3],
            latency=0.1,
            tokens_per_second=30.0,
            finish_reason="length"
        )
        
        # Create requests with different priorities
        high_priority_req = InferenceRequest(id="high", prompt="urgent", max_tokens=10)
        low_priority_req = InferenceRequest(id="low", prompt="background", max_tokens=10)
        
        # Schedule with priorities
        high_response = await scheduler.schedule_request(high_priority_req, priority='interactive')
        low_response = await scheduler.schedule_request(low_priority_req, priority='background')
        
        assert high_response.id == "high"
        assert low_response.id == "low"
        
    @pytest.mark.asyncio
    async def test_batch_formation(self, scheduler, mock_engine):
        """Test optimal batch formation"""
        requests = [
            InferenceRequest(id=f"batch_{i}", prompt=f"test {i}", max_tokens=10)
            for i in range(6)  # More than max_batch_size
        ]
        
        # Mock batch processing
        def mock_batch_generate(reqs):
            return [
                InferenceResponse(
                    id=req.id,
                    text=f"response for {req.id}",
                    tokens=[1, 2, 3],
                    latency=0.1,
                    tokens_per_second=30.0,
                    finish_reason="length"
                )
                for req in reqs
            ]
            
        with patch.object(scheduler, '_execute_batch', side_effect=mock_batch_generate):
            responses = await scheduler.process_batch(requests, batch_size=4)
            
        assert len(responses) == 6
        for response in responses:
            assert response.success == True
            
    def test_queue_status(self, scheduler):
        """Test queue status reporting"""
        status = scheduler.get_queue_status()
        
        assert 'queue_length' in status
        assert 'avg_batch_size' in status
        assert 'batching_efficiency' in status
        assert isinstance(status['total_scheduled'], int)

class TestPerformanceMonitor:
    """Test suite for performance monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return PerformanceMonitor(history_size=100, sample_interval=0.1)
        
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.history_size == 100
        assert monitor.sample_interval == 0.1
        assert not monitor.monitoring_active
        
    def test_metrics_collection(self, monitor):
        """Test metrics collection"""
        monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect some metrics
        monitor.stop_monitoring()
        
        metrics = monitor.get_current_metrics()
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'cpu_usage' in metrics['system']
        assert 'memory_usage' in metrics['system']
        
    def test_inference_metrics_recording(self, monitor):
        """Test inference metrics recording"""
        from openinferencev2.monitor import InferenceMetrics
        
        metrics = InferenceMetrics(
            request_id="test_001",
            latency=0.1,
            tokens_generated=50,
            tokens_per_second=500.0,
            batch_size=1,
            gpu_memory_peak=2.5
        )
        
        monitor.record_inference_metrics(metrics)
        
        stats = monitor.get_inference_statistics()
        assert stats['total_requests'] == 1
        assert stats['avg_latency'] == 0.1
        assert stats['avg_throughput'] == 500.0
        
    def test_threshold_alerts(self, monitor):
        """Test threshold-based alerting"""
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
            
        monitor.add_callback('threshold_alert', alert_callback)
        
        # Simulate high CPU usage
        with patch('psutil.cpu_percent', return_value=95.0):
            system_metrics = monitor._collect_system_metrics()
            monitor._check_thresholds(None, system_metrics)
            
        assert len(alerts_received) > 0
        assert 'CPU usage high' in alerts_received[0]

class TestBenchmarkSuite:
    """Benchmark testing suite"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_latency_benchmark(self, engine):
        """Benchmark inference latency"""
        request = InferenceRequest(
            id="latency_bench",
            prompt="The quick brown fox jumps over the lazy dog",
            max_tokens=100,
            temperature=0.0  # Deterministic
        )
        
        latencies = []
        
        for i in range(10):
            start_time = time.time()
            
            with patch.object(engine, '_generate_pytorch') as mock_generate:
                mock_generate.return_value = InferenceResponse(
                    id="latency_bench",
                    text="Benchmark response text",
                    tokens=list(range(100)),
                    latency=0.1,
                    tokens_per_second=1000.0,
                    finish_reason="length"
                )
                
                await engine.generate(request)
                
            latency = time.time() - start_time
            latencies.append(latency)
            
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")
        
        # Performance assertions
        assert avg_latency < 1.0  # Should be under 1 second
        assert p95_latency < 2.0  # P95 should be under 2 seconds
        
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, engine):
        """Benchmark inference throughput"""
        requests = [
            InferenceRequest(
                id=f"throughput_{i}",
                prompt=f"Throughput test prompt number {i}",
                max_tokens=50
            )
            for i in range(20)
        ]
        
        start_time = time.time()
        
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_responses = [
                InferenceResponse(
                    id=req.id,
                    text="Throughput test response",
                    tokens=list(range(50)),
                    latency=0.1,
                    tokens_per_second=500.0,
                    finish_reason="length"
                )
                for req in requests
            ]
            mock_generate.side_effect = mock_responses
            
            scheduler = RequestScheduler(engine, max_batch_size=8)
            responses = await scheduler.process_batch(requests)
            
        total_time = time.time() - start_time
        throughput = len(requests) / total_time
        total_tokens = sum(len(resp.tokens) for resp in responses)
        tokens_per_second = total_tokens / total_time
        
        print(f"Request throughput: {throughput:.1f} req/s")
        print(f"Token throughput: {tokens_per_second:.1f} tokens/s")
        
        # Performance assertions
        assert throughput > 10.0  # At least 10 requests per second
        assert tokens_per_second > 500.0  # At least 500 tokens per second

if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])