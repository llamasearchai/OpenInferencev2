#!/usr/bin/env python3
"""
Comprehensive test suite for OpenInferencev2
"""
import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openinferencev2.config import Config
from openinferencev2.openinferencev2 import OpenInferencev2Engine, InferenceRequest, InferenceResponse
from openinferencev2.scheduler import RequestScheduler
from openinferencev2.monitor import PerformanceMonitor
from openinferencev2.optimization import ModelOptimizer

class TestOpenInferencev2Engine:
    """Test suite for OpenInferencev2Engine"""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create a test engine instance"""
        config = Config({
            'num_gpus': 1,
            'max_batch_size': 4,
            'use_fp16': False,  # Easier for testing
            'use_flash_attention': False,
            'use_cuda_graphs': False
        })
        
        engine = OpenInferencev2Engine("/tmp/test_model", config.__dict__)
        
        # Mock tokenizer for testing
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.decode.return_value = "Test response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        engine.tokenizer = mock_tokenizer
        
        yield engine
        
        # Cleanup
        try:
            await engine.shutdown()
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_single_inference(self, engine):
        """Test single inference request"""
        request = InferenceRequest(
            id="test_001",
            prompt="Hello, world!",
            max_tokens=50,
            temperature=0.7
        )
        
        # Mock the PyTorch generation
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_generate.return_value = InferenceResponse(
                id="test_001",
                text="Hello! How can I help you today?",
                tokens=[1, 2, 3, 4, 5],
                latency=0.0,  # Will be calculated by engine
                tokens_per_second=0.0,  # Will be calculated by engine
                finish_reason="length"
            )
            
            response = await engine.generate(request)
            
            assert response.id == "test_001"
            assert response.success
            assert len(response.tokens) == 5
            assert response.latency >= 0  # Should be calculated
            assert response.tokens_per_second >= 0  # Should be calculated
    
    @pytest.mark.asyncio
    async def test_batch_inference(self, engine):
        """Test batch inference processing"""
        requests = [
            InferenceRequest(
                id=f"batch_{i}",
                prompt=f"Test prompt {i}",
                max_tokens=25
            )
            for i in range(3)
        ]
        
        # Mock batch processing
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_responses = [
                InferenceResponse(
                    id=req.id,
                    text=f"Response for {req.id}",
                    tokens=list(range(25)),
                    latency=0.0,  # Will be calculated by engine
                    tokens_per_second=0.0,  # Will be calculated by engine
                    finish_reason="length"
                )
                for req in requests
            ]
            mock_generate.side_effect = mock_responses
            
            scheduler = RequestScheduler(engine, max_batch_size=4)
            responses = await scheduler.process_batch(requests)
            
            assert len(responses) == 3
            for response in responses:
                assert response.success
                assert len(response.tokens) == 25
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, engine):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor(history_size=10)
        engine.monitor = monitor
        
        await monitor.start_monitoring()
        
        request = InferenceRequest(
            id="perf_test",
            prompt="Performance test",
            max_tokens=10
        )
        
        # Mock generation
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_generate.return_value = InferenceResponse(
                id="perf_test",
                text="Performance response",
                tokens=list(range(10)),
                latency=0.0,  # Will be calculated by engine
                tokens_per_second=0.0,  # Will be calculated by engine
                finish_reason="length"
            )
            
            response = await engine.generate(request)
            
            # Check that metrics were recorded
            stats = monitor.get_current_stats()
            assert stats['total_requests'] >= 1
            
        await monitor.stop_monitoring()
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = Config({
            'num_gpus': 2,
            'max_batch_size': 16
        })
        assert valid_config.is_valid()
        
        # Invalid configuration
        invalid_config = Config({
            'num_gpus': 0,  # Should be at least 1
            'max_batch_size': -1  # Should be positive
        })
        assert not invalid_config.is_valid()
        errors = invalid_config.validate()
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in inference"""
        request = InferenceRequest(
            id="error_test",
            prompt="Error test",
            max_tokens=10
        )
        
        # Mock an error during generation
        with patch.object(engine, '_generate_pytorch') as mock_generate:
            mock_generate.side_effect = Exception("Test error")
            
            response = await engine.generate(request)
            
            assert not response.success
            assert "error" in response.error_message.lower()
            assert response.id == "error_test"
    
    @pytest.mark.asyncio
    async def test_streaming_inference(self, engine):
        """Test streaming inference capability"""
        request = InferenceRequest(
            id="stream_test",
            prompt="Stream test",
            max_tokens=20
        )
        
        # Mock the base generate method
        with patch.object(engine, 'generate') as mock_generate:
            mock_generate.return_value = InferenceResponse(
                id="stream_test",
                text="This is a streaming response test",
                tokens=list(range(20)),
                latency=0.1,
                tokens_per_second=200.0,
                finish_reason="length"
            )
            
            chunks = []
            async for chunk in engine.generate_stream(request):
                chunks.append(chunk)
                
            assert len(chunks) > 0
            full_text = "".join(chunks)
            assert "streaming" in full_text


class TestRequestScheduler:
    """Test suite for RequestScheduler"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine"""
        engine = Mock()
        engine.generate = AsyncMock()
        return engine
    
    @pytest.fixture
    def scheduler(self, mock_engine):
        """Create a scheduler instance"""
        return RequestScheduler(mock_engine, max_batch_size=4, max_queue_size=100)
    
    @pytest.mark.asyncio
    async def test_request_processing(self, scheduler, mock_engine):
        """Test request processing with priority"""
        high_priority_req = InferenceRequest(
            id="high_priority",
            prompt="High priority request",
            max_tokens=10
        )
        
        # Mock engine response
        mock_engine.generate.return_value = InferenceResponse(
            id="high_priority",
            text="High priority response",
            tokens=list(range(10)),
            latency=0.1,
            tokens_per_second=100.0,
            finish_reason="length"
        )
        
        response = await scheduler.process_request(high_priority_req)
        
        assert response.id == "high_priority"
        assert response.success
    
    @pytest.mark.asyncio
    async def test_batch_formation(self, scheduler, mock_engine):
        """Test batch formation and processing"""
        requests = [
            InferenceRequest(
                id=f"batch_req_{i}",
                prompt=f"Batch request {i}",
                max_tokens=10
            )
            for i in range(5)
        ]
        
        # Mock batch processing
        def mock_batch_generate(req):
            return InferenceResponse(
                id=req.id,
                text=f"Response for {req.id}",
                tokens=list(range(10)),
                latency=0.1,
                tokens_per_second=100.0,
                finish_reason="length"
            )
        
        mock_engine.generate.side_effect = mock_batch_generate
        
        responses = await scheduler.process_batch(requests)
        
        assert len(responses) == 5
        for response in responses:
            assert response.success
    
    def test_scheduler_stats(self, scheduler):
        """Test scheduler statistics"""
        stats = scheduler.get_stats()
        
        assert 'total_scheduled' in stats
        assert 'total_processed' in stats
        assert 'avg_queue_time' in stats
        assert 'queue_length' in stats


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create a monitor instance"""
        return PerformanceMonitor(history_size=100, collection_interval=0.1)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.history_size == 100
        assert monitor.collection_interval == 0.1
        assert not monitor.monitoring_active
    
    def test_metrics_collection(self, monitor):
        """Test metrics collection"""
        stats = monitor.get_current_stats()
        
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'successful_requests' in stats
        assert 'failed_requests' in stats
    
    @pytest.mark.asyncio
    async def test_inference_metrics_recording(self, monitor):
        """Test recording inference metrics"""
        response = InferenceResponse(
            id="metrics_test",
            text="Metrics test response",
            tokens=list(range(50)),
            latency=0.1,
            tokens_per_second=500.0,
            finish_reason="length"
        )
        
        await monitor.record_inference(response)
        
        stats = monitor.get_current_stats()
        assert stats['total_requests'] >= 1
    
    def test_threshold_alerts(self, monitor):
        """Test alert threshold system"""
        # Register an alert callback
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
            
        monitor.register_alert_callback(alert_callback)
        
        # Set a low threshold for testing
        monitor.set_alert_threshold('high_latency', 0.01)
        
        # This should be a simple test without triggering actual alerts
        assert len(monitor.alert_thresholds) > 0


class TestBenchmarkSuite:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def engine(self):
        """Create engine for benchmarking"""
        config = Config({'num_gpus': 1, 'max_batch_size': 8})
        engine = OpenInferencev2Engine("/tmp/benchmark_model", config.__dict__)
        
        # Mock tokenizer for testing
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.decode.return_value = "Test response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        engine.tokenizer = mock_tokenizer
        
        return engine
    
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
                    latency=0.0,
                    tokens_per_second=0.0,
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
                    latency=0.0,  # Will be calculated by engine
                    tokens_per_second=0.0,  # Will be calculated by engine
                    finish_reason="length"
                )
                for req in requests
            ]
            mock_generate.side_effect = mock_responses
            
            scheduler = RequestScheduler(engine, max_batch_size=8)
            responses = await scheduler.process_batch(requests)
        
        total_time = time.time() - start_time
        total_time = max(total_time, 0.001)  # Ensure minimum time to avoid division by zero
        throughput = len(requests) / total_time
        total_tokens = sum(len(resp.tokens) for resp in responses)
        tokens_per_second = total_tokens / total_time
        
        print(f"Request throughput: {throughput:.1f} req/s")
        print(f"Token throughput: {tokens_per_second:.1f} tokens/s")
        
        # Performance assertions
        assert throughput > 0.0  # Should have some throughput
        assert tokens_per_second > 0.0  # Should have some token throughput

if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])