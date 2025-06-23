"""
Request scheduler for OpenInferencev2
Implements intelligent batching and priority-based request scheduling
"""
import asyncio
import heapq
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from .openinferencev2 import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

@dataclass
class ScheduledRequest:
    """Request with scheduling metadata"""
    priority: float
    arrival_time: float
    request: InferenceRequest
    estimated_tokens: int
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority

class RequestScheduler:
    """Intelligent request scheduler for optimal batching and throughput"""
    
    def __init__(self, engine, max_batch_size: int = 32, max_queue_size: int = 1000):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Scheduling queues
        self.priority_queue = []  # Min-heap for priority scheduling
        self.batch_queue = deque()  # Ready-to-process batches
        
        # Scheduling state
        self.current_batch = []
        self.batch_formation_deadline = None
        self.running = False
        
        # Performance tracking
        self.stats = {
            'total_scheduled': 0,
            'total_processed': 0,
            'avg_queue_time': 0.0,
            'avg_batch_size': 0.0,
            'batching_efficiency': 0.0,
            'queue_length': 0,
            'processing_rate': 0.0
        }
        
        # Configuration
        self.batch_timeout = 0.1  # Maximum time to wait for batch formation
        self.priority_weights = {
            'interactive': 1.0,
            'batch': 0.5,
            'background': 0.1
        }
        
        # Async processing
        self.processing_task = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def start(self):
        """Start the scheduler background processing"""
        if self.running:
            return
            
        self.running = True
        self.processing_task = asyncio.create_task(self._background_processor())
        self.logger.info("Request scheduler started")
        
    async def stop(self):
        """Stop the scheduler and wait for completion"""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
        self.executor.shutdown(wait=True)
        self.logger.info("Request scheduler stopped")
        
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single request with optimal scheduling"""
        if len(self.priority_queue) >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
            
        # Create scheduled request
        scheduled_req = ScheduledRequest(
            priority=self._calculate_priority(request),
            arrival_time=time.time(),
            request=request,
            estimated_tokens=self._estimate_tokens(request)
        )
        
        # Add to priority queue
        heapq.heappush(self.priority_queue, scheduled_req)
        self.stats['total_scheduled'] += 1
        self.stats['queue_length'] = len(self.priority_queue)
        
        # Process the request
        return await self._process_scheduled_request(scheduled_req)
        
    async def process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Process a batch of requests"""
        if not requests:
            return []
            
        try:
            self.logger.debug(f"Processing batch of {len(requests)} requests")
            
            # Execute batch inference
            responses = []
            for request in requests:
                try:
                    response = await self.engine.generate(request)
                    responses.append(response)
                except Exception as e:
                    self.logger.error(f"Failed to process request {request.id}: {e}")
                    responses.append(InferenceResponse(
                        id=request.id,
                        text="",
                        tokens=[],
                        latency=0.0,
                        tokens_per_second=0.0,
                        finish_reason='error',
                        success=False,
                        error_message=str(e)
                    ))
                    
            # Update statistics
            self.stats['total_processed'] += len(responses)
            successful_responses = [r for r in responses if r.success]
            
            if successful_responses:
                avg_latency = sum(r.latency for r in successful_responses) / len(successful_responses)
                self._update_batch_stats(len(requests), avg_latency)
                
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Return error responses for all requests
            return [
                InferenceResponse(
                    id=req.id,
                    text="",
                    tokens=[],
                    latency=0.0,
                    tokens_per_second=0.0,
                    finish_reason='error',
                    success=False,
                    error_message=str(e)
                )
                for req in requests
            ]
            
    async def _process_scheduled_request(self, scheduled_req: ScheduledRequest) -> InferenceResponse:
        """Process a single scheduled request"""
        # Wait for optimal batching opportunity
        optimal_batch = await self._form_optimal_batch(scheduled_req)
        
        # Execute the batch
        batch_responses = await self._execute_batch([req.request for req in optimal_batch])
        
        # Find and return the response for our request
        for response in batch_responses:
            if response.id == scheduled_req.request.id:
                # Update stats
                queue_time = time.time() - scheduled_req.arrival_time
                self._update_stats(queue_time, len(optimal_batch))
                return response
                
        raise RuntimeError(f"Response not found for request {scheduled_req.request.id}")
        
    async def _form_optimal_batch(self, anchor_request: ScheduledRequest) -> List[ScheduledRequest]:
        """Form an optimal batch around an anchor request"""
        batch = [anchor_request]
        batch_tokens = anchor_request.estimated_tokens
        
        # Set deadline for batch formation
        deadline = time.time() + self.batch_timeout
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            # Look for compatible requests
            compatible_request = await self._find_compatible_request(batch_tokens, deadline)
            
            if compatible_request is None:
                break
                
            batch.append(compatible_request)
            batch_tokens += compatible_request.estimated_tokens
            
            # Remove from priority queue
            if compatible_request in self.priority_queue:
                self.priority_queue.remove(compatible_request)
                heapq.heapify(self.priority_queue)
            
        return batch
        
    async def _find_compatible_request(self, current_batch_tokens: int, deadline: float) -> Optional[ScheduledRequest]:
        """Find a request that's compatible with the current batch"""
        max_additional_tokens = self._calculate_max_additional_tokens(current_batch_tokens)
        
        # Check priority queue for compatible requests
        for req in self.priority_queue:
            if req.estimated_tokens <= max_additional_tokens:
                # Check if we have time to wait
                if time.time() < deadline - 0.01:  # Leave 10ms buffer
                    return req
                    
        return None
        
    def _calculate_max_additional_tokens(self, current_tokens: int) -> int:
        """Calculate maximum additional tokens for batch"""
        # Heuristic: aim for balanced batch utilization
        max_total_tokens = self.max_batch_size * 512  # Assume 512 avg tokens per request
        return max(0, max_total_tokens - current_tokens)
        
    def _estimate_tokens(self, request: InferenceRequest) -> int:
        """Estimate token requirements for a request"""
        # Simple heuristic based on prompt length and max_tokens
        prompt_tokens = len(request.prompt.split()) * 1.3  # Rough approximation
        return int(prompt_tokens + request.max_tokens)
        
    def _calculate_priority(self, request: InferenceRequest) -> float:
        """Calculate request priority (lower = higher priority)"""
        # Base priority on request type or other factors
        request_type = getattr(request, 'type', 'interactive')
        base_priority = self.priority_weights.get(request_type, 1.0)
        
        # Adjust based on estimated processing time
        estimated_time = self._estimate_tokens(request) / 1000.0  # Rough estimate
        
        # Higher priority for shorter requests
        priority = base_priority * (1.0 + estimated_time)
        
        return priority
        
    async def _execute_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Execute a batch of requests"""
        return await self.process_batch(requests)
        
    async def _background_processor(self):
        """Background task for processing queued requests"""
        while self.running:
            try:
                # Process any queued batches
                if self.batch_queue:
                    batch = self.batch_queue.popleft()
                    await self._execute_batch(batch)
                    
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")
                await asyncio.sleep(0.1)
                
    def _update_stats(self, queue_time: float, batch_size: int):
        """Update performance statistics"""
        # Update queue time
        total_requests = self.stats['total_processed']
        if total_requests > 0:
            self.stats['avg_queue_time'] = (
                (self.stats['avg_queue_time'] * (total_requests - 1) + queue_time) / total_requests
            )
        else:
            self.stats['avg_queue_time'] = queue_time
            
        # Update batch size
        self._update_batch_stats(batch_size, 0.0)
        
    def _update_batch_stats(self, batch_size: int, latency: float):
        """Update batch-related statistics"""
        current_batches = self.stats.get('total_batches', 0)
        
        # Update average batch size
        if current_batches > 0:
            self.stats['avg_batch_size'] = (
                (self.stats['avg_batch_size'] * current_batches + batch_size) / (current_batches + 1)
            )
        else:
            self.stats['avg_batch_size'] = batch_size
            
        self.stats['total_batches'] = current_batches + 1
        
        # Calculate batching efficiency
        theoretical_max = self.max_batch_size
        self.stats['batching_efficiency'] = self.stats['avg_batch_size'] / theoretical_max
        
        # Update processing rate
        if latency > 0:
            self.stats['processing_rate'] = batch_size / latency
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current scheduler statistics"""
        stats = self.stats.copy()
        stats['queue_length'] = len(self.priority_queue)
        stats['batch_queue_length'] = len(self.batch_queue)
        stats['is_running'] = self.running
        return stats
        
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_scheduled': 0,
            'total_processed': 0,
            'avg_queue_time': 0.0,
            'avg_batch_size': 0.0,
            'batching_efficiency': 0.0,
            'queue_length': 0,
            'processing_rate': 0.0,
            'total_batches': 0
        }
        
    def __repr__(self) -> str:
        """String representation of scheduler"""
        return (f"RequestScheduler(max_batch_size={self.max_batch_size}, "
                f"max_queue_size={self.max_queue_size}, "
                f"queue_length={len(self.priority_queue)}, "
                f"running={self.running})")
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform scheduler health check"""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'queue_length': len(self.priority_queue),
            'batch_queue_length': len(self.batch_queue),
            'max_queue_size': self.max_queue_size,
            'queue_utilization': len(self.priority_queue) / self.max_queue_size,
            'processing_rate': self.stats.get('processing_rate', 0.0),
            'batching_efficiency': self.stats.get('batching_efficiency', 0.0)
        }