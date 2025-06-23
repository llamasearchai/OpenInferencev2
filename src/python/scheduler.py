"""
Advanced Request Scheduler
Optimizes batching and resource utilization
"""
import asyncio
import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
from .openinferencev2 import InferenceRequest, InferenceResponse
@dataclass
class ScheduledRequest:
    """Request with scheduling metadata"""
    request: InferenceRequest
    priority: float
    arrival_time: float
    estimated_tokens: int
    
    def __lt__(self, other):
        return self.priority < other.priority
class RequestScheduler:
    """Intelligent request scheduler for optimal batching"""
    
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
            'batching_efficiency': 0.0
        }
        
        # Configuration
        self.batch_timeout = 0.1  # Maximum time to wait for batch formation
        self.priority_weights = {
            'interactive': 1.0,
            'batch': 0.5,
            'background': 0.1
        }
        
    async def schedule_request(self, request: InferenceRequest, priority: str = 'interactive') -> InferenceResponse:
        """Schedule a single request"""
        if len(self.priority_queue) >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
            
        # Estimate token requirements
        estimated_tokens = self._estimate_tokens(request)
        
        # Create scheduled request
        scheduled_req = ScheduledRequest(
            request=request,
            priority=self.priority_weights.get(priority, 1.0),
            arrival_time=time.time(),
            estimated_tokens=estimated_tokens
        )
        
        # Add to priority queue
        heapq.heappush(self.priority_queue, scheduled_req)
        self.stats['total_scheduled'] += 1
        
        # Process the request
        return await self._process_scheduled_request(scheduled_req)
        
    async def process_batch(self, requests: List[InferenceRequest], batch_size: Optional[int] = None) -> List[InferenceResponse]:
        """Process a batch of requests"""
        if batch_size is None:
            batch_size = min(len(requests), self.max_batch_size)
            
        # Split into appropriately sized batches
        batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
        
        all_responses = []
        for batch in batches:
            batch_responses = await self._execute_batch(batch)
            all_responses.extend(batch_responses)
            
        return all_responses
        
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
        
    async def _execute_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Execute a batch of requests"""
        self.logger.debug(f"Executing batch of {len(requests)} requests")
        
        start_time = time.time()
        
        # Sort requests by estimated complexity for optimal GPU utilization
        sorted_requests = sorted(requests, key=self._get_request_complexity)
        
        # Execute batch through engine
        if hasattr(self.engine, 'generate_batch'):
            # Use native batch processing if available
            responses = await self.engine.generate_batch(sorted_requests)
        else:
            # Fallback to parallel individual processing
            tasks = [self.engine.generate(req) for req in sorted_requests]
            responses = await asyncio.gather(*tasks)
            
        execution_time = time.time() - start_time
        self.logger.debug(f"Batch execution completed in {execution_time:.3f}s")
        
        self.stats['total_processed'] += len(requests)
        
        return responses
        
    def _get_request_complexity(self, request: InferenceRequest) -> float:
        """Calculate request complexity for sorting"""
        # Consider prompt length, max_tokens, and sampling parameters
        complexity = len(request.prompt) * 0.1
        complexity += request.max_tokens * 0.5
        complexity += request.temperature * 10  # Higher temperature = more computation
        return complexity
        
    def _update_stats(self, queue_time: float, batch_size: int):
        """Update scheduler statistics"""
        # Update average queue time
        total_requests = self.stats['total_processed']
        if total_requests > 0:
            self.stats['avg_queue_time'] = (
                (self.stats['avg_queue_time'] * (total_requests - 1) + queue_time) / 
                total_requests
            )
        else:
            self.stats['avg_queue_time'] = queue_time
            
        # Update average batch size
        total_batches = self.stats.get('total_batches', 0) + 1
        current_avg_batch_size = self.stats.get('avg_batch_size', 0)
        self.stats['avg_batch_size'] = (
            (current_avg_batch_size * (total_batches - 1) + batch_size) / 
            total_batches
        )
        self.stats['total_batches'] = total_batches
        
        # Calculate batching efficiency
        theoretical_max = self.max_batch_size * total_batches
        actual_throughput = self.stats['total_processed']
        self.stats['batching_efficiency'] = actual_throughput / theoretical_max if theoretical_max > 0 else 0
        
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            'queue_length': len(self.priority_queue),
            'current_batch_size': len(self.current_batch),
            'avg_queue_time': self.stats['avg_queue_time'],
            'avg_batch_size': self.stats['avg_batch_size'],
            'batching_efficiency': self.stats['batching_efficiency'],
            'total_scheduled': self.stats['total_scheduled'],
            'total_processed': self.stats['total_processed']
        }
        
    async def start_scheduler(self):
        """Start the background scheduler task"""
        self.running = True
        asyncio.create_task(self._scheduler_loop())
        
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check for batch formation opportunities
                if self.priority_queue and len(self.current_batch) == 0:
                    # Start forming a new batch
                    anchor_request = heapq.heappop(self.priority_queue)
                    self.current_batch = await self._form_optimal_batch(anchor_request)
                    
                    # Queue batch for execution
                    if self.current_batch:
                        self.batch_queue.append(self.current_batch)
                        self.current_batch = []
                        
                # Process ready batches
                if self.batch_queue:
                    batch = self.batch_queue.popleft()
                    asyncio.create_task(self._execute_batch([req.request for req in batch]))
                    
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(0.1)
                
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False