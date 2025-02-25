import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class OperationMetrics:
    """Tracks metrics for a single operation."""
    operation_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    success: bool = False
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{self.operation_name} {status} ({self.duration:.2f}s)"


class PerformanceMonitor:
    """Utility for monitoring performance of long-running operations."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.operations: Dict[str, OperationMetrics] = {}
        self.start_time = time.time()
        logger.debug(f"Initialized {component_name} performance monitor")
    
    @contextmanager
    def measure(self, operation_name: str, **metadata):
        """Context manager to measure an operation's duration."""
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata
        )
        
        logger.debug(f"Starting operation: {operation_name}")
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.error = e
            metrics.success = False
            raise
        finally:
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            self.operations[operation_name] = metrics
            
            log_level = logging.INFO if metrics.success else logging.WARNING
            logger.log(log_level, f"{operation_name} completed in {metrics.duration:.2f}s")
    
    def track_operation(self, operation_name: str, **metadata):
        """Decorator to track function execution time."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.measure(operation_name, **metadata) as metrics:
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.measure(operation_name, **metadata) as metrics:
                    return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        total_time = time.time() - self.start_time
        successful_ops = sum(1 for op in self.operations.values() if op.success)
        failed_ops = len(self.operations) - successful_ops
        
        return {
            "component": self.component_name,
            "total_duration": total_time,
            "operations_count": len(self.operations),
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "operations": {
                name: {
                    "duration": op.duration,
                    "success": op.success,
                    "error": str(op.error) if op.error else None,
                    **op.metadata
                }
                for name, op in self.operations.items()
            }
        }
    
    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of all operations."""
        report = self.report()
        success_rate = (report["successful_operations"] / 
                       max(report["operations_count"], 1)) * 100
        
        logger.log(level, f"Performance Summary for {self.component_name}:")
        logger.log(level, f"  Total Duration: {report['total_duration']:.2f}s")
        logger.log(level, f"  Operations: {report['operations_count']} total, "
                          f"{report['successful_operations']} successful "
                          f"({success_rate:.1f}%)")
        
        # Log individual operations at debug level
        for name, op in self.operations.items():
            status = "SUCCESS" if op.success else "FAILURE"
            logger.debug(f"  {name}: {status} in {op.duration:.2f}s")


# Add missing import at the top to avoid error
import asyncio