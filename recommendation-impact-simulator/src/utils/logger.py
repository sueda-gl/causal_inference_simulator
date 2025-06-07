"""Logger configuration for the application."""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from ..config import get_settings


class LoggerConfig:
    """Configure application-wide logging using loguru."""
    
    def __init__(self):
        """Initialize logger configuration."""
        self.settings = get_settings()
        self._configured = False
        
    def configure(self, 
                  log_dir: Optional[Path] = None,
                  enable_file_logging: bool = True,
                  enable_json_logging: bool = False) -> None:
        """
        Configure loguru for the entire application.
        
        Args:
            log_dir: Directory for log files (defaults to logs/)
            enable_file_logging: Whether to log to files
            enable_json_logging: Whether to use JSON format for structured logging
        """
        if self._configured:
            return
            
        # Remove default handler
        logger.remove()
        
        # Console logging with custom format
        self._setup_console_logging()
        
        # File logging
        if enable_file_logging:
            self._setup_file_logging(log_dir)
            
        # JSON logging for production
        if enable_json_logging:
            self._setup_json_logging(log_dir)
            
        # Set up exception handling
        self._setup_exception_handling()
        
        self._configured = True
        logger.info(f"Logger configured for {self.settings.app_name} v{self.settings.app_version}")
        
    def _setup_console_logging(self) -> None:
        """Set up console logging with colored output."""
        # Development-friendly format for console
        if self.settings.debug:
            console_format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            console_format = self.settings.log_format
            
        logger.add(
            sys.stderr,
            format=console_format,
            level=self.settings.log_level,
            colorize=True,
            enqueue=True,  # Thread-safe logging
        )
        
    def _setup_file_logging(self, log_dir: Optional[Path] = None) -> None:
        """Set up file logging with rotation."""
        if log_dir is None:
            log_dir = Path("logs")
            
        log_dir.mkdir(exist_ok=True)
        
        # General application log
        logger.add(
            log_dir / "app_{time:YYYY-MM-DD}.log",
            format=self.settings.log_format,
            level=self.settings.log_level,
            rotation="1 day",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )
        
        # Error log
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format=self.settings.log_format,
            level="ERROR",
            rotation="1 day",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
        
        # Debug log (only in debug mode)
        if self.settings.debug:
            logger.add(
                log_dir / "debug_{time:YYYY-MM-DD}.log",
                format=self.settings.log_format,
                level="DEBUG",
                rotation="100 MB",
                retention="7 days",
                compression="zip",
                enqueue=True,
            )
            
    def _setup_json_logging(self, log_dir: Optional[Path] = None) -> None:
        """Set up JSON logging for structured logs."""
        if log_dir is None:
            log_dir = Path("logs")
            
        log_dir.mkdir(exist_ok=True)
        
        def json_formatter(record: Dict[str, Any]) -> str:
            """Format log record as JSON."""
            import json
            
            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "app_name": self.settings.app_name,
                "app_version": self.settings.app_version,
            }
            
            # Add extra fields if present
            if record.get("extra"):
                log_entry.update(record["extra"])
                
            # Add exception info if present
            if record.get("exception"):
                log_entry["exception"] = {
                    "type": record["exception"].type.__name__,
                    "value": str(record["exception"].value),
                    "traceback": record["exception"].traceback,
                }
                
            return json.dumps(log_entry)
        
        logger.add(
            log_dir / "app_{time:YYYY-MM-DD}.json",
            format=json_formatter,
            level=self.settings.log_level,
            rotation="1 day",
            retention="30 days",
            compression="zip",
            serialize=True,
            enqueue=True,
        )
        
    def _setup_exception_handling(self) -> None:
        """Set up global exception handling."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions."""
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
                
            logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
        sys.excepthook = handle_exception
        

# Global logger configuration instance
_logger_config = LoggerConfig()


def setup_logging(**kwargs) -> None:
    """
    Set up application logging.
    
    This should be called once at application startup.
    
    Args:
        **kwargs: Arguments passed to LoggerConfig.configure()
    """
    _logger_config.configure(**kwargs)


def get_logger(name: Optional[str] = None) -> "logger":
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Convenience functions for structured logging
def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_ms": duration * 1000,
            "performance_metric": True,
            **kwargs
        }
    )


def log_business_metric(metric_name: str, value: float, **kwargs) -> None:
    """Log business metrics."""
    logger.info(
        f"Business metric: {metric_name}={value}",
        extra={
            "metric_name": metric_name,
            "metric_value": value,
            "business_metric": True,
            **kwargs
        }
    )


def log_causal_estimate(treatment: str, effect: float, ci_lower: float, ci_upper: float, **kwargs) -> None:
    """Log causal effect estimates."""
    logger.info(
        f"Causal estimate: {treatment} effect={effect:.4f} CI=[{ci_lower:.4f}, {ci_upper:.4f}]",
        extra={
            "treatment": treatment,
            "effect": effect,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "causal_estimate": True,
            **kwargs
        }
    )


# Context managers for structured logging
class log_context:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs):
        """Initialize with context variables."""
        self.context = kwargs
        self.token = None
        
    def __enter__(self):
        """Enter context."""
        self.token = logger.contextualize(**self.context)
        self.token.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.token:
            self.token.__exit__(exc_type, exc_val, exc_tb)


class log_duration:
    """Context manager for logging operation duration."""
    
    def __init__(self, operation: str, **kwargs):
        """Initialize with operation name."""
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log duration."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            log_performance(self.operation, duration, **self.kwargs)
        else:
            logger.error(
                f"Operation {self.operation} failed after {duration:.2f}s",
                extra={
                    "operation": self.operation,
                    "duration_ms": duration * 1000,
                    "failed": True,
                    **self.kwargs
                }
            )
