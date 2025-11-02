"""
logger.py

Configurable logging setup for GridironMetrics.
Features:
- Log rotation with size and time-based triggers
- Color-coded console output
- Configurable log levels per module
- JSON logging support for structured output
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from queue import Queue
import atexit

# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[41m'  # Red background
}
RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname for console output"""
    
    def format(self, record):
        # Add colors if not windows or if running in a modern terminal
        if not sys.platform.startswith('win') or os.getenv('WT_SESSION'):
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{RESET}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
            
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

# Default paths
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
JSON_LOG_FILE = os.path.join(LOG_DIR, 'app.json.log')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Global queue and listener for thread-safe logging (singleton pattern)
_log_queue = None
_queue_listener = None

def _get_or_create_queue_listener(log_file: str, max_bytes: int, backup_count: int):
    """Get or create the singleton queue listener for thread-safe logging."""
    global _log_queue, _queue_listener
    
    if _log_queue is None:
        _log_queue = Queue(-1)
        
        # File handler with rotation (UTF-8 encoding for Unicode support)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Start queue listener
        _queue_listener = logging.handlers.QueueListener(_log_queue, file_handler, respect_handler_level=True)
        _queue_listener.start()
        
        # Ensure listener stops on exit
        atexit.register(_queue_listener.stop)
    
    return _log_queue

def get_logger(
    name: str,
    level: str = 'INFO',
    enable_json: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    extra: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to enable JSON structured logging
        log_file: Custom log file path (default: LOG_DIR/app.log)
        max_bytes: Max size of each log file before rotation
        backup_count: Number of backup files to keep
        extra: Additional fields to include in JSON logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers if any
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Use global queue for thread-safe file logging
    log_queue = _get_or_create_queue_listener(log_file or LOG_FILE, max_bytes, backup_count)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    # Optional JSON structured logging (not implemented with queue for now)
    if enable_json:
        json_handler = logging.handlers.RotatingFileHandler(
            JSON_LOG_FILE,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)
    
    # Add extra context if provided
    if extra:
        logger = logging.LoggerAdapter(logger, extra)
    
    return logger

# Default logger instance
default_logger = get_logger('gridironmetrics')