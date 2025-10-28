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
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file or LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Optional JSON structured logging
    if enable_json:
        json_handler = logging.handlers.RotatingFileHandler(
            JSON_LOG_FILE,
            maxBytes=max_bytes,
            backupCount=backup_count
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