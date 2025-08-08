import logging
from logging.handlers import QueueHandler, QueueListener
import colorlog
import warnings
import multiprocessing
import atexit
from langchain_core._api.deprecation import LangChainDeprecationWarning

log_queue = multiprocessing.Queue()
_listener = None  # Global listener instance

def _configure_library_loggers():
  """Configure third-party library loggers to only show errors."""
  library_loggers = [
    "ctranslate2",
    "transformers", 
    "kokoro",
    "langchain",
    "urllib3",
    "faster_whisper",
    "openai",
  ]
  
  for logger_name in library_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class ProjectLogFilter(logging.Filter):
  """Filter to only show debug/info from project loggers, but errors from all loggers."""
  
  def __init__(self, project_prefix):
    super().__init__()
    self.project_prefix = project_prefix
  
  def filter(self, record):
    # Always allow ERROR and CRITICAL from any logger
    if record.levelno >= logging.ERROR:
      return True
    
    # For DEBUG, INFO, WARNING - only allow from project loggers
    return record.name.startswith(self.project_prefix)


def setup_logging(level=logging.DEBUG):
  """Set up project logger with a multiprocessing-safe QueueHandler."""
  logger = logging.getLogger("speech_to_speech")
  logger.setLevel(level)
  logger.propagate = False  # Prevent double logging

  # Clear any existing handlers to avoid duplicates
  if logger.handlers:
    logger.handlers.clear()

  queue_handler = QueueHandler(log_queue)
  logger.addHandler(queue_handler)
  
  _configure_library_loggers()

  return logger

def setup_worker_logging(queue=None, level=logging.DEBUG):
  """Configure logging for worker processes."""
  
  worker_queue = queue if queue is not None else log_queue
  
  # Clear any existing handlers on root logger
  root_logger = logging.getLogger()
  # Check if already configured
  if any(isinstance(h, QueueHandler) for h in root_logger.handlers):
      return
  if root_logger.handlers:
    root_logger.handlers.clear()
  
  # Add queue handler to root logger so all child loggers use it
  queue_handler = QueueHandler(worker_queue)
  root_logger.addHandler(queue_handler)
  root_logger.setLevel(level)
  
  # Also configure the main project logger
  logger = logging.getLogger("speech_to_speech")
  logger.setLevel(level)
  logger.propagate = True  # Let it propagate to root logger with queue handler
  logger.handlers.clear()  # Clear any handlers that might have been added
  
  _configure_library_loggers()

def start_listener(level=logging.DEBUG):
  """Start a listener for log records from all processes."""
  global _listener
  
  if _listener is not None:
    return _listener  # Already started
  
  log_filter = ProjectLogFilter("speech_to_speech")
  
  # Colored formatter for console
  color_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(message)s',
    log_colors={
      'DEBUG':    'cyan',
      'INFO':     'green',
      'WARNING':  'yellow',
      'ERROR':    'red',
      'CRITICAL': 'bold_red',
    }
  )

  # Console handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(color_formatter)
  stream_handler.setLevel(level)
  stream_handler.addFilter(log_filter)
  
  handlers = [stream_handler]

  _listener = QueueListener(log_queue, *handlers)
  _listener.start()
  
  # Register cleanup function
  atexit.register(stop_listener)
  
  return _listener

def stop_listener():
  """Stop the logging listener."""
  global _listener
  if _listener is not None:
    _listener.stop()
    _listener = None

def get_logger(name=None):
  """Get a logger instance. If no name provided, returns main project logger."""
  if name is None:
    return logging.getLogger("speech_to_speech")
  return logging.getLogger(name)
  
def get_log_queue():
  """Get the logging queue for passing to worker processes."""
  return log_queue