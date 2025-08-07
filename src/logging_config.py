import logging
import colorlog
import warnings

def setup_logging(level=logging.DEBUG):
  """Configure logging with colorlog once."""
  # Create a ColoredFormatter
  formatter = colorlog.ColoredFormatter(
      # '%(log_color)s%(levelname)s:%(name)s:%(message)s',
      '%(log_color)s%(message)s',
      log_colors={
          'DEBUG':    'cyan',
          'INFO':     'green',
          'WARNING':  'yellow',
          'ERROR':    'red',
          'CRITICAL': 'bold_red',
      }
  )

  handler = logging.StreamHandler()
  handler.setFormatter(formatter)

  # Project logger
  project_logger = logging.getLogger("speech_to_speech")
  project_logger.setLevel(level)
  project_logger.propagate = False  # Don't pass to root logger
  if not project_logger.handlers:
      project_logger.addHandler(handler)
      
#   logging.getLogger("ctranslate2").setLevel(logging.CRITICAL)
#   logging.getLogger("transformers").setLevel(logging.CRITICAL)
#   logging.getLogger("kokoro").setLevel(logging.CRITICAL)
#   # Ignore all warnings
#   warnings.filterwarnings("ignore")