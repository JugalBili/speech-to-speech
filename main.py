from dotenv import load_dotenv
import os
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from src.streaming.s2s_pipeline import main as s2s_pipeline_interrupt
from config import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
  warnings.filterwarnings("ignore")

  s2s_pipeline_interrupt()

if __name__ == "__main__":
  load_dotenv()
  main()