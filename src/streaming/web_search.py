from ddgs import DDGS
from ddgs.exceptions import DDGSException
import logging
import wikipedia
import fandom
import os
import random
from urllib.parse import urlparse, unquote, urljoin
from urllib.robotparser import RobotFileParser
import urllib.request
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass
import pandas as pd
from io import StringIO
import requests
import re

class WebSearcher:
  def __init__(self, interrupt_count : SynchronizedClass):
    self.logger = logging.getLogger("speech_to_speech.web_search")
    self.interrupt_count = interrupt_count

    self.robot_parser = RobotFileParser()
    self.playwright = sync_playwright().start()
    self.browser = self.playwright.firefox.launch(headless=True)
    self.timeout = 5000 # milliseconds
    self.website_scrape_limit = 3
        
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    user_agent_file_path = os.path.join(project_root_dir, "data", "user-agents.txt")
    user_agent_file = open(user_agent_file_path, 'r')
    self.user_agents = user_agent_file.read().split("\n")


  def _can_fetch(self, website):
    try:
      parsed_url = urlparse(website)
      robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
      req = urllib.request.Request(
          robots_url,
          headers={"User-Agent": self.user_agents[random.randint(0, len(self.user_agents) - 1)]}
      )

      with urllib.request.urlopen(req, timeout=1) as response:
        robots_txt = response.read().decode("utf-8", errors="ignore")

        # Parse it into the RobotFileParser
        self.robot_parser = RobotFileParser()
        self.robot_parser.parse(robots_txt.splitlines())

        return self.robot_parser.can_fetch(self.user_agents[random.randint(0, len(self.user_agents) - 1)], website)

    except Exception as e:
        self.logger.error(f"Error reading robots.txt for {website}: {e}")
        return True  # Assume allowed if robots.txt can't be read

  
  def parse_tables(self, html, class_filter):
    soup = BeautifulSoup(html, "html.parser")
    tables_html = soup.find_all("table", class_=class_filter)

    table_texts = []
    for idx, table in enumerate(tables_html, start=1):
      caption = ""
      try:
        caption_tag = table.find("caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else ""
        caption = re.sub(r"\[\d+\]", "", caption).strip()

        df_list = pd.read_html(StringIO(str(table)))
        if df_list:
          df = df_list[0]

          # Ensure column headers are strings
          headers = [str(h) for h in df.columns]
          
          if headers and headers[0].lower().startswith("vte"):
            continue

          # Build markdown header row
          header_row = "| " + " | ".join(headers) + " |"
          separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

          # Build data rows
          data_rows = [
            "| " + " | ".join(str(cell) for cell in row) + " |"
            for row in df.to_numpy()
          ]

          # Combine all into one table string
          markdown_table = "\n".join([header_row, separator_row] + data_rows)
          table_texts.append(f"[Table] {caption}\n{markdown_table}")

      except Exception as e:
        self.logger.error(f"Error parsing table {caption}: {e}")

    return table_texts


  def _fetch_wiki_content(self, website):
    parsed_url = urlparse(website)
    path_components = parsed_url.path[1:].split("/")
    title = path_components[path_components.index("wiki") + 1]
    title = unquote(title)  # Decode %20 etc.
    
    page = wikipedia.page(title, auto_suggest=False)
    
    response = requests.get(website)
    response.raise_for_status()
    html = response.text

    tables = self.parse_tables(html, "wikitable")
    
    return page.content.replace("\n", ""), tables


  def _fetch_fandom_content(self, website):
    fandom.set_user_agent(self.user_agents[random.randint(0, len(self.user_agents) - 1)])
    parsed_url = urlparse(website)
    
    fandom_page = parsed_url.netloc.split(".")[0]
    
    path_components = parsed_url.path[1:].split("/")
    title = path_components[path_components.index("wiki") + 1]
    title = unquote(title)  # Decode %20 etc.
    
    fandom.set_wiki(fandom_page)
    page = fandom.page(title)
    
    response = requests.get(website)
    response.raise_for_status()
    html = response.text

    tables = self.parse_tables(html, "wikitable")
    
    return page.plain_text.replace("\n", ""), tables


  def _fetch_other_content(self, website):
    html = None
    
    try:
      page = self.browser.new_page()

      page.set_extra_http_headers({
        "User-Agent": self.user_agents[random.randint(0, len(self.user_agents) - 1)]
      })
      page.goto(website, timeout=self.timeout, wait_until="domcontentloaded")
      html = page.content()
      page.close()

    except Exception as e:
      self.logger.error(e)
      return None, None

    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["script", "source", "style", "head", "img", "svg", "a", "form", "link", "iframe"]):
      tag.decompose()
  
    # Remove hidden elements
    for element in soup.select('[aria-hidden="true"], [style*="display:none"], [style*="visibility:hidden"]'):
      element.decompose()

    # Remove classes and data-* attributes
    for element in soup.find_all(True):  # True = all tags
      if "class" in element.attrs:
        del element.attrs["class"]
      for attr in list(element.attrs):
        if attr.startswith("data-"):
          del element.attrs[attr]

    # Extract and clean text
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())  # Collapse multiple spaces

    return text.replace("\n", ""), []


  def fetch_content(self, websites):
    websites_read = 0
    website_data = []

    for website in websites:
      if self.interrupt_count.value > 0:
        return None

      content = None
      tables = []
      if "wikipedia.org" in website:
        content, tables = self._fetch_wiki_content(website)
      elif "fandom.com" in website:
        content, tables = self._fetch_fandom_content(website)
      else:
        scrapable = self._can_fetch(website)
        if scrapable:
          content, tables = self._fetch_other_content(website)

      if content != None:
        websites_read += 1

        website_data.append({
          "content": content,
          "tables": tables,
          "source": website
        })

      if websites_read == self.website_scrape_limit:
        break

    return website_data  


  def ddg_search(self, request):
    retry_search = True
    results = None

    while retry_search:
      self.logger.debug("Search started")
      try: 
        ddgs_client = DDGS(timeout=2)
        results = ddgs_client.text(request, region="us-en", max_results=10, backend="duckduckgo")
        retry_search = False  
        
      except DDGSException as e:
        self.logger.error(e)
        self.logger.debug("Retrying search")

    self.logger.info(request)
    self.logger.info(results)

    results = [entry["href"] for entry in results]
    
    return results


if __name__ == "__main__":
  # for testing purposes

  client = WebSearcher(Value("i", 0))
  content = client.fetch_content([])

  print(content)