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


logger = logging.getLogger("speech_to_speech.web_search")


class WebSearcher:
  def __init__(self):
    self.robot_parser = RobotFileParser()
    self.playwright = sync_playwright().start()
    self.browser = self.playwright.firefox.launch(headless=True)
    self.timeout = 5000 # milliseconds
    self.website_scrape_limit = 3
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    user_agent_file_path = os.path.join(current_dir, 'user-agents.txt')
    user_agent_file = open(user_agent_file_path, 'r')
    self.user_agents = user_agent_file.read().split("\n")


  def _can_fetch(self, website):
    # print("checkign scrapable")
    # parsed_url = urlparse(website)
    # robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    # print(robots_url)
    # self.robot_parser.set_url(robots_url)
    # try:
    #     print("reading")
    #     self.robot_parser.read()
    #     print("read")
    #     return self.robot_parser.can_fetch(self.user_agents[random.randint(0, len(self.user_agents) - 1)], website)
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
        logger.error(f"Error reading robots.txt for {website}: {e}")
        return True  # Assume allowed if robots.txt can't be read


  def _fetch_wiki_content(self, website):
    parsed_url = urlparse(website)
    path_components = parsed_url.path[1:].split("/")
    title = path_components[path_components.index("wiki") + 1]
    title = unquote(title)  # Decode %20 etc.
    
    page = wikipedia.page(title, auto_suggest=False)
    
    return page.content.replace("\n", "")


  def _fetch_fandom_content(self, website):
    fandom.set_user_agent(self.user_agents[random.randint(0, len(self.user_agents) - 1)])
    parsed_url = urlparse(website)
    
    fandom_page = parsed_url.netloc.split(".")[0]
    
    path_components = parsed_url.path[1:].split("/")
    title = path_components[path_components.index("wiki") + 1]
    title = unquote(title)  # Decode %20 etc.
    
    fandom.set_wiki(fandom_page)
    page = fandom.page(title)
    
    return page.plain_text.replace("\n", "")


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
      logger.error(e)
      return None

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
    return text.replace("\n", "")


  def fetch_content(self, websites):
    websites_read = 0
    website_data = []

    for website in websites:
      content = None
      if "wikipedia.org" in website:
        content = self._fetch_wiki_content(website)
      elif "fandom.com" in website:
        content = self._fetch_fandom_content(website)
      else:
        scrapable = self._can_fetch(website)
        if scrapable:
          content = self._fetch_other_content(website)

      if content != None:
        websites_read += 1

        website_data.append({
          "content": content,
          "source": website
        })

      if websites_read == self.website_scrape_limit:
        break

    return website_data  


  def ddg_search(self, request):
    retry_search = True
    results = None

    while retry_search:
      logger.debug("Search started")
      try: 
        ddgs_client = DDGS(timeout=2)
        results = ddgs_client.text(request, region="us-en", max_results=10, backend="chrome,duckduckgo,brave")
        retry_search = False  
        
      except DDGSException as e:
        logger.error(e)
        logger.debug("Retrying search")

    logger.info(results)

    results = [entry["href"] for entry in results]
    
    return results


if __name__ == "__main__":
  # for testing purposes

  client = WebSearcher()
  content = client.fetch_content([""])

  print(content)