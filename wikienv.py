import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

import wikipediaapi
import spacy

NERLIST=['PERSON', 'NOPR', 'FACILITY', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'PER', 'MISC', 'EVT', 'PROD', 'DRV', 'GPE_LOC', 'GPE_ORG']

def show_ents(sent):
  '''
    find the entities in the ambiguous search
  '''
  if sent.ents:
    ent_list=[]
    for ent in sent.ents:
      if ent.label_ in NERLIST:
        ent_list.append(ent.text)
    return ent_list

# import wikipedia

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    self.wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    response_text = requests.get(search_url).text
    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
      # nlp=spacy.load('en_core_web_sm')
      # ent_list=show_ents(nlp(entity))
      # self.obs = f"Could not find {entity}. Similar: {ent_list}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        self.page = ""
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

  def print_sections(self, entity):
    """
    To get all top level sections of page, you have to use property sections. 
    It returns list of WikipediaPageSection, so you have to use recursion to get all subsections.
    """
    page_py = self.wiki_wiki.page(entity)

    def print_sections(sections, level=0):
      result = ''
      for s in sections:
        # print(f"{'*' * (level + 1)}: {s.title} - {s.text[0:40]}")
        result += f"{'*' * (level + 1)}: {s.title}\n"
        result += print_sections(s.sections, level + 1)
      return result
    
    return print_sections(page_py.sections)
  
  def sections_by_title(self, entity, section_title):
    """
    To get all sections of page with given title, you have to use function sections_by_title. 
    It returns the all WikipediaPageSection with this title.

    NOTE: there is also a function on the wikipedia-api that returns one section, 
    but we just return everything
    """
    page_py = self.wiki_wiki.page(entity)
    return page_py.sections_by_title(section_title).text
  
  def links(self, entity):
    """
    If you want to get all links to other wiki pages from given page, 
    you need to use property links. It's map, where key is page title and value is WikipediaPage.
    """
    page_py = self.wiki_wiki.page(entity)
    result = ""
    links = page_py.links
    for title in sorted(links.keys()):
      result += f"{title}: {links[title]}\n"
    return result
    
  
  def page_categories(self, entity):
    """
    If you want to get all categories under which page belongs, you should use property categories. 
    It's map, where key is category title and value is WikipediaPage.
    """
    # raise NotImplementedError
    page_py = self.wiki_wiki.page(entity)
    result = ""

    categories = page_py.categories
    for title in sorted(categories.keys()):
      result += f"{title}: {categories[title]}\n"
    return result
  
  def pages_from_category(self, category):
    """
    To get all pages from given category, you should use property categorymembers.
    It returns all members of given category. You have to implement recursion and deduplication by yourself.
    """
    # raise NotImplementedError
    category_py = self.wiki_wiki.page(category)
    def print_categorymembers(categorymembers, level=0, max_level=1):
      result = ""
      for c in categorymembers.values():
        result += f"{c.title}: {c.ns} (ns: {c.ns})\n"
        # print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)
      return result
    
    return print_categorymembers(category_py.categorymembers)

  
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    if self.answer is not None:  # already finished
      done = True
      return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."

    # add more actions here
    elif self.action_is(action, "print_sections"):
      entity = action[len("print_sections["):-1]
      self.obs = self.print_sections(entity)
    elif self.action_is(action, "sections_by_title"):
      entity, section_title = action[len("sections_by_title["):-1].split(",")
      self.obs = self.sections_by_title(entity, section_title)
    elif self.action_is(action, "links"):
      entity = action[len("links["):-1]
      self.obs = self.links(entity)
    elif self.action_is(action, "page_categories"):
      entity = action[len("page_categories["):-1]
      self.obs = self.page_categories(entity)
    elif self.action_is(action, "pages_from_category"):
      category = action[len("pages_from_category["):-1]
      self.obs = self.pages_from_category(category)

    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
  
  @staticmethod
  def action_is(s, action):
    return s.startswith(f"{action}[") and s.endswith("]")
