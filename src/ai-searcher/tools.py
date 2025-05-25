from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool, tool
from pydantic import BaseModel
import urllib.request
from bs4 import BeautifulSoup
from typing import Literal

PageType = Literal['competitions', 'clubs']

class Parrot(BaseModel):
  counts_list: list[int]

@tool(description='Считает количество попугаев')
def sum_parrots(parrot: Parrot) -> str:
  """Считает количество попугаев

  Args:
      parrot (Parrot): экземпляр класса Parrot в нем есть список с числами о количестве попугаев
  Returns:
      str: строка формата 'Количество - <int>'
  """
  print(parrot)
  return f'Количество - {sum([int(i) for i in parrot.counts_list if type(i)== int or i.isdigit()])}'


@tool(description='получить список чемпионатов на https://soccer365.ru/competitions/')
def get_tournament_page():
  """получить список чемпионатов
  """
  html_doc = urllib.request.urlopen(f"https://soccer365.ru/competitions")
  soup = BeautifulSoup(html_doc, 'html.parser')
  return [
    f'{tour.a["href"]}\n{tour.span.get_text()}' for tour in soup.find_all('div', {'class':'season_item'})
  ]


tools = [
  get_tournament_page,
  Tool(
      name        = 'Поиск',
      func        = DuckDuckGoSearchAPIWrapper().run,
      description = 'Полезно для поиска информацию в интернете'
  ),
  sum_parrots,
]