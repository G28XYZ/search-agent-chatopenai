import uuid
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_core.messages import SystemMessage
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils import print_stream, pre_model_hook
from tools import tools


model = ChatOpenAI(
  model           = "gpt-4o",
  stream_usage    = True,
  openai_api_base = "http://localhost:10000/",
  openai_api_key  = "not-needed",
  temperature     = 0,
  max_retries     = 5,
)

class State(AgentState):
    context: dict[str, any]


memory              = MemorySaver()
prompt_text         = 'Поговорите с человеком, ответив на следующие вопросы как можно лучше.'
config              = { "configurable": { "thread_id": uuid.uuid4().hex } }
summarization_model = model.bind(max_tokens=512)


agent = create_react_agent(
  model          = model,
  tools          = tools,
  checkpointer   = InMemorySaver(),
  prompt         = SystemMessage(prompt_text),
  # pre_model_hook = summarization_node,
  pre_model_hook  = pre_model_hook,
  # state_schema=State,
)

# inputs = {"messages": [HumanMessage('Что умеешь ?')]}
# print_stream(agent.stream(inputs, config=config, stream_mode="updates"))


while True:
  user_input = HumanMessage(input('MSG: '))
  inputs = {"messages": [user_input]}
  print_stream(agent.stream(inputs, config=config, stream_mode="updates"))
  # print(agent.invoke({ "messages": [user_input] }, config)["messages"][-1].pretty_print())
