from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)

from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# def pre_model_hook(req) -> dict:
  
#   for idx, msg in enumerate(req['messages']):
#     print(f'pre_model_hook {idx}) {msg.content}')
#     content = msg.content
#     if content is None or len(content.strip()) == 0:
#       try:
#         if 'tool_calls' in msg.additional_kwargs:
#           req['messages'][idx].content = 'call_tools'
#       except:
#         pass
#       # msg['content'] = 
  
#   new_messages = req['messages']
  
#   return {
#     "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]
#   }

def normalize_messages(messages: dict) -> dict:
  """метод для обработки истории сообщений, который заполняет поле content в AiMessage
  (т.к. возникает ошибка при обработке сообщений в котором content пустой)
  todo - доработать или найти другое решение, возможно баг
  """
  for idx, msg in enumerate(messages):
    # print(f'pre_model_hook {idx}) {msg.content}')
    content = msg.content
    if content is None or len(content.strip()) == 0:
      try:
        if 'tool_calls' in msg.additional_kwargs:
          messages[idx].content = 'Вызвал инструменты - '
          messages[idx].content += ', '.join([str(ai_tool['name']) for ai_tool in msg.tool_calls])
      except:
        pass
  return messages

def pre_model_hook(state):
  """хук предварительной отправки сообщений к модели с обрезкой истории оп max_tokens
  (примерно 10 сообщений)
  """
  trimmed_messages = trim_messages(
      state["messages"],
      strategy      = "last",
      token_counter = count_tokens_approximately,
      max_tokens    = 1000,
      start_on      = "human",
      end_on        = ("human", 'tool'),
  )
  trimmed_messages = normalize_messages(trimmed_messages)
  messages = []
  
  for idx, msg in enumerate(trimmed_messages):
    try:
      aiIdx = idx + 1
      if isinstance(msg, HumanMessage) and aiIdx < len(trimmed_messages):
        aiMessage = trimmed_messages[aiIdx];
        if isinstance(aiMessage, AIMessage) and len(aiMessage.tool_calls) > 0:
          # messages.append(RemoveMessage(id=trimmed_messages[idx].id))
          continue
    except:
      pass
    messages.append(msg)
  
  # print('\n'.join([f'{i}) {msg.content}' for i, msg in enumerate(trimmed_messages)]))
  return { "llm_input_messages": trimmed_messages[-2:] }


def print_stream(stream, output_messages_key="llm_input_messages"):
    for chunk in stream:
        for node, update in chunk.items():
            # print(f"Update from node: {node}")
            messages_key = (
                output_messages_key if node == "pre_model_hook" else "messages"
            )
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    # print(message)
                    pass
                else:
                    message.pretty_print()