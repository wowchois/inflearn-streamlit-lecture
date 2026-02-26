from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from few_shot_config import answer_examples

store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
  embedding = OllamaEmbeddings(model="bge-m3") #embedding model change -> "bge-m3" is better than "nomic-embed-text" for korean text
  index_name = "chroma_markdown_tax_bge_m3" #english case : "chroma-eng-tax"
  database = Chroma(
      persist_directory="./chroma-markdown-tax-bge-m3", #english case : "./chroma-eng-tax"
      collection_name=index_name,
      embedding_function=embedding,
  )
  retriever = database.as_retriever(search_kwargs={'k': 2}) #change k(4 -> 2) value to adjust number of retrieved documents

  return retriever


def get_llm():
  llm = ChatOllama(model="qwen2.5:3b", num_ctx=2048, num_predict=256, timeout=30) #llama3.2 > 지시 잘 안따름, #phi3.5 > english model
  return llm


def get_history_retriever():
  llm = get_llm()
  retriever = get_retriever()
  # 검색 쿼리 재구성 프롬프트
  contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
  
  contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
  ])
  history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
  )

  return history_aware_retriever


def get_dictionary_chain():
  llm = get_llm()
  dictionary = ["사람을 나타내는 표현 -> 거주자"] #["Expressions referring to a person -> resident"]

# Please review the user's question and refer to our dictionary to modify it if necessary.
#   If you determine that no modification is needed, you may leave the user's question as is.
#   In that case, please return only the question.
#dictionary: {dictionary}
#User's question: {{query}}

  prompt = ChatPromptTemplate.from_template(f"""
  사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
  만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
  그런 경우에는 질문만 리턴해주세요

  사전: {dictionary}
  사용자의 질문: {{query}}
  """)

  dictionary_chain = prompt | llm | StrOutputParser()

  return dictionary_chain


def get_rag_chain():
  llm = get_llm()

  # few-shot 예시 포함 프롬프트
  example_prompt = ChatPromptTemplate.from_messages(
      [
          ("human", "{input}"),
          ("ai", "{answer}"),
      ]
  )

  few_shot_prompt = FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=answer_examples
  )

  # 답변 생성 프롬프트
  system_prompt = (
    # "You are an expert in Korean Income Tax Law. Please answer the user's questions regarding income tax law."
    # "Utilize the documents provided below to formulate your answers."
    # "If you do not know the answer, please say so."
    # "Answer in exactly 2-3 sentences. No more."
    # "Please answer questions by stating the **conclusion first**, followed by the supporting details."
    # "When providing an answer, begin with 'According to Article (XX) of the Income Tax Act, ...'"
    "당신은 한국의 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
    "아래에 제공된 문서를 활용해서 답변해주시고"
    "2-3 문장의 짧은 내용의 답변을 제공해주세요. 그 이상은 제공하지 말아주세요."
    "답변을 알 수 없다면 모른다고 답변해주세요"
    "질문에 답변할 때는 **결론**부터 말하고, 그 다음에 근거를 설명해주세요."
    "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주세요"
    "\n\n"
    "{context}"
  )

  qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    few_shot_prompt,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
  ])

  # 1. 대화 맥락 고려해서 문서 검색
  history_aware_retriever = get_history_retriever()

  # 2. 검색된 문서 합쳐서 LLM에 전달
  combine_documents_chain = create_stuff_documents_chain(llm, prompt=qa_prompt)

  # 3. 두 개를 연결해서 최종 RAG 체인 완성
  rag_chain = create_retrieval_chain(history_aware_retriever, combine_documents_chain)

  chain_with_history = RunnableWithMessageHistory(
      rag_chain,
      get_by_session_id,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer"
  ).pick("answer")

  return chain_with_history


def get_ai_response(user_message):
  dictionary_chain = get_dictionary_chain()
  rag_chain = get_rag_chain()
  concat_chain = {"input": dictionary_chain} | rag_chain

  ai_response = concat_chain.stream({
    "query": user_message
  }, config={"configurable": {"session_id": "user_1"}})

  return ai_response

