import streamlit as st
from dotenv import load_dotenv

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from tax_llm import get_ai_response

# streamlit 은 새로고침 할 때 마다 전체 코드를 다시 실행함. 
# 따라서 대화 내용을 유지하기 위해 session_state를 사용하여 대화 내용을 저장해야 함.
# streamlit 자체가 리액트로 구성되어 있어서 리액트 최적화 되어, 변경된 부분만 다시 렌더링 함.

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")
st.title("🤖 소득세 챗봇")
st.caption("소득세 관련 질문에 답변해 드립니다. 질문을 입력해주세요.")

user_input = st.chat_input("소득세에 관련된 궁금한 내용들을 말씀해주세요. 예시: '소득세 신고 방법이 궁금해요.'")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
   with st.chat_message(message["role"]):
      st.write(message["content"])

if user_input:
  with st.chat_message("user"):
    st.write(user_input)
  
  #session_state : 대화 내용을 저장
  st.session_state.message_list.append({"role": "user", "content": user_input})

  with st.spinner("답변을 생성하는 중..."):
      ai_response = get_ai_response(user_input)
      with st.chat_message("ai"):
        ai_message = st.write_stream(ai_response)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})


#-- user Question 예시 --
#"What is the comprehensive income tax for a salaried worker with an annual salary of 50 million Korean Won?"
#"What is the comprehensive income tax for a resident with an annual salary of 50 million Korean Won?"
#연봉 5천만원 직장인의 종합소득세는?
#연간 소득이 5천만원인 직장인의 평균 종합 소득세 계산해줘.
#연간 소득이 5,500만원인 40대 일반 직장인의 평균 종합 소득 세금을 계산 해주세요.
