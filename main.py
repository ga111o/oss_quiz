import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# 페이지 설정
st.set_page_config(page_title="퀴즈 앱", page_icon="❓")

# 제목
st.title("퀴즈 앱")

# 객관식 문제
st.header("객관식 문제")
st.write("다음 중 파이썬의 기본 데이터 타입이 아닌 것은?")
options = ["int", "float", "string", "array"]
selected_option = st.radio("선택하세요:", options)
correct_option = "array"  # 정답

# 주관식 문제
st.header("주관식 문제")
st.write("파이썬에서 리스트와 튜플의 차이점을 설명하세요.")
user_answer = st.text_area("답변을 입력하세요:")
correct_answer = "리스트는 수정 가능하고 튜플은 수정이 불가능합니다."

# 정답 확인 버튼
if st.button("정답 확인"):
    # 객관식 채점
    if selected_option == correct_option:
        st.success("객관식 정답입니다! 🎉")
    else:
        st.error(f"객관식 오답입니다. 정답은 '{correct_option}'입니다.")

    # 주관식 채점 (임베딩 기반)
    if user_answer:
        # 임베딩 모델 로드
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 답변 임베딩
        user_embedding = model.encode(user_answer)
        correct_embedding = model.encode(correct_answer)
        
        # 코사인 유사도 계산
        similarity = np.dot(user_embedding, correct_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(correct_embedding)
        )
        
        # 유사도가 0.7 이상이면 정답으로 처리
        if similarity >= 0.7:
            st.success(f"주관식 정답입니다! 🎉")
        else:
            st.error(f"주관식 오답입니다.")
            st.info("참고 정답: " + correct_answer)
    else:
        st.warning("주관식 답변을 입력해주세요.")