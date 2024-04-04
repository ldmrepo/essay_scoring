import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from predictor import Predictor
from model import KoBERTRubricMTLScorer, KoBERTRubricScorer, KoBERTRubricClassifier
from ensemble import RubricScorerEnsemble

# 모델 및 토크나이저 로드
MODEL_NAME = 'monologg/kobert'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 모델 인스턴스 생성 및 학습된 가중치 로드
@st.cache(allow_output_mutation=True)
def load_models():
    models = {
        "MTL Scorer": KoBERTRubricMTLScorer(num_tasks=3, num_classes=5),
        "Rubric Scorer": KoBERTRubricScorer(num_tasks=3),
        "Rubric Classifier": KoBERTRubricClassifier(num_classes=5)
    }
    models["Rubric Scorer"].load_state_dict(torch.load("KoBERTRubricScorer_weights.pt"))
    models["MTL Scorer"].load_state_dict(torch.load("KoBERTRubricMTLScorer_weights.pt"))
    models["Rubric Classifier"].load_state_dict(torch.load("KoBERTRubricClassifier_weights.pt"))
    return models

# 예측 및 시각화
def predict_and_visualize(models, selected_models, prompt, response):
    all_predictions = {}
    for model_name in selected_models:
        model = models[model_name]
        predictor = Predictor(model=model, tokenizer=tokenizer)
        scores, _ = predictor.predict(prompt, response)
        all_predictions[model_name] = scores
        st.write(f"{model_name} 예측 점수: {scores}")

    if len(all_predictions) > 1:
        ensemble_scores = np.mean(list(all_predictions.values()), axis=0)
        st.write(f"앙상블 예측 점수: {ensemble_scores}")

    plt.figure(figsize=(10, 6))
    for model_name, scores in all_predictions.items():
        plt.plot(scores, label=model_name)
    if len(all_predictions) > 1:
        plt.plot(ensemble_scores, label="Ensemble", linewidth=2, linestyle='--')
    plt.title("모델별 예측 점수 및 앙상블 결과")
    plt.ylabel("점수")
    plt.xlabel("루브릭 항목")
    plt.legend()
    st.pyplot(plt)

# Streamlit 애플리케이션 정의
def main():
    st.title("텍스트 평가 서비스")

    # 사용자 입력
    prompt = st.text_area("Prompt", "인공지능의 발전이 사회에 미치는 영향에 대해 논하시오.")
    response = st.text_area("Response", "인공지능의 발전은 사회에 많은 변화를 가져올 것입니다.")

    # 모델 선택
    models = load_models()
    selected_models = st.multiselect('모델 선택', list(models.keys()), default=list(models.keys()))

    # 예측 및 시각화
    if st.button("Evaluate"):
        predict_and_visualize(models, selected_models, prompt, response)

if __name__ == "__main__":
    main()