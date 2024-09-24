# WG
Simple LSTM Discharge Prediction Model

## 프로젝트 개요
이 프로젝트는 LSTM(Long Short-Term Memory) 모델을 사용하여 강수량(Precipitation)과 유입량(Intake)을 기반으로 유출량(Discharge)을 예측하는 모델을 구현합니다.

## 데이터 준비
데이터는 CSV 파일 형식으로 제공되며, 다음과 같은 열을 포함합니다:
- `Date`: 날짜
- `Precipitation`: 강수량
- `Intake`: 유입량
- `Discharge`: 유출량

데이터는 `pandas` 라이브러리를 사용하여 읽고 전처리합니다.

## 모델 구조
모델은 LSTM 레이어를 사용하여 시계열 데이터를 처리합니다. 기본적인 모델 구조는 다음과 같습니다:
- LSTM 레이어
- Attention 레이어
- Projection 레이어

## 학습 방법
모델은 pytorch 라이브러리를 사용하여 학습됩니다. 학습 데이터는 학습용과 검증용으로 분할되며, `Mean Squared Error`(MSE)를 손실 함수로 사용합니다.

## 평가 방법
모델의 성능은 테스트 데이터셋을 사용하여 평가됩니다. hydroeval패키지를 사용하여 간단하게 NSE, KGE값을 계산합니다.

## 사용 예시
다음은 간단한 예시입니다:
```bash
python utils.py
```

utils.py 에는 configs의 test.yaml 파일을 읽어 이 설정 파일을 바탕으로 다양한 실험 설정 파일을 생성할 수 있습니다.
utils.py 내에 파라미터를 변경하여 각 그리드를 조합하여 파일을 생성합니다.

```bash
python main.py --config CONFIG_FILE_PATH
```

main.py 에 config 경로를 인자로 주면 그 설정파일을 읽고 학습하며 ./runs 내에 결과가 저장됩니다.
