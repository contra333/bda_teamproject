# KOCOH 데이터

원본 데이터는 Git에 올리지 않는다. 아래 고정 버전을 내려받아 같은 결과를 재현한다.

- 데이터셋: KOCOH v2 (E-Park/KOCOH)
- 출처: https://huggingface.co/datasets/E-Park/KOCOH
- 원본 설명: https://github.com/eparkatgithub/KOCOH
- 논문: Park & Song (2026), https://aclanthology.org/2026.lrec-1.324/
- 라이선스 표기: Hugging Face CC BY-NC 4.0 / GitHub README TBD (2026-10-12 확인). 출처를 표시하고 비영리 수업 연구에 사용한다. 두 출처의 라이선스 표기가 달라 추가 확인이 필요하다.
- 규모: 3,000행 × 18열, 101개 Set. 1행은 Context–Comment 쌍이다.
- 고정 revision: `c26d8e86f3d7a9a6dcd4e308cb747b6bc6427d6f`
- SHA256: `d5eb3c7bb4fd97471bd77efd9eb7f98ebc162151ef9725e821510551a4485a38`

프로젝트 루트에서 `python -m src.download_data`를 실행한다. `data/KOCOH_v.2.csv`와 버전·해시를 담은 `data/provenance.json`을 저장한다. 기존 원본의 해시가 다르면 덮어쓰지 않고 중단한다.

## 분할과 모델 입력

`src/preprocess.py`는 같은 Context를 공유하는 Set을 합쳐 **77개 post_group**을 만든다. 묶음 단위 StratifiedGroupKFold(5, shuffle=True, seed=42)를 사용하므로 같은 게시글 묶음과 Context가 train/test 양쪽에 들어가지 않는다.

각 바깥 train을 다시 묶음 단위 **4-fold**로 나누어 OOF 예측으로 임계값을 선택한다. 이후 바깥 train 전체로 재학습하고 test에서 평가한다. 모든 모델은 같은 분할을 사용한다. `python src/baseline.py` 실행 시 `results/split_assignment.csv`와 `results/inner_val_composition.csv`가 갱신된다. 분할만 확인하려면 `python src/preprocess.py --inner`를 실행한다.

모델 입력은 Comment 또는 Context·Comment이며, 길이 기준선만 이 텍스트에서 계산한 글자 수를 추가한다. Type·식별자·부가 주석은 모델 입력에 포함하지 않는다. TF-IDF와 표준화 등 학습되는 변환은 train에서만 fit한다. 필수 텍스트·정답의 결측은 제거 개수를 출력하며, 현재 원본의 제거 대상은 0행이다.
