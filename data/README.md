# data/

원본 데이터 파일은 Git에 올리지 않는다 (`.gitignore`에서 제외). 이 문서만 보고
누구나 같은 데이터를 다시 받을 수 있게 아래를 채운다.

- 데이터셋 이름: KOCOH v2 (E-Park/KOCOH)
- 출처 URL: https://huggingface.co/datasets/E-Park/KOCOH
- 원본 설명: https://github.com/eparkatgithub/KOCOH
- 논문: Park & Song (2026), https://aclanthology.org/2026.lrec-1.324/
- 라이선스 표기: Hugging Face CC BY-NC 4.0 / GitHub README TBD (2026-10-12 확인). 출처를 표시하고 비영리 수업 연구에 사용한다. 표기 차이는 임의로 해소했다고 주장하지 않는다.
- 규모: 3,000행 × 18열, 101개 Set. 1행은 Context–Comment 쌍이다.
- 버전 고정: c26d8e86f3d7a9a6dcd4e308cb747b6bc6427d6f
- SHA256: d5eb3c7bb4fd97471bd77efd9eb7f98ebc162151ef9725e821510551a4485a38
- 받는 방법: WSL Ubuntu-24.04에서 프로젝트 루트로 이동한 뒤 `~/envs/research/bin/python -m src.download_data`
- 저장할 파일명: `KOCOH_v.2.csv` (원본), `provenance.json` (버전·해시)
- 기존 원본은 덮어쓰지 않는다. 해시가 다르면 중단하고 원인을 확인한다.

## 현재 중간발표 분할

`src/midterm/preprocess.py`는 같은 Context를 공유하는 Set을 합쳐 77개 post_group을 만들고, 묶음 단위 StratifiedGroupKFold(5, shuffle=True, seed=42)를 사용한다. `src/midterm/preprocess_v2.py`는 바깥 train에 묶음 단위 4-fold를 더한다. 임계값을 내부 OOF로 정한 뒤 train 전체로 재학습하며 test는 선택에 쓰지 않는다. 제공·재현 분할표는 `results/midterm/`에 있다. 원본 행·문장은 유지한다.

## W5 당시 분할과 모델 입력 (과거 EDA 재현용)

`~/envs/research/bin/python -m src.preprocess`로 `split_assignment.csv`, `split_recipe.json`, `train.csv`, `valid.csv`, `test.csv`를 만든다. 출력은 모두 로컬 데이터이며 Git에 올리지 않는다.

Set 기준 5-fold StratifiedGroupKFold, shuffle=True, seed=42에서 fold 0을 test, fold 1을 valid, 나머지를 train으로 사용한다. scikit-learn 1.9.1 기준 1,770/639/591행이다. 원본 순서를 바꾸거나 성능에 맞춰 seed를 다시 고르지 않는다. 기존 분할과 재계산 결과가 달라지면 자동 덮어쓰기하지 않는다.

CSV에는 점검용 메타데이터가 남아 있다. 모델에는 `src.preprocess.model_inputs`로 선택한 `Comment` 또는 `Context`, `Comment`만 전달한다. `Type`, `Set`, `Index`, 부가 주석은 입력 특성이 아니다. TF-IDF와 기타 학습되는 변환은 train에만 fit한다.

Index 1500(Type 1, Set 43)과 1501(Type 2, Set 44)의 댓글은 모두 P다. 제작상 대응 사례로 보이지만 Set이 달라 자동 수정하지 않았다. Set 44는 출처를 추론할 원본 게시글 정보도 없다. 분할 간 반복 텍스트의 실제 개수는 `results/split_overlap.csv`를 확인한다.

> 4주차 데이터 수집 계획서와 같은 내용이 여기 남으면 된다.
