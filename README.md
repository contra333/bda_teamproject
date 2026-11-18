# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

KOCOH v2 3,000행을 사용한다. 같은 Context를 공유하는 Set을 77개 게시글 묶음으로 합쳐 5-fold(seed 42)로 평가하며, 임계값은 각 train 내부의 묶음 단위 4-fold OOF 예측으로 선택한다. 데이터 출처와 전처리 기준은 [데이터 설명](data/README.md)에 있다.

## 폴더 구조

```text
src/
  preprocess.py   전처리와 5-fold 분할
  baseline.py     Dummy · TF-IDF 기준선
  plm.py          PLM 학습 (고정 임베딩 · fine-tuning · LoRA)
  evaluate.py     저장된 점수로 지표와 판정 계산
notebooks/        EDA, 중간보고서, 가설 2·추가검증, Colab 실행 노트북
results/          결과 CSV
figures/          그림
docs/             보고서와 발표자료
data/  model/     원본 데이터와 모델 가중치 (커밋하지 않음)
```

## 코드와 결과

EDA와 모델 비교 노트북에 실행 결과와 해석을 함께 저장했다.

| 내용 | 코드·노트북 | 결과 파일 |
|---|---|---|
| EDA · 5주차 | [EDA 노트북](notebooks/project_data_eda.ipynb) · [전처리 코드](src/preprocess.py) | [결측률](figures/kocoh_missing_ratio.png) · [클래스 비율](figures/kocoh_class_ratio.png) · [분할 비교](figures/kocoh_split_compare.png) |
| 전처리·데이터 분할 · 5주차 | [전처리 코드](src/preprocess.py) | [행별 분할표](results/split_assignment.csv) · [내부 검증 구성](results/inner_val_composition.csv) |
| 기준선 · 6주차 | [학습·평가 코드](src/baseline.py) · [모델 비교 노트북](notebooks/project_mid_report.ipynb) | [성능 80행](results/baseline_folds.csv) · [유형별 결과](results/baseline_by_type.csv) |
| 댓글만 사용했을 때의 상한 진단 | [기준선 코드](src/baseline.py) | [정답 구성에 따른 진단값](results/comment_only_ceiling.csv) |
| PLM 탐색 | [PLM 코드](src/plm.py) · [모델 비교 노트북](notebooks/project_mid_report.ipynb) | [성능 60행](results/plm_folds.csv) · [모델 비교](figures/kocoh_mid_report.png) · [LoRA 비교](figures/kocoh_lora_vs_finetune.png) |
| 가설 1 · 읽는 방식 비교 | [PLM 코드](src/plm.py) · [평가 코드](src/evaluate.py) | [행별 점수](results/plm_h1_preds.csv) · [내부 validation 점수](results/plm_h1_inner_preds.csv) · [요약표](results/h1_summary.csv) · [fold별 지표](results/h1_folds.csv) · [판정](results/h1_tests.csv) · [임계값 비교](results/h1_threshold.csv) |
| 가설 2 · 글 바꿔 끼우기 | [가설 2 노트북](notebooks/project_h2.ipynb) · [PLM 코드](src/plm.py) | [겉모습·내용 효과](figures/project_h2_style_vs_content.png) · [조건별 혐오 판정 비율](figures/project_h2_af_hate_rate.png) |
| 가설 2 추가검증 · 겉모습과 주제 분리 | [추가검증 노트북](notebooks/project_h2_plus.ipynb) · [PLM 코드](src/plm.py) | [글별 주제 표시(Claude 판단)](results/h2_context_topics.csv) · [후보 수](results/h2_swap_candidates.csv) · [세 모델 행별 점수](results/plm_h2_topic_preds.csv) · [학습 기록](results/plm_h2_topic.csv) · [조건별 혐오 확률](figures/project_h2_plus_conditions.png) · [효과 크기](figures/project_h2_plus_effects.png) · [Claude 판정 비교](results/h2_claude_label_compare.csv) · [Claude 판정 그림](figures/project_h2_plus_claude.png) · [세 모델 비교](figures/project_h2_plus_models.png) · [붙인 글별 비교](figures/project_h2_plus_conditions_by_post.png) |

6주차 두 기준선의 macro-F1은 **Dummy 최빈값 0.427**, **TF-IDF + Logistic Regression(댓글만) 0.647**이다(5-fold 평균). `baseline_folds.csv`에는 기본 10조합과 임계값 조정 6조합의 fold별 성능이 들어 있다.

EDA 노트북의 train 2,410행 / test 590행은 첫 fold 예시다. 분할표 15,000행은 원본 3,000행 × 5개 fold의 배정을 기록한 것이다. 현재 결과는 seed 42의 5-fold이며, 여러 seed의 반복실험과 구별한다. 댓글 상한 진단은 현재 정답 구성을 이용한 값이다.

## 실행

Python 3.12와 `requirements.txt`의 패키지를 사용한다. 아래 명령은 저장소 루트에서 실행하며, 가상환경 생성·활성화 예시는 Linux/WSL의 Bash 기준이다. CPU 기준선은 Ubuntu 24.04에서 실행을 확인했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m src.download_data
python src/preprocess.py --eda
python src/preprocess.py
python src/baseline.py
```

전처리 명령은 EDA 그림 3개를, 기준선 명령은 `results/`의 CPU 결과 CSV 5개를 갱신한다. 노트북은 같은 환경의 커널로 위에서부터 실행한다. 분할만 확인하려면 `python src/preprocess.py --inner`를 사용한다.

`plm_folds.csv`는 Apple GPU(MPS, fp32)에서 측정한 PLM 탐색 결과 60행이다. 모델 비교 노트북은 기본적으로 저장된 PLM 결과를 읽으며, `RUN_TRAINING=False`, `CHECK_MODEL_TOKENS=False`로 되어 있다. PLM 재학습에는 별도로 torch·transformers·peft와 모델 가중치가 필요하다.

## PLM 실험

학습은 `src/plm.py`, 평가는 `src/evaluate.py`가 맡는다. `plm.py`가 test 행마다 점수를 파일로 남기고, `evaluate.py`는 그 파일만 읽어 표를 만든다. 그래서 지표를 추가하거나 임계값을 바꿀 때 다시 학습하지 않아도 된다. 점수 덧셈 조건은 각 fold의 train 안에서 만든 내부 validation 점수로 결합 비율을 정하므로, 2번 단계의 점수가 있어야 계산된다.

### 실행

```bash
# 1. 학습: 가설 1의 네 조건, seed 3개. 끊기면 같은 명령을 다시 실행한다 (끝난 학습은 건너뜀)
python src/plm.py --models KcELECTRA-base-v2022 --methods finetune --out results/plm_h1.csv --resume     --seeds 42 43 44 --conditions Comment-only Context+Comment Context-only Bi-encoder

# 2. 점수 덧셈과 임계값 선택에 쓰는 내부 validation 점수
python src/plm.py --models KcELECTRA-base-v2022 --methods finetune --out results/plm_h1.csv --inner     --seeds 42 43 44 --conditions Comment-only Context+Comment Context-only Bi-encoder

# 3. 평가: PLM 학습 없이 몇 초면 끝난다
python src/evaluate.py --preds results/plm_h1_preds.csv
```

1·2번은 GPU가 필요하다. Colab에서는 [실행 노트북](notebooks/colab_h1_rerun.ipynb)을 쓴다. 결과가 이미 저장소에 있으므로 표만 다시 보려면 3번만 실행하면 된다. 옵션은 `python src/plm.py --help`와 각 파일 맨 위 설명에 있다.

### 결과 파일

| 파일 | 내용 | 만드는 단계 |
|---|---|---|
| `results/plm_h1.csv` | 학습 한 번마다의 점수, 학습 시간, 메모리, 기기 | 1 |
| `results/plm_h1_preds.csv` | test 행별 점수. `score`가 0보다 크면 혐오 (혐오 logit − 비혐오 logit) | 1 |
| `results/plm_h1_inner_preds.csv` | 내부 validation 행별 점수 (점수 덧셈, 임계값 선택에 사용) | 2 |
| `results/h1_summary.csv` | 조건별 요약 (macro-F1, TPR, FPR, AUROC, PR-AUC, 혼동행렬, 짝 정답률) | 3 |
| `results/h1_folds.csv` | fold별 지표 | 3 |
| `results/h1_tests.csv` | 조건 간 비교 판정 | 3 |
| `results/h1_threshold.csv` | 임계값 0.5와 조정 후의 비교 | 3 |

`--out`을 바꾸면 이름이 같은 규칙으로 정해진다. `results/plm_x.csv`로 학습하면 점수는 `plm_x_preds.csv`, 평가 결과는 `x_summary.csv` 등이 된다.

### 알아 둘 것

- `plm.py`는 학습 전에 데이터 분할이 `results/split_assignment.csv`와 같은지 확인하고, 다르면 멈춘다. scikit-learn 버전에 따라 분할이 달라질 수 있기 때문이다. `requirements.txt`의 버전(1.9.1)을 쓴다.
- `split_assignment.csv`는 `baseline.py`와 `preprocess.py --inner`가 다시 쓴다. 이 파일이 바뀌어 있으면 커밋하지 않고 원인을 먼저 확인한다.
- 가설 1의 결과는 기준 분할에서 seed 42·43·44로 다시 돌린 것이다(Colab T4). 처음 실행은 분할이 기준과 달랐고, 그 점수는 Git 이력에 있다.

## 보고서와 연구 기록

- 9주차: [중간보고서](docs/중간보고서_수수컴정.docx) · [발표자료](docs/중간발표_수정본.pptx)
- Notion: [연구관리](https://app.notion.com/p/bda0538d38d3837793b3016b2d21c682) · [실험로그](https://www.notion.so/4050538d38d3817cb014d68a07e4a72d)
- 다음 일정: 발표자료 기준 10주차 seed 3회 반복, 11주차 결과 분석.
- 협업: [팀 규약](TEAM.md). 저장소를 복제한 뒤 `git config core.hooksPath hooks`를 설정한다.
