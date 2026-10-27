# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

KOCOH v2 3,000행을 사용한다. 같은 Context를 공유하는 Set을 77개 게시글 묶음으로 합쳐 5-fold(seed 42)로 평가하며, 임계값은 각 train 내부의 묶음 단위 4-fold OOF 예측으로 선택한다. 데이터 출처와 전처리 기준은 [데이터 설명](data/README.md)에 있다.

## 코드와 결과

안지수 님이 전달한 통합본을 기준으로 한다. 실행 출력이 저장된 노트북 두 개에서 분석 과정과 해석을 확인할 수 있다.

| 내용 | 코드·노트북 | 결과 파일 |
|---|---|---|
| EDA · 5주차 | [EDA 노트북](notebooks/project_data_eda.ipynb) · [전처리 코드](src/preprocess.py) | [결측률](figures/kocoh_missing_ratio.png) · [클래스 비율](figures/kocoh_class_ratio.png) · [분할 비교](figures/kocoh_split_compare.png) |
| 전처리·데이터 분할 · 5주차 | [전처리 코드](src/preprocess.py) | [행별 분할표](results/split_assignment.csv) · [내부 검증 구성](results/inner_val_composition.csv) |
| 기준선 · 6주차 | [학습·평가 코드](src/baseline.py) · [모델 비교 노트북](notebooks/project_mid_report.ipynb) | [성능 80행](results/baseline_folds.csv) · [유형별 결과](results/baseline_by_type.csv) |
| 댓글만 사용했을 때의 상한 진단 | [기준선 코드](src/baseline.py) | [정답 구성에 따른 진단값](results/comment_only_ceiling.csv) |
| PLM 탐색 | [PLM 코드](src/plm.py) · [모델 비교 노트북](notebooks/project_mid_report.ipynb) | [성능 60행](results/plm_folds.csv) · [모델 비교](figures/kocoh_mid_report.png) · [LoRA 비교](figures/kocoh_lora_vs_finetune.png) |

6주차 두 기준선의 macro-F1은 **Dummy 최빈값 0.427**, **TF-IDF + Logistic Regression(댓글만) 0.647**이다(5-fold 평균). `baseline_folds.csv`에는 기본 10조합과 임계값 조정 6조합의 fold별 성능이 들어 있다.

EDA 노트북의 train 2,410행 / test 590행은 첫 fold 예시다. 분할표 15,000행은 원본 3,000행 × 5개 fold의 배정을 기록한 것이다. 현재 결과는 seed 42의 5-fold이며, 여러 seed의 반복실험과 구별한다. 댓글 상한 진단은 현재 정답 구성을 이용한 값이다.

## 실행

Python 3.12 환경에서 저장소 루트를 기준으로 실행한다. 이 PC에서는 WSL Ubuntu-24.04의 `~/envs/research` 환경을 사용한다.

```bash
source ~/envs/research/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m src.download_data
python src/preprocess.py --eda
python src/preprocess.py
python src/baseline.py
```

전처리 명령은 EDA 그림 3개를, 기준선 명령은 `results/`의 CPU 결과 CSV 5개를 갱신한다. 노트북은 같은 환경의 커널로 위에서부터 실행한다. 분할만 확인하려면 `python src/preprocess.py --inner`를 사용한다.

PLM CSV는 팀원이 제공한 탐색 실행 결과다. 모델 비교 노트북은 기본적으로 저장된 PLM 결과를 읽으며, `RUN_TRAINING=False`, `CHECK_MODEL_TOKENS=False`로 되어 있다. PLM 재학습에는 별도로 torch·transformers·peft와 모델 가중치가 필요하다.

## 보고서와 연구 기록

- 9주차: [중간보고서](docs/중간보고서_수수컴정.docx) · [발표자료](docs/중간발표_수정본.pptx)
- Notion: [연구관리](https://app.notion.com/p/bda0538d38d3837793b3016b2d21c682) · [실험로그](https://www.notion.so/4050538d38d3817cb014d68a07e4a72d)
- 다음 일정: 발표자료 기준 10주차 seed 3회 반복, 11주차 결과 분석.
- 협업: [팀 규약](TEAM.md). 처음 받은 PC에서는 `git config core.hooksPath hooks`를 설정한다.
