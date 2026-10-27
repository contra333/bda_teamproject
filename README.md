# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

## 중간발표 자료와 결과

- 최종 제출물: [발표자료](docs/중간발표_수정본.pptx) · [중간보고서](docs/중간보고서_수수컴정.docx)
- 분석 노트북: [데이터·EDA](notebooks/project_data_eda.ipynb) · [모델 비교](notebooks/project_mid_report.ipynb)
- Notion: [중간보고서](https://www.notion.so/4050538d38d381e285cbec8e94e6632b) · [팀 실험로그](https://www.notion.so/4050538d38d3817cb014d68a07e4a72d)

현재 발표 범위는 **Dummy와 TF-IDF + Logistic Regression·Linear SVM 기준선**이다. KOCOH v2 3,000행, 같은 맥락을 공유하는 77개 게시글 묶음의 5-fold(seed 42) 결과를 사용한다. 임계값은 각 train 안의 묶음 단위 4-fold OOF 예측으로 선택한다.

| macro-F1, 5-fold 평균 ± 표준편차 | 댓글만 | 맥락+댓글 |
|---|---|---|
| Dummy 최빈값 · 보고서 최소 기준선 | 0.427 ± 0.005 | 입력 무관 |
| Dummy 무작위 · 발표자료 비교 기준 | 0.495 ± 0.024 | 입력 무관 |
| TF-IDF + Logistic Regression | 0.647 ± 0.014 | 0.633 ± 0.033 |
| TF-IDF + Linear SVM | 0.644 ± 0.014 | 0.618 ± 0.034 |
| Logistic Regression · 내부 검증으로 임계값 조정 | 0.644 ± 0.013 | 0.662 ± 0.027 |

Logistic Regression의 맥락 조건 혐오 Recall은 임계값 조정 후 0.303에서 0.535로 올랐다. 댓글만 조건에도 같은 선택 절차를 적용하면 Recall은 0.588이며, 맥락 추가에 따른 하락은 0.053이다. `pr_auc` 열은 `average_precision_score`(AP) 값이다.

## 실행

팀 통합본의 `preprocess.py`가 분할과 전처리를, `baseline.py`가 기본 기준선·Context-only·유형별 분석·임계값 조정을 한 번에 실행한다. `baseline_v2.py`, `baseline_v3.py`를 따로 실행할 필요가 없다.

WSL Ubuntu-24.04의 **팀프로젝트 폴더**에서 실행한다. 다른 PC에서는 Python 3.12 환경에 [requirements-midterm.txt](requirements-midterm.txt)를 설치한다. 데이터 출처·고정 버전은 [data/README.md](data/README.md)를 따른다.

```bash
source ~/envs/research/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m src.download_data
python src/baseline.py
python -m unittest discover -s tests -v
```

기준선 실행은 `results/`의 CPU 결과 CSV 다섯 개를 갱신한다. 기본 10조합 × 5-fold와 임계값 조정 6조합 × 5-fold로 **기준선 결과는 80행**이다. 제공된 통합본의 CPU 결과·분할표와 재실행 결과가 허용 오차 0.000001 이내에서 일치함을 확인했다.

노트북은 같은 Python 환경을 커널로 선택해 위에서부터 실행한다. 데이터·EDA 노트북은 데이터와 분할을 확인하고, 모델 비교 노트북은 CPU 기본 기준선을 재실행한 뒤 저장된 결과로 표·그림을 만든다. 두 노트북의 실행 출력도 저장되어 있다.

`plm.py`와 `results/plm_folds.csv`는 팀 통합본에 포함된 PLM 코드와 **팀원이 제공한 이전 탐색 결과**다. 이번 검증에서는 PLM을 재학습하지 않았다. 노트북 첫 설정 셀의 `RUN_TRAINING=False`, `CHECK_MODEL_TOKENS=False`가 기본이며, 모델 다운로드·토큰 측정·PLM 학습은 선택 실행이다. 학습에는 별도의 모델 가중치와 torch·transformers·peft가 필요하다. 다음 실험 전에 모델 revision·패키지 버전, 예측 확률 저장과 반복 조건을 확정한다.

## 파일 위치

| 위치 | 용도 |
|---|---|
| `src/preprocess.py` | EDA, 게시글 묶음 분할, 전처리, 내부 4-fold |
| `src/baseline.py` | CPU 기준선과 임계값·유형별 비교 |
| `src/plm.py` | PLM 동결·미세조정·LoRA 실험 |
| `src/download_data.py` | 고정 버전 데이터 다운로드·해시 확인 |
| `notebooks/` | 팀 통합본의 EDA·모델 비교 노트북 두 개 |
| `results/` | 기준선·유형별 결과, 댓글별 진단값, 내부 검증 구성, 분할표, PLM 결과 CSV 여섯 개 |
| `figures/` | 노트북과 문서에 사용하는 그림 |
| `docs/` | 발표자료·중간보고서와 기존 수업 문서 |
| `data/` | 로컬 원본 데이터와 출처 안내 |

`comment_only_ceiling.csv`는 현재 데이터의 댓글별 정답 구성을 이용한 진단값이다. 학습된 모델의 실측 성능이나 다른 데이터에서도 보장되는 상한으로 해석하지 않는다. 과거 코드와 결과는 Git 이력에서 확인한다.

## 다음 실험과 팀 규약

발표자료 기준으로 **10주차에는 PLM 학습·seed 3개 반복**, 예측 확률 저장, Context-only와 Colab T4 시간을 확인한다. **11주차에는 반복 결과·임계값·유형별 오류 분석**, LoRA 비교와 최종 표·그림을 정리한다.

[팀 규약](TEAM.md) · [Notion 연구관리](https://app.notion.com/p/bda0538d38d3837793b3016b2d21c682). 저장소를 받은 PC에서는 `git config core.hooksPath hooks`를 한 번 실행한다. 원본 데이터·모델 가중치·인증정보는 커밋하지 않는다. 실험 일지는 Notion에 기록하며 `baseline.py`의 마지막 콘솔 출력은 Notion 자동 전송이 아니다.
