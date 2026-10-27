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

## 과제별 코드와 결과 위치

| 과제·내용 | 코드·실행 설명 | 확인할 결과 |
|---|---|---|
| 5주차 EDA | [실행 출력이 있는 EDA 노트북](notebooks/project_data_eda.ipynb) | [결측률](figures/kocoh_missing_ratio.png) · [클래스 비율](figures/kocoh_class_ratio.png) · [분할 비교](figures/kocoh_split_compare.png) |
| 5주차 전처리·데이터 분할 | [전처리 코드](src/preprocess.py) · [데이터·분할 설명](data/README.md) | [행별 분할표](results/split_assignment.csv) · [내부 검증 구성](results/inner_val_composition.csv) |
| 6주차 기준선 | [학습·평가 코드](src/baseline.py) · [모델 비교 노트북의 Task 1](notebooks/project_mid_report.ipynb) | [모델·조건·fold별 성능 80행](results/baseline_folds.csv) · [유형별 결과](results/baseline_by_type.csv) |
| 9주차 보고서·발표 | [중간보고서](docs/중간보고서_수수컴정.docx) · [발표자료](docs/중간발표_수정본.pptx) | 제출용 PDF는 별도 준비 필요 |
| PLM 탐색 결과 | [PLM 코드](src/plm.py) | [팀 제공 참고 결과](results/plm_folds.csv). 현재 중간발표 기준선 결과와 구별 |
| 데이터 확보 | [고정 버전 다운로드](src/download_data.py) · [출처](data/README.md) | 원본은 로컬 `data/`에 저장하며 Git에는 올리지 않음 |

EDA 노트북의 train 2,410행 / test 590행은 **첫 번째 fold의 예시**다. 실제 기준선은 바깥 5-fold 전체에서 평가하고, 임계값은 각 train 내부의 묶음 단위 4-fold로 선택한다. 분할표의 15,000행은 원본 3,000행에 대해 바깥 fold 다섯 개의 역할을 기록한 것이며, 데이터가 늘어난 것이 아니다. 현재 결과는 **seed 42의 5-fold**이며 서로 다른 seed 다섯 개의 반복실험은 아니다.

기존 `docs/수수컴정_W5_EDA_발표안내.pdf`와 `docs/수수컴정_W5_EDA_보고서.md`는 이전 Set 단위 고정 분할을 설명한다. 최신 제출본은 위 EDA·분할 결과와 경로로 갱신해야 한다. 이번 일괄 제출은 3·4·5·6·9주차의 여섯 제출물을 PDF로 준비하고, 5주차 PDF에 EDA 그림과 위 전처리 코드 링크를 포함한다.

`comment_only_ceiling.csv`는 현재 데이터의 댓글별 정답 구성을 이용한 진단값이다. 학습된 모델의 실측 성능이나 다른 데이터에서도 보장되는 상한으로 해석하지 않는다. 과거 코드와 결과는 Git 이력에서 확인한다.

## 다음 실험과 팀 규약

발표자료 기준으로 **10주차에는 PLM 학습·seed 3개 반복**, 예측 확률 저장, Context-only와 Colab T4 시간을 확인한다. **11주차에는 반복 결과·임계값·유형별 오류 분석**, LoRA 비교와 최종 표·그림을 정리한다.

[팀 규약](TEAM.md) · [Notion 연구관리](https://app.notion.com/p/bda0538d38d3837793b3016b2d21c682). 저장소를 받은 PC에서는 `git config core.hooksPath hooks`를 한 번 실행한다. 원본 데이터·모델 가중치·인증정보는 커밋하지 않는다. 실험 일지는 Notion에 기록하며 `baseline.py`의 마지막 콘솔 출력은 Notion 자동 전송이 아니다.
