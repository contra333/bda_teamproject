# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

## 10/27 중간발표 자료

- 최종 파일: [발표자료](중간발표_수정본.pptx) · [중간보고서](중간보고서_수수컴정.docx).
- [Notion 중간보고서](https://www.notion.so/4050538d38d381e285cbec8e94e6632b): 연구 질문, 성공 기준, 데이터, 결과, 남은 계획과 두 최종 파일.
- [실험로그](https://www.notion.so/4050538d38d3817cb014d68a07e4a72d): 상세 설정·수치·그림과 해석 범위.
- [발표용 그림·표](docs/W9_발표자료_연결안내.md) · [결과 확인 노트북](notebooks/w9_midterm_자동화.ipynb).

현재 발표 범위는 **Dummy와 TF-IDF + Logistic Regression·Linear SVM 기준선**이다. KOCOH v2 3,000행, 같은 맥락을 공유하는 77개 게시글 묶음의 5-fold(seed 42) 결과를 사용한다. CPU 기준선과 임계값 조정 결과는 재실행해 대조했다.

| macro-F1, 5-fold 평균 ± 표준편차 | 댓글만 | 맥락+댓글 |
|---|---|---|
| Dummy 최빈값 · 보고서 최소 기준선 | 0.427 ± 0.005 | 입력 무관 |
| Dummy 무작위 · 발표자료 비교 기준 | 0.495 ± 0.024 | 입력 무관 |
| TF-IDF + Logistic Regression | 0.647 ± 0.014 | 0.633 ± 0.033 |
| TF-IDF + Linear SVM | 0.644 ± 0.014 | 0.618 ± 0.034 |
| Logistic Regression · 내부 검증으로 임계값 조정 | 0.644 ± 0.013 | 0.662 ± 0.027 |

Logistic Regression의 맥락 조건 혐오 Recall은 임계값 조정 후 0.303에서 0.535로 올랐다. 댓글만 조건에도 같은 선택 절차를 적용하면 Recall은 0.588이며, 맥락 추가에 따른 하락은 0.053이다. `pr_auc` 열은 코드의 `average_precision_score`(AP) 값이다.

10주차에는 KcELECTRA-base-v2022와 KLUE-RoBERTa-base를 같은 분할에서 학습하고 seed 3개 반복, 예측 확률 저장, 맥락만 입력하는 조건과 Colab T4 시간을 확인한다. 11주차에는 임계값 조정, 유형별 오류 분석, LoRA 비교와 최종 표·그림을 마친다. **일정은 최종 발표자료를 따른다.**

## CPU 실험 실행

WSL Ubuntu-24.04에서 **팀프로젝트 폴더**를 열고 실행한다. 이 PC는 `~/envs/research`를 사용한다. 다른 PC는 Python 3.12 환경에 [requirements-midterm.txt](requirements-midterm.txt)를 설치한다. 데이터 출처와 다운로드 방법은 [data/README.md](data/README.md)에 있다.

아래 순서는 **기본 기준선 → 맥락만 입력하는 조건 추가 → 임계값 조정 → 결과 대조**다. `baseline_v2`는 `baseline`의 출력, `baseline_v3`는 `baseline_v2`의 출력을 읽으므로 순서대로 실행한다.

```bash
source ~/envs/research/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m src.download_data
python -m src.midterm.baseline
python -m src.midterm.baseline_v2
python -m src.midterm.baseline_v3
python -m src.midterm.validate
```

공유된 결과로 표·그림만 다시 만들 때는 `python -m src.midterm.summarize`를 실행한다. 결과 확인 노트북도 원본 데이터나 모델 학습 없이 실행할 수 있다.

PLM 학습은 위 절차에 포함하지 않는다. [plm.py](src/midterm/plm.py)와 `provided/plm_folds.csv`는 이전 탐색 코드·참고 결과로 보존하며 이번 발표의 성능 표·그림에는 사용하지 않는다. 다음 실험은 확률 저장·seed 반복·검증 절차를 보완해 실행해야 한다. 별도 모델 가중치와 torch·transformers·peft가 필요하며, 기존 탐색 실행의 모델 revision·패키지 버전은 고정되어 있지 않다.

## 파일 위치

| 위치 | 용도 |
|---|---|
| `src/midterm/` | 현재 모델 비교 코드 |
| `중간발표_수정본.pptx`, `중간보고서_수수컴정.docx` | 제출용 최종 자료 |
| `notebooks/w9_midterm_자동화.ipynb` | 저장된 결과와 그림 확인 |
| `results/midterm/provided/` | 기준선 CSV·분할표와 이전 PLM 참고 결과 |
| `results/midterm/` | 요약 표와 검증 결과·원본 해시 |
| `results/midterm/reproduced/` | 재실행 때 생성하는 로컬 비교본. GitHub에는 올리지 않음 |
| `figures/midterm/` | 발표용 PNG·PDF |
| `data/`, `docs/` | 로컬 데이터와 보고서·설명 자료 |

## 이전 기록과 팀 규약

- W5: [EDA 보고서](docs/수수컴정_W5_EDA_보고서.md) · [공유 PDF](docs/수수컴정_W5_EDA_발표안내.pdf) · [노트북](notebooks/w5_eda_자동화.ipynb). 실행은 `python -m src.eda`다. `src/preprocess.py`와 기존 `results/`, `figures/`의 W5 자료는 당시 Set 기준 고정 분할(1,770/639/591행)을 사용한다. 현재 모델 비교의 분할과 구별한다.
- [Notion 연구관리](https://app.notion.com/p/bda0538d38d3837793b3016b2d21c682) · [팀 규약](TEAM.md). 저장소를 받은 PC에서는 `git config core.hooksPath hooks`를 한 번 실행한다.
- W5 설명의 편집 기준본은 `docs/수수컴정_W5_EDA_발표안내.docx`다. PDF와 Markdown은 수정된 Word에서 갱신한다. 원본 데이터·모델 가중치·인증정보는 커밋하지 않는다.
