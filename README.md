# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

## 먼저 읽을 자료

- [KOCOH 데이터 이해와 EDA 결과](docs/수수컴정_W5_EDA_보고서.md): 데이터 구조, 기존 논문과 팀 계획, 실제 숫자와 처리 이유.
- [공유용 PDF](docs/수수컴정_W5_EDA_발표안내.pdf): 같은 내용의 9쪽 보고서. 수정용 Word는 Notion 페이지에 첨부.
- [팀 Notion EDA 페이지](https://app.notion.com/p/5-KOCOH-EDA-3f70538d38d381d1b7c7c83e0d85889a) · [이번 실험로그](https://app.notion.com/p/W5-KOCOH-3f70538d38d38159bc23f328c0c37732)
- [KOCOH 데이터셋 기록](https://app.notion.com/p/KOCOH-v2-Context-Comment-3-000-3f70538d38d381f0b6bfc4f1828acede) · [원논문 기록](https://app.notion.com/p/KOCOH-Korean-Context-Dependent-Hate-Speech-Dataset-3f70538d38d381c39755f72fcf5d7b90)
- [팀 규약](TEAM.md): 역할, 회의, 파일과 기여 기록 방식.

[중간보고서](https://www.notion.so/4050538d38d381e285cbec8e94e6632b) · [실험 결과·첨부](https://www.notion.so/4050538d38d3817cb014d68a07e4a72d)

## 현재 단계 — 10/27 중간발표

KOCOH v2 **3,000행, 77개 post_group**으로 같은 게시글 맥락이 train/test에 겹치지 않는 5-fold(seed 42) 비교를 마쳤다. TF-IDF 임계값은 각 바깥 train 안의 묶음 4-fold OOF로 선택하고 train 전체로 다시 학습한다.

| macro-F1 (5개 fold 평균 ± 표본 SD) | 댓글만 | 맥락+댓글 |
|---|---:|---:|
| Dummy most_frequent | 0.427 ± 0.005 | 입력 무관 |
| TF-IDF + LogReg, 기본 임계값 | 0.647 ± 0.014 | 0.633 ± 0.033 |
| TF-IDF + LogReg, 내부 OOF 임계값 | 0.644 ± 0.013 | 0.662 ± 0.027 |
| KcELECTRA fine-tuning | 0.672 | 0.809 |
| KLUE-RoBERTa fine-tuning | 0.637 | 0.778 |

PLM은 팀원이 제공한 단일 seed 결과이며 이번 통합에서 재학습하지 않았다. CPU 기준선은 WSL에서 재실행해 제공 CSV와 대조했다. fold SD는 seed 반복의 변동이나 신뢰구간이 아니다. PLM Context-only·seed 반복·임계값 조정·T4 시간 확인은 10·11주차 계획이다.

- **[발표자료 연결 안내](docs/W9_발표자료_연결안내.md)**: 그림·표·발표 해석 주의점.
- **[실행 출력이 있는 노트북](notebooks/w9_midterm_자동화.ipynb)**: 모델 재학습 없이 결과와 그림을 확인한다.
- [입력 조건 비교](figures/midterm/01_context_comparison.png) · [임계값 조정 전후](figures/midterm/02_threshold_comparison.png).
- [검증 결과](results/midterm/validation.json) · [제공 원본 해시](results/midterm/source_manifest.json).

원실험·PLM: 소지연 / 확장·임계값 비교: 안지수 / 통합·CPU 재현·게시: 진군학. 원본 Word 중간보고서는 Notion에 첨부한다.

## 중간발표 재현 순서

이 PC의 실행 환경은 WSL Ubuntu-24.04 `~/envs/research`이다. **팀프로젝트 루트**에서 실행한다. 다른 PC는 Python 3.12 환경에 `requirements-midterm.txt`를 설치한다. 모델 비교와 기존 EDA는 별도 모듈이므로 서로의 결과를 덮어쓰지 않는다.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
~/envs/research/bin/python -m src.download_data
~/envs/research/bin/python -m src.midterm.baseline
~/envs/research/bin/python -m src.midterm.baseline_v2
~/envs/research/bin/python -m src.midterm.baseline_v3
~/envs/research/bin/python -m src.midterm.validate
~/envs/research/bin/python -m src.midterm.summarize
~/envs/research/bin/python -m unittest discover -s tests -v
```

- `src/midterm/`: 팀 제공 코드의 import·경로를 통합한 모듈. 통계·모델 설정은 유지했다.
- `results/midterm/provided/`: 두 ZIP에서 받은 원래 CSV. 변경하지 않는다.
- `results/midterm/reproduced/`: CPU 재실행 결과. 재실행 시 이 폴더의 산출물만 갱신한다.
- `results/midterm/`: 계산한 요약, 재현 환경, 검증 결과. `figures/midterm/`: PNG 300 dpi와 벡터 PDF.
- 기준선 실행은 모델 가중치·Notion 인증이 필요 없다. 제공 결과 노트북은 원본 데이터 없이도 읽고 실행할 수 있다.

PLM 코드는 `src/midterm/plm.py`에 보존했다. 학습에는 torch·transformers·peft와 별도 모델 가중치가 필요하다. 팀 제공 결과의 모델 revision과 패키지 버전이 고정되어 있지 않아 완전한 PLM 재현을 보장하지 않는다. 원본 코드는 로컬 `models/KcELECTRA-base-v2022`(beomi/KcELECTRA-base-v2022), `models/klue-roberta-base`(klue/roberta-base)를 읽는다. PLM 재학습은 이번 CPU 재현 절차에 포함하지 않는다.

## W5 EDA 재현

W5의 Set 기준 고정 분할(1,770/639/591행), 코드·노트북·보고서는 당시 기록으로 보존한다. 현재 중간발표의 post_group 5-fold와 다르므로 모델 비교에 혼용하지 않는다.

```bash
~/envs/research/bin/python -m src.eda
```

- [W5 실행 노트북](notebooks/w5_eda_자동화.ipynb), `src/preprocess.py`, 기존 `results/`와 `figures/`는 W5용이다.
- 데이터 출처·고정 버전·해시는 [data/README.md](data/README.md)에 있다. 원본·모델 가중치·인증정보는 커밋하지 않는다.

## 커밋 전 검사 설치

저장소를 받은 **각 팀원 PC에서 한 번** 실행한다. 이 PC에는 설치와 동작 검증을 마쳤다.

```bash
git config core.hooksPath hooks
```

`hooks/pre-commit`은 스테이징된 파일을 검사하여 `data/` 원본, 10 MiB 초과 파일, Notion 토큰 문자열과 `.env`를 차단한다. 팀 규약상 다운로드 설명인 `data/README.md`는 허용한다. Windows Git과 WSL Git에서 사용할 수 있는 Bash 스크립트다. 원본·큰 파일·가짜 토큰 차단 및 정상 커밋 통과를 임시 저장소에서 검증했다.

## 자료와 기록 위치

| 위치 | 역할 |
|---|---|
| `data/` | 로컬 원본과 분할 파일. 원본은 커밋하지 않음 |
| `src/`, `notebooks/` | 재실행할 코드. 노트북은 1인 1파일 |
| `results/`, `figures/` | 집계 결과와 그림 |
| `docs/` | 읽는 설명 Markdown과 공유 PDF |
| Notion | 연구 맥락, 결과 요약, 판단, 담당과 다음 행동 |

공유 설명의 편집 기준본은 `docs/수수컴정_W5_EDA_발표안내.docx`다. PDF와 `docs/수수컴정_W5_EDA_보고서.md`는 수정된 Word에서 다시 생성한다. 기존 Markdown으로 Word를 덮어쓰지 않는다. Notion에는 요약·결정과 코드/문서 링크를 남긴다. figure 제작 및 에이전트 작업 규칙은 `AGENTS.md`에 두고 공유 보고서에는 넣지 않는다.
