# 게시글 맥락 추가에 따른 한국어 혐오표현 탐지 성능 비교

빅데이터분석및이론(AMA8039) 수수컴정 팀 프로젝트.

**연구 질문:** 댓글만으로는 판단하기 어려운 한국어 맥락 의존적 혐오표현에서, 게시글 맥락을 함께 활용하면 혐오표현 탐지 성능을 개선할 수 있는가?

## 먼저 읽을 자료

- [KOCOH 데이터 이해와 EDA 결과](docs/수수컴정_W5_EDA_보고서.md): 데이터 구조, 기존 논문과 팀 계획, 실제 숫자와 처리 이유.
- [공유용 PDF](docs/수수컴정_W5_EDA_발표안내.pdf): 같은 내용의 9쪽 보고서. 수정용 Word는 Notion 페이지에 첨부.
- [팀 Notion EDA 페이지](https://app.notion.com/p/5-KOCOH-EDA-3f70538d38d381d1b7c7c83e0d85889a) · [이번 실험로그](https://app.notion.com/p/W5-KOCOH-3f70538d38d38159bc23f328c0c37732)
- [KOCOH 데이터셋 기록](https://app.notion.com/p/KOCOH-v2-Context-Comment-3-000-3f70538d38d381f0b6bfc4f1828acede) · [원논문 기록](https://app.notion.com/p/KOCOH-Korean-Context-Dependent-Hate-Speech-Dataset-3f70538d38d381c39755f72fcf5d7b90)
- [팀 규약](TEAM.md): 역할, 회의, 파일과 기여 기록 방식.

## 현재 단계

KOCOH v2의 EDA와 고정 분할·전처리 준비를 마쳤다. 원본 3,000행은 모두 유지했고, 학습/검증/시험은 Set 기준 1,770/639/591행이다. 모델 학습·성능 평가는 아직 수행하지 않았다.

전통적 분류 방법과 사전학습 언어모델 각각에서 댓글만 / 맥락+댓글 조건을 비교한다는 방향은 팀 문서에 있다. 구체적인 모델·동결 표현 또는 fine-tuning·주요 지표와 성공 기준은 아직 확인할 사항이다. Notion의 KcBERT·KcELECTRA·macro-F1은 후보 메모로 구분한다.

원본·모델 가중치·인증정보는 공유 저장소에 포함하지 않는다. 원본은 아래 명령으로 고정 버전을 내려받는다.

## 재현 순서

이 PC의 수업 실행 환경은 WSL Ubuntu-24.04의 `~/envs/research`이다. 먼저 팀프로젝트 루트로 이동한 뒤 실행한다. 다른 팀원은 `requirements-eda.txt`에 기록한 실행 버전을 참고해 자신의 Python 환경을 준비한다.

```bash
~/envs/research/bin/python -m src.download_data
~/envs/research/bin/python -m src.eda
~/envs/research/bin/python -m unittest discover -s tests -v
```

- [실행 완료 노트북](notebooks/w5_eda_자동화.ipynb): 진군학 자동화 담당 파일.
- `src/preprocess.py`: 검증, Set 분할, 모델 입력 열 선택. 학습되는 변환은 후속 모델 코드에서 train에만 fit한다.
- `data/README.md`: 출처·원본 해시·다운로드 안내.
- `results/`: 재생성할 수 있는 집계표와 예외 ID. `figures/`: PNG와 PDF 그림 4쌍.

학습 전에는 Set 43/44 예외와 반복 입력의 평가 방침, tokenizer 기준 길이·잘림을 확인한다. 같은 분할을 모든 조건에서 재사용하고 test 점수를 보고 분할을 바꾸지 않는다.

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

공유 설명의 수정 원본은 `docs/수수컴정_W5_EDA_보고서.md`다. Notion에는 요약·결정과 코드/문서 링크를 남긴다. figure 제작 및 에이전트 작업 규칙은 `AGENTS.md`에 두고 공유 보고서에는 넣지 않는다.
