"""KOCOH v.2 데이터 전처리 모듈.

이 파일이 하는 일 (한 줄 요약)
    "데이터를 먼저 train/test로 나누고, 그 다음에 전처리한다. 전처리 기준값은 train에서만 구한다."

왜 이 순서인가?
    전처리(결측 채우기, 표준화 등)에 쓰는 평균·중앙값 같은 값을 전체 데이터로 구하면,
    시험용(test) 데이터의 정보가 학습에 몰래 섞인다. 그러면 평가 점수가 실제보다 좋게 나온다.
    그래서 "나누기 → train으로 기준값 계산 → 같은 기준값을 train/test에 적용" 순서를 지킨다.

용어
    Context      : 게시글 내용을 요약한 문장 (시험 문제의 '지문' 같은 것)
    Comment      : 그 게시글에 달린 댓글 (지문에 딸린 '문제')
    Hate speech  : 예측할 정답. 1 = 혐오 표현, 0 = 아님
    Set          : 같은 원글에서 나온 Context 문장들의 묶음 번호 (1~101)
    post_group   : 이 파일에서 만드는 '게시글 묶음' 번호. 같은 Context를 공유하는 Set끼리 합친 것

분할 방식 (method)
    "group" (기본): 게시글 묶음(post_group) 단위로 통째로 나눈다.
                    → test에 나오는 지문(Context)은 train에 한 번도 나오지 않는다.
    "row"   (이전): 댓글 한 줄 단위로 섞어서 나눈다.
                    → 같은 지문이 train/test 양쪽에 들어간다(test의 약 97%). 비교용으로만 남겨 둠.

주요 함수
    load_and_split()  : train/test 한 번 나누고 전처리까지 해서 돌려준다.
    cv_splits()       : 5개 fold로 돌아가며 나눈다(교차검증). fold마다 전처리를 새로 한다.
    get_xy()          : 모델에 넣을 입력(X)과 정답(y)만 꺼낸다. 정답을 누설하는 열은 절대 넘기지 않는다.

터미널에서 실행
    python -m src.midterm.preprocess         # 분할 결과 + 5-fold 요약 + 두 방식 비교 표 출력, 비교 그림 저장
    python -m src.midterm.preprocess --eda   # 기초 EDA 표 출력 + figures/ 에 그림 2장 저장(기존 파일 덮어씀)

다른 코드에서 사용하는 예
    from .preprocess import load_and_split, get_xy
    train, test, scaler = load_and_split()
    X_train, y_train = get_xy(train, use_context=False)   # Comment-only 실험
    X_train, y_train = get_xy(train, use_context=True)    # Context + Comment 실험
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ───────────────────────────── 기본 설정 ─────────────────────────────
SEED = 42  # 랜덤 시드. 같은 값이면 언제 실행해도 똑같이 나뉜다.
N_SPLITS = 5  # 5개로 나누면 한 조각이 전체의 약 20% → test 20%
ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "midterm" / "reproduced"
DATA_PATH = ROOT / "data" / "KOCOH_v.2.csv"  # 규칙: 데이터는 data/ 안에서만 읽는다

TARGET = "Hate speech"  # 예측할 정답 열
TEXT_COLS = ["Context", "Comment"]  # 모델이 읽을 텍스트 열

# 정답을 '몰래 알려주는' 열 → 전처리 단계에서 아예 지운다.
#   - Type: 값만 보면 정답이 100% 정해진다 (1 → 전부 혐오, 2·3 → 전부 비혐오)
#   - Date·Source·Link·Title: Type=2 행(전부 비혐오)에서만 비어 있다.
#     "비어 있으면 비혐오"라는 힌트가 되므로, 행이 아니라 열을 지운다.
LEAKY_COLS = ["Type", "Date", "Source", "Link", "Title"]

# 지우지는 않지만 모델 입력으로 쓰면 안 되는 열 (get_xy 가 절대 넘기지 않음)
#   - 보조 라벨: 혐오의 종류 등을 사람이 표시한 '또 다른 정답'이라 입력에 넣으면 누설
#   - 식별·분할용 번호: 행 번호나 묶음 번호는 내용과 무관한 값
AUX_LABEL_COLS = ["Counter speech", "Gender", "Disability", "Race/Nationality",
                  "Region (Korea)", "Age", "Profanity", "Ikiyano-style"]
ID_COLS = ["Index", "Set", "post_group"]

NUM_COLS = ["context_len", "comment_len"]  # 텍스트에서 새로 만드는 숫자 특성 (글자 수)

# 그림 색 (배경, 막대, 글자, 격자선)
SURFACE, BAR, INK, GRID = "#fcfcfb", "#2a78d6", "#3d3d3a", "#e4e3dd"


# ───────────────────────────── 1. 읽기 ─────────────────────────────
def make_post_groups(df):
    """'게시글 묶음' 번호(post_group)를 만든다.

    왜 Set을 그대로 쓰지 않나?
        Set 하나에는 같은 원글을 조금씩 바꾼 Context가 여러 개 들어 있다.
        그런데 일부 Context 문장(18개)은 서로 다른 Set에 '똑같이' 나온다.
        예) "광주광역시의 맛있는 음식 다섯 가지를 소개했다." → Set 14, 16, 19, 25
        Set 단위로만 나누면 이런 문장이 train과 test 양쪽에 들어갈 수 있다.

    어떻게 묶나? (union-find)
        같은 Context를 하나라도 공유하는 Set끼리 같은 묶음으로 합친다.
        A와 B가 공유하고 B와 C가 공유하면 A·B·C가 모두 한 묶음이 된다.
        결과: 101개 Set → 77개 post_group

    반환: 각 행의 post_group 번호 (pandas Series)
    """
    # parent[s] = s가 속한 묶음의 대표 Set. 처음에는 각자 자기 자신이 대표.
    parent = {s: s for s in df["Set"].unique()}

    def find(x):
        """x가 속한 묶음의 최종 대표를 찾는다 (대표의 대표를 따라 올라감)."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 다음 검색이 빨라지도록 경로를 줄여 둔다
            x = parent[x]
        return x

    # Context 문장마다, 그 문장이 나온 Set들을 모두 첫 번째 Set의 묶음으로 합친다.
    for sets in df.groupby("Context")["Set"].unique():
        for s in sets[1:]:
            parent[find(s)] = find(sets[0])
    return df["Set"].map(find).rename("post_group")


def _read_data(path):
    """CSV를 읽고 post_group 열을 붙여 돌려준다. data/ 밖의 파일은 거부한다."""
    path = Path(path).resolve()
    if (ROOT / "data") not in path.parents:
        raise ValueError(f"데이터는 data/ 안에서만 읽습니다: {path}")
    df = pd.read_csv(path)
    print(f"원본: {len(df)}행 × {df.shape[1]}열")
    return df.assign(post_group=make_post_groups(df))


# ───────────────────────────── 2. 나누기 ─────────────────────────────
def _group_kfold(df, seed=SEED, n_splits=N_SPLITS):
    """게시글 묶음 단위 K-fold 분할기.

    StratifiedGroupKFold 가 하는 일
        - Group     : 같은 post_group의 행은 반드시 같은 쪽(train 또는 test)으로 간다.
        - Stratified: 그 제약 안에서 각 fold의 혐오 비율이 전체(약 25%)와 비슷해지도록 묶음을 배치한다.
                      묶음째 움직여야 해서 비율이 완벽히 같아지지는 않는다.
        - shuffle + random_state: 묶음 배치 순서를 seed로 고정된 무작위로 섞는다.

    반환: (train 위치, test 위치) 쌍을 n_splits번 내놓는 반복자
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return sgkf.split(df, df[TARGET], df["post_group"])


def split_train_test(df, method="group", test_size=0.2, seed=SEED):
    """전처리하기 전의 원본 df를 train/test로 한 번 나눈다. 반환: (train, test)

    method="group": 게시글 묶음 단위. 5-fold의 '첫 번째' fold를 test로 쓴다.
                    (점수가 좋은 fold를 골라 쓰는 일이 없도록 항상 첫 번째로 고정)
    method="row"  : 이전 방식. 행 단위로 섞어서 정확히 test_size 만큼 떼어낸다. 비교용.
    """
    if method == "row":
        return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    if method == "group":
        if "post_group" not in df:
            df = df.assign(post_group=make_post_groups(df))
        train_idx, test_idx = next(_group_kfold(df, seed=seed, n_splits=round(1 / test_size)))
        return df.iloc[train_idx], df.iloc[test_idx]
    raise ValueError(f"method 는 'group' 또는 'row': {method}")


# ───────────────────────────── 3. 전처리 ─────────────────────────────
def _drop_missing(df, name):
    """Context·Comment·정답 중 하나라도 비어 있는 행만 지우고, 몇 개 지웠는지 출력한다.

    텍스트나 정답이 없으면 학습·평가에 쓸 수 없으므로 지운다.
    (현재 데이터에는 이런 행이 0개다. 그래도 규칙상 개수와 이유를 항상 출력한다.)
    """
    required = TEXT_COLS + [TARGET]
    mask = df[required].isna().any(axis=1)
    print(f"  [{name}] 결측 행 제거: {int(mask.sum())}개 (이유: {', '.join(required)} 중 하나라도 결측)")
    return df.loc[~mask].copy()


def _add_num_features(df):
    """텍스트 길이(글자 수)를 숫자 특성으로 추가한다."""
    df["context_len"] = df["Context"].str.len()
    df["comment_len"] = df["Comment"].str.len()
    return df


def _preprocess(train, test, verbose=True):
    """이미 나뉜 train/test에 전처리를 적용한다. 기준값은 모두 train에서만 구한다.

    반환: (train, test, scaler) — scaler는 train으로 맞춘 표준화 도구(평균·표준편차를 기억)
    """
    # (1) 누설 열 지우기 — 열만 지우고 행은 그대로 둔다.
    if verbose:
        print(f"  누설 열 제외(행 삭제 아님): {LEAKY_COLS}")
    train = train.drop(columns=LEAKY_COLS)
    test = test.drop(columns=LEAKY_COLS)

    # (2) 결측 처리
    #     - 텍스트/정답이 빈 행은 지운다 (개수 출력)
    #     - 숫자 특성이 비면 'train의 중앙값'으로 채운다 (test의 값을 보지 않기 위해)
    train = _add_num_features(_drop_missing(train, "train"))
    test = _add_num_features(_drop_missing(test, "test"))
    medians = train[NUM_COLS].median()
    train[NUM_COLS] = train[NUM_COLS].fillna(medians)
    test[NUM_COLS] = test[NUM_COLS].fillna(medians)

    # (3) 표준화: (값 - 평균) / 표준편차 로 바꿔 숫자 크기를 맞춘다.
    #     fit(평균·표준편차 계산)은 train으로만 하고, test에는 그 값을 그대로 적용(transform)만 한다.
    #     그래서 test의 표준화 후 평균이 정확히 0이 아닌 것이 정상이다.
    scaler = StandardScaler().fit(train[NUM_COLS])
    train[NUM_COLS] = scaler.transform(train[NUM_COLS])
    test[NUM_COLS] = scaler.transform(test[NUM_COLS])

    return train.reset_index(drop=True), test.reset_index(drop=True), scaler


# ───────────────────────────── 4. 외부에서 쓰는 함수 ─────────────────────────────
def load_and_split(path=DATA_PATH, test_size=0.2, seed=SEED, method="group"):
    """데이터를 읽어 train/test로 '한 번' 나누고 전처리까지 한다.

    순서: 읽기 → 나누기(먼저!) → 누설 열 제외 → 결측 처리 → 표준화(train으로만 fit)

    method="group"(기본): 게시글 묶음 단위. test는 약 20% (묶음 크기 때문에 정확히 20%는 아님)
    method="row"        : 이전 행 단위 방식. 비교용으로만 사용.
    반환: (train_df, test_df, scaler)

    주의: 반환된 df에는 보조 라벨·ID 열도 남아 있다. 모델 입력은 반드시 get_xy()로 꺼낼 것.
    """
    print(f"seed = {seed}")
    df = _read_data(path)
    train, test = split_train_test(df, method=method, test_size=test_size, seed=seed)
    print(f"  분할 방식: {method} → train {len(train)}행 / test {len(test)}행")
    return _preprocess(train, test)


def cv_splits(path=DATA_PATH, n_splits=N_SPLITS, seed=SEED):
    """교차검증용: 게시글 묶음 단위로 n_splits개 fold를 차례로 돌려준다.

    왜 필요한가?
        한 번만 나누면 test에 게시글 묶음이 7개뿐이라, 어떤 게시글이 test에 걸렸느냐에 따라
        점수가 크게 흔들린다. fold 5개를 모두 test로 한 번씩 써 보고 평균·표준편차를 보고하면
        훨씬 믿을 만하다. 모든 행이 정확히 한 번씩 test가 된다.

    전처리는 fold마다 새로 한다 (그 fold의 train으로만 기준값을 구함).
    fold 0은 load_and_split()의 test와 같은 분할이다.

    사용 예:
        for fold, train, test, scaler in cv_splits():
            X_tr, y_tr = get_xy(train, use_context=True)
            X_te, y_te = get_xy(test, use_context=True)
            ...  # 학습·평가 후 점수를 리스트에 모아 평균·표준편차 계산

    내놓는 값: (fold 번호, train_df, test_df, scaler) 를 n_splits번
    """
    print(f"seed = {seed}")
    df = _read_data(path)
    rows = []
    folds = []
    for fold, (tr_idx, te_idx) in enumerate(_group_kfold(df, seed=seed, n_splits=n_splits)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        # 묶음이 train/test에 겹치지 않는지 매 fold 확인 (겹치면 즉시 멈춤)
        assert not set(tr["post_group"]) & set(te["post_group"]), "post_group이 train/test에 겹침"
        rows.append({
            "fold": fold,
            "train 행": len(tr),
            "test 행": len(te),
            "test 비율(%)": round(len(te) / len(df) * 100, 3),
            "train 혐오(%)": round(tr[TARGET].mean() * 100, 3),
            "test 혐오(%)": round(te[TARGET].mean() * 100, 3),
            "test 묶음 수": te["post_group"].nunique(),
        })
        folds.append((fold, tr, te))
    # 시작 전에 fold별 크기·혐오 비율을 한눈에 보여준다
    print("\n[fold별 크기와 혐오 비율]")
    print(pd.DataFrame(rows).set_index("fold").to_string())

    for fold, tr, te in folds:
        print(f"\n[fold {fold}] 전처리")
        train, test, scaler = _preprocess(tr, te, verbose=(fold == 0))
        yield fold, train, test, scaler


def get_xy(df, use_context=True, use_length=False):
    """전처리된 df에서 모델 입력(X)과 정답(y)만 꺼낸다.

    '허용 목록' 방식: 아래에 적힌 열만 X에 들어간다. 나머지(보조 라벨, Index·Set·post_group 등)는
    실수로라도 모델에 들어가지 않는다.

    use_context=False → X 열: Comment                  (Comment-only 실험)
    use_context=True  → X 열: Context, Comment         (Context + Comment 실험)
    use_length=True   → 위에 더해 표준화된 글자 수(context_len 또는 comment_len)도 넣는다.
                        Comment-only일 때는 comment_len만 넣는다 (맥락 정보가 섞이지 않도록).

    반환: (X: DataFrame, y: numpy 배열 0/1)
    """
    cols = ["Context", "Comment"] if use_context else ["Comment"]
    if use_length:
        cols += ["context_len", "comment_len"] if use_context else ["comment_len"]

    # 혹시 금지 열이 섞였다면 바로 멈춘다 (위 목록을 나중에 고치다 실수하는 경우 대비)
    banned = set(cols) & set(LEAKY_COLS + AUX_LABEL_COLS + ID_COLS + [TARGET])
    assert not banned, f"입력에 쓰면 안 되는 열: {banned}"

    return df[cols].copy(), df[TARGET].to_numpy()


# ───────────────────────────── 5. EDA·비교 (그림) ─────────────────────────────
def _style(ax):
    """그림 공통 꾸미기: 배경색, 위·오른쪽 테두리 제거, 눈금 글자색."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def eda(path=DATA_PATH):
    """크기·열별 결측 비율·수치형 범위·예측 대상 비율 표 출력 + 그림 2장 저장(덮어씀)."""
    df = pd.read_csv(path)
    pd.set_option("display.width", 200)

    print("\n[1] 크기")
    print(pd.DataFrame({"행": [df.shape[0]], "열": [df.shape[1]]}).to_string(index=False))

    missing = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "결측 개수": df.isna().sum(),
        "결측 비율(%)": (df.isna().mean() * 100).round(3),
    })
    print("\n[2] 열별 결측 비율 (결측 행은 제거하지 않음)")
    print(missing.to_string())

    num = df.select_dtypes(include="number")
    num_range = pd.DataFrame({
        "min": num.min(),
        "max": num.max(),
        "mean": num.mean().round(3),
        "고유값 수": num.nunique(),
    })
    print("\n[3] 수치형 열 범위")
    print(num_range.to_string())

    counts = df[TARGET].value_counts(dropna=False).sort_index()
    target_tbl = pd.DataFrame({
        "개수": counts,
        "비율(%)": (counts / counts.sum() * 100).round(3),
    })
    target_tbl.index.name = f"{TARGET} (0=비혐오, 1=혐오)"
    print(f"\n[4] 예측 대상 '{TARGET}' 비율")
    print(target_tbl.to_string())

    # 그림 1: 열별 결측 비율 (가로 막대)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    _style(ax)
    miss_pct = missing["결측 비율(%)"][::-1]
    bars = ax.barh(miss_pct.index, miss_pct.values, height=0.7, color=BAR, zorder=2)
    ax.bar_label(bars, labels=[f"{v:.1f}%" if v > 0 else "0%" for v in miss_pct.values],
                 padding=3, fontsize=8, color=INK)
    ax.set_xlabel("Missing ratio (%)")
    ax.set_ylabel("Column")
    ax.set_xlim(0, max(miss_pct.max() * 1.2, 1))
    ax.set_title(f"KOCOH v.2: missing ratio by column (n = {len(df):,})")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    out1 = ROOT / "figures" / "kocoh_missing_ratio.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")

    # 그림 2: 예측 대상 클래스 비율 (세로 막대)
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=SURFACE)
    _style(ax)
    pct = target_tbl["비율(%)"]
    labels = {0: "0 (non-hate)", 1: "1 (hate)"}
    bars = ax.bar([labels.get(k, str(k)) for k in pct.index], pct.values, width=0.6, color=BAR, zorder=2)
    ax.bar_label(bars, labels=[f"{p:.1f}% (n={c:,})" for p, c in zip(pct.values, target_tbl["개수"])],
                 padding=3, fontsize=9, color=INK)
    ax.set_xlabel(f"{TARGET} label (class)")
    ax.set_ylabel("Share of comments (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"KOCOH v.2: '{TARGET}' class ratio (n = {len(df):,})")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    out2 = ROOT / "figures" / "kocoh_class_ratio.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")

    print(f"\n저장(기존 파일 덮어씀): {out1.relative_to(ROOT)}, {out2.relative_to(ROOT)}")


def compare_splits(df=None, test_size=0.2, seed=SEED):
    """이전 방식(행 단위)과 변경 방식(게시글 묶음 단위)을 같은 기준으로 비교한 표를 만든다.

    핵심 지표는 'train에서 본 Context인 test 행(%)':
    시험(test)에 나온 지문이 공부(train) 때 이미 본 지문인 비율. 0%여야 공정한 평가다.
    """
    if df is None:
        df = pd.read_csv(DATA_PATH)
    df = df.assign(post_group=make_post_groups(df))
    rows = {}
    for method, label in [("row", "이전: 행 단위"), ("group", "변경: 게시글 묶음 단위")]:
        tr, te = split_train_test(df, method=method, test_size=test_size, seed=seed)
        ctx_seen = te["Context"].isin(set(tr["Context"]))  # test 각 행의 지문이 train에도 있나?
        rows[label] = {
            "train 행": len(tr),
            "test 행": len(te),
            "test 비율(%)": round(len(te) / len(df) * 100, 3),
            "train 혐오(%)": round(tr[TARGET].mean() * 100, 3),
            "test 혐오(%)": round(te[TARGET].mean() * 100, 3),
            "양쪽 공통 Set 수": len(set(tr["Set"]) & set(te["Set"])),
            "양쪽 공통 Context 수": len(set(tr["Context"]) & set(te["Context"])),
            "train에서 본 Context인 test 행": int(ctx_seen.sum()),
            "train에서 본 Context인 test 행(%)": round(ctx_seen.mean() * 100, 3),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_split_compare(cmp, out=ROOT / "figures" / "kocoh_split_compare.png"):
    """비교 그림: (a) train/test 혐오 비율, (b) test 지문이 train에도 있는 비율."""
    methods = list(cmp.index)
    names = ["Before: row split", "After: post-group split"]
    x = np.arange(len(methods))
    width = 0.36
    test_color = "#eb6834"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

    # (a) 방식별 train/test 혐오 비율 (막대 2개씩)
    _style(ax1)
    for i, (col, label, color) in enumerate([("train 혐오(%)", "train", BAR), ("test 혐오(%)", "test", test_color)]):
        vals = cmp[col].astype(float).values
        bars = ax1.bar(x + (i - 0.5) * width, vals, width, color=color, label=label,
                       edgecolor=SURFACE, linewidth=2, zorder=2)
        ax1.bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=3, fontsize=9, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n(test n={int(t):,})" for n, t in zip(names, cmp["test 행"])])
    ax1.set_xlabel("Split method")
    ax1.set_ylabel("Hate speech share (%)")
    ax1.set_ylim(0, 40)
    ax1.set_title("(a) Class balance: hate share in train vs test")
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.legend(title="Split", frameon=False, loc="upper right")

    # (b) 방식별 '이미 본 지문' 비율
    _style(ax2)
    vals = cmp["train에서 본 Context인 test 행(%)"].astype(float).values
    cnts = cmp["train에서 본 Context인 test 행"].astype(int).values
    bars = ax2.bar(x, vals, 0.5, color=test_color, zorder=2)
    ax2.bar_label(bars, labels=[f"{v:.1f}%\n({c:,} / {int(t):,} rows)" for v, c, t in zip(vals, cnts, cmp["test 행"])],
                  padding=3, fontsize=9, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_xlabel("Split method")
    ax2.set_ylabel("Test rows whose Context also appears in train (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Context leakage: test rows with a seen context")
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    fig.suptitle(f"KOCOH v.2 train/test split: row vs post-group (seed={SEED}, test≈20%)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"저장: {out.relative_to(ROOT)}")


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", action="store_true", help="EDA 표 출력 + 그림 2장 저장(덮어씀)")
    args = parser.parse_args()

    np.random.seed(SEED)
    if args.eda:
        print(f"seed = {SEED}")
        eda()
    else:
        # (1) 한 번 나누기 (기본: 게시글 묶음 단위)
        train, test, scaler = load_and_split()

        def ratio(s):
            return f"0={(s == 0).mean() * 100:.3f}% / 1={(s == 1).mean() * 100:.3f}%"

        print("\n[분할 크기 — 게시글 묶음 단위]")
        print(pd.DataFrame({
            "행": [len(train), len(test)],
            "열": [train.shape[1], test.shape[1]],
            "post_group 수": [train["post_group"].nunique(), test["post_group"].nunique()],
            f"{TARGET} 비율": [ratio(train[TARGET]), ratio(test[TARGET])],
        }, index=["train", "test"]).to_string())
        print(f"\n표준화(train fit) mean={dict(zip(NUM_COLS, [float(v) for v in scaler.mean_.round(3)]))}, "
              f"scale={dict(zip(NUM_COLS, [float(v) for v in scaler.scale_.round(3)]))}")
        print(f"test 표준화 후 평균(0이 아니어도 정상): {test[NUM_COLS].mean().round(3).to_dict()}")

        # (2) 모델 입력 꺼내기 예시
        for use_context in (False, True):
            X, y = get_xy(train, use_context=use_context)
            name = "Context + Comment" if use_context else "Comment-only"
            print(f"get_xy({name}): X 열={list(X.columns)}, X {X.shape}, y {y.shape}")

        # (3) 5-fold 교차검증 분할 요약 (전처리까지 실제로 돌려 봄)
        print("\n" + "─" * 60)
        print("5-fold 교차검증 분할")
        for fold, tr, te, _ in cv_splits():
            pass

        # (4) 이전 방식과 비교 표 + 그림
        cmp = compare_splits()
        print("\n[이전(행 단위) vs 변경(게시글 묶음 단위) 비교]")
        print(cmp.astype(object).T.to_string())
        plot_split_compare(cmp)
