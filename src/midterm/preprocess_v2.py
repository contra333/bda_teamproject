"""KOCOH v.2 분할 v2: preprocess.py 의 5-fold 안에 '묶음 단위 내부 4-fold' 를 더한다.

이 파일이 하는 일 (한 줄 요약)
    "test fold 는 그대로 두고, 각 fold 의 train 을 다시 게시글 묶음(post_group) 단위로 4등분해
     임계값 같은 설정을 고르는 데만 쓴다. 분할은 파일로 남기고, 묶음이 겹치면 즉시 멈춘다."

왜 필요한가?
    판정 임계값(0.5)을 바꾸려면 '무엇을 기준으로 고를지' 데이터가 필요하다. test fold 를 보고 고르면 누설이다.
    그래서 train 안에서만 고른다. 내부 4-fold 를 돌아가며 한 조각씩 validation 으로 쓰면(OOF)
    train 의 모든 묶음이 한 번씩 validation 이 되므로, validation 한 조각의 구성에 덜 휘둘린다.

    validation 은 학습 때 보지 않은 묶음(unseen)이어야 한다 — test 상황(처음 보는 게시글)을 흉내 내기 위해서다.
    대신 고른 뒤에는 train 전체로 다시 학습하므로, 최종 모델이 못 보는 것은 지금처럼 test 묶음뿐이다.

지켜야 할 점
    - outer fold(test) 는 preprocess.cv_splits() 와 완전히 같다 → 기존 결과와 바로 비교 가능
    - inner-train / validation / test 사이에 post_group 이 겹치면 assert 로 멈춘다
    - 모든 모델·조건이 같은 분할을 쓰도록 results/midterm/reproduced/split_assignment_v2.csv 에 기록한다
    - validation 구성(행·묶음 수·혐오 비율·대상 집단별 혐오 행 수)을 출력·저장한다.
      치우친 조각이 있어도 점수를 보고 seed 를 바꾸지 않는다 (한계로 기록)

다른 코드에서 사용하는 예
    from .preprocess_v2 import cv_splits_v2
    for fold, train, test, inner in cv_splits_v2():
        for itr, iva in inner:          # train.iloc[itr] 로 학습, train.iloc[iva] 로 예측
            ...
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .preprocess import RESULT_DIR, DATA_PATH, N_SPLITS, ROOT, SEED, TARGET, cv_splits

N_INNER = 4  # train 의 1/4(약 600행)씩 validation — 묶음 수가 6~17개로, 1/8(약 300행, 4~7개)보다 안정적
TARGET_GROUP_COLS = ["Gender", "Disability", "Race/Nationality", "Region (Korea)", "Age"]  # 구성 점검용 (입력 아님)
SPLIT_CSV = RESULT_DIR / "split_assignment_v2.csv"
COMPOSITION_CSV = RESULT_DIR / "inner_val_composition.csv"


def inner_group_folds(train, n_splits=N_INNER, seed=SEED):
    """한 fold 의 train 을 post_group 단위로 n_splits 등분한다. 반환: [(inner-train 위치, validation 위치), ...]"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for itr, iva in sgkf.split(train, train[TARGET], train["post_group"]):
        overlap = set(train["post_group"].iloc[itr]) & set(train["post_group"].iloc[iva])
        assert not overlap, f"inner-train 과 validation 에 같은 post_group: {sorted(overlap)}"
        folds.append((itr, iva))
    return folds


def _composition(fold, k, val):
    """validation 한 조각의 구성. 대상 집단 열은 '구성 점검용' 으로만 읽는다 (모델 입력 아님)."""
    hate = val[val[TARGET] == 1]
    row = {"outer_fold": fold, "inner_fold": k, "rows": len(val), "groups": val["post_group"].nunique(),
           "hate_pct": round(val[TARGET].mean() * 100, 1), "hate_rows": len(hate)}
    row.update({f"hate_{c}": int(hate[c].sum()) for c in TARGET_GROUP_COLS})
    return row


def cv_splits_v2(path=DATA_PATH, n_splits=N_SPLITS, n_inner=N_INNER, seed=SEED, save=True):
    """cv_splits() 와 같은 outer fold 에 내부 묶음 fold 를 붙여 돌려준다.

    내놓는 값: (fold 번호, train_df, test_df, inner) 를 n_splits 번
        inner = [(inner-train 위치, validation 위치), ...] — train_df 기준 위치(iloc)
    save=True 이면 분할 기록과 validation 구성을 results/midterm/reproduced/ 에 저장한다.
    """
    assignment, composition, out = [], [], []
    for fold, train, test, _ in cv_splits(path, n_splits=n_splits, seed=seed):
        assert not set(train["post_group"]) & set(test["post_group"]), "train 과 test 에 같은 post_group"
        inner = inner_group_folds(train, n_splits=n_inner, seed=seed)
        role = pd.Series(-1, index=range(len(train)))
        for k, (_, iva) in enumerate(inner):
            role.iloc[iva] = k
            composition.append(_composition(fold, k, train.iloc[iva]))
        assignment.append(pd.DataFrame({"Index": train["Index"], "outer_fold": fold, "role": "train",
                                        "inner_val_fold": role.to_numpy()}))
        assignment.append(pd.DataFrame({"Index": test["Index"], "outer_fold": fold, "role": "test",
                                        "inner_val_fold": -1}))
        out.append((fold, train, test, inner))

    comp = pd.DataFrame(composition)
    print(f"\n[내부 validation 구성 — outer {n_splits} × inner {n_inner} = {len(comp)}조각] (대상 집단은 점검용)")
    print(comp.to_string(index=False))
    small = comp[comp["groups"] < 6]
    if len(small):
        print(f"  ⚠ 묶음 6개 미만 조각 {len(small)}개 — 점수를 보고 seed 를 바꾸지 말고 한계로 기록할 것")
    if save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(assignment).to_csv(SPLIT_CSV, index=False)
        comp.to_csv(COMPOSITION_CSV, index=False)
        print(f"저장: {SPLIT_CSV.relative_to(ROOT)}, {COMPOSITION_CSV.relative_to(ROOT)}")
    yield from out


if __name__ == "__main__":
    for fold, train, test, inner in cv_splits_v2():
        print(f"fold {fold}: train {len(train)} / test {len(test)} / validation 조각 {[len(v) for _, v in inner]}")
