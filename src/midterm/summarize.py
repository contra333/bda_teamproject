"""검증한 팀 제공 CSV로 중간발표 표와 그림을 만든다. 학습은 하지 않는다."""
SEED = 42

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .preprocess import ROOT

RESULTS = ROOT / "results/midterm"
FIGURES = ROOT / "figures/midterm"
BLUE, ORANGE = "#2864A5", "#B55A22"


def tables():
    base = pd.read_csv(RESULTS / "provided/baseline_v2_folds.csv")
    tuned = pd.read_csv(RESULTS / "provided/baseline_v3_folds.csv")
    plm = pd.read_csv(RESULTS / "provided/plm_folds.csv")
    base["method"] = "default threshold"
    plm["rule"] = "default threshold"
    combined = pd.concat([base, plm], ignore_index=True)
    summary = combined.groupby(["model", "method", "condition"], sort=False)[["macro_f1", "recall_hate"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "model_summary.csv", index=False)
    thresholds = tuned.groupby(["model", "condition", "rule"], sort=False)[["macro_f1", "recall_hate", "pr_auc"]].agg(["mean", "std"])
    thresholds.columns = ["_".join(c) for c in thresholds.columns]
    thresholds = thresholds.reset_index()
    thresholds.to_csv(RESULTS / "threshold_summary.csv", index=False)
    return base, tuned, plm, summary, thresholds


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#222222", "text.color": "#222222",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_models(base, plm):
    selections = [("TF-IDF + Logistic Regression", base[base.model == "TF-IDF LogReg"]),
                  ("KcELECTRA (fine-tuning)", plm[(plm.model == "KcELECTRA-base-v2022") & (plm.method == "finetune")]),
                  ("KLUE-RoBERTa (fine-tuning)", plm[(plm.model == "klue-roberta-base") & (plm.method == "finetune")])]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for j, (cond, color, hatch) in enumerate([("Comment-only", BLUE, ""), ("Context+Comment", ORANGE, "//")]):
        means = [g.loc[g.condition == cond, "macro_f1"].mean() for _, g in selections]
        sd = [g.loc[g.condition == cond, "macro_f1"].std(ddof=1) for _, g in selections]
        bars = ax.barh(np.arange(3) + (j-.5)*.30, means, height=.27, xerr=sd,
                       color=color, hatch=hatch, edgecolor="white", capsize=3, label=cond)
        for b, m, e in zip(bars, means, sd):
            ax.text(m+e+.014, b.get_y()+b.get_height()/2, f"{m:.3f}", va="center", fontsize=11)
    floor = base.loc[base.model == "Dummy most_frequent", "macro_f1"].mean()
    ax.axvline(floor, color="#666666", ls=":", lw=1.4)
    ax.text(floor+.008, 2.63, f"Majority baseline {floor:.3f}", fontsize=10)
    ax.set(yticks=np.arange(3), yticklabels=[x[0] for x in selections], xlim=(0,1), ylim=(2.85,-.65), xlabel="Macro-F1 (0–1)")
    ax.grid(axis="x", color="#DDDDDD", lw=.6);ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.14), ncol=2, frameon=False)
    fig.suptitle("Effect of adding context at the default threshold", fontsize=16, y=1.04)
    fig.text(.02,-.025,"KOCOH v2 · 3,000 rows · 77 post groups · 5 outer folds · seed 42\nBars: mean; error bars: sample SD across folds (not seed repeats or confidence intervals).",fontsize=9.5)
    fig.tight_layout();save(fig,"01_context_comparison")


def plot_threshold(tuned):
    g=tuned[tuned.model=="TF-IDF LogReg"]
    fig, axes=plt.subplots(1,2,figsize=(11,5.1),sharey=True)
    conditions=["Comment-only","Context+Comment"]
    for ax, metric, title in zip(axes,["macro_f1","recall_hate"],["Macro-F1","Hate recall"]):
        for j,(rule,label,color,hatch) in enumerate([("0.5 고정","Fixed 0.5",BLUE,""),("OOF 조정","Train-only OOF threshold",ORANGE,"//")]):
            means=[g.loc[(g.condition==c)&(g.rule==rule),metric].mean() for c in conditions]
            sd=[g.loc[(g.condition==c)&(g.rule==rule),metric].std(ddof=1) for c in conditions]
            bars=ax.bar(np.arange(2)+(j-.5)*.32,means,.29,yerr=sd,label=label,color=color,hatch=hatch,edgecolor="white",capsize=3)
            for b,m,e in zip(bars,means,sd):ax.text(b.get_x()+b.get_width()/2,m+e+.018,f"{m:.3f}",ha="center",fontsize=10)
        ax.set(xticks=[0,1],xticklabels=["Comment only","Context + Comment"],ylim=(0,1),title=title,ylabel="Score (0–1)")
        ax.grid(axis="y",color="#DDDDDD",lw=.6);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
    fig.suptitle("TF-IDF + Logistic Regression: threshold comparison",fontsize=16,y=1.10)
    fig.text(.02,-.03,"Same 5 outer test folds · thresholds selected using 4 inner group folds · seed 42\nBars: mean; error bars: sample SD across outer folds. Both input conditions receive the same selection procedure.",fontsize=9.5)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,"02_threshold_comparison")


def main():
    style();base,tuned,plm,summary,thresholds=tables()
    plot_models(base,plm);plot_threshold(tuned)
    print("표 2개 및 PNG/PDF 그림 2쌍 저장 완료")


if __name__=="__main__":main()
