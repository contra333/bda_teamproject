"""Matplotlib EDA figures using the course W5 comparison-figure conventions.

English figure labels follow W5 slide 68 and the local transfer-learning plot.
300 dpi and vector PDF export are team additions, not instructor requirements.
"""
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

BLUE = '#2870B2'
ORANGE = '#D67523'
INK = '#303030'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.linewidth': .8,
    'axes.titlelocation': 'left', 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white'}


def save_figure(fig, path):
    """Save matched raster/vector figures, replacing completed files atomically."""
    try:
        for suffix in ['.png', '.pdf']:
            target = path.with_suffix(suffix)
            pending = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as handle:
                    pending = Path(handle.name)
                    fig.savefig(handle, format=suffix[1:], dpi=300, facecolor='white')
                os.replace(pending, target)
            finally:
                if pending is not None and pending.exists():
                    pending.unlink()
    finally:
        plt.close(fig)


def count_axis(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#b0b0b0', alpha=.25, linewidth=.7)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_locator(MaxNLocator(integer=True))


def draw_eda(frame, profile, lens, figures, output):
    """Plot measured counts only; no model runs or inferential error bars."""
    n = len(frame)
    with plt.rc_context(STYLE):
        # Figure 1: overall missingness does not identify where it occurs.
        # The conditional panel uses a separate, explicit within-Type denominator.
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.2), layout='constrained',
            gridspec_kw={'width_ratios': [1.25, 1]})
        ax = axes[0]
        positions = np.arange(len(profile))
        ax.barh(positions, profile.missing_pct, color=BLUE, height=.7)
        ax.scatter(np.zeros(len(profile)), positions, s=9, color=INK, zorder=3)
        ax.set_yticks(positions, profile.column, fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, 32)
        ax.set_title('(a) Missing values by column')
        ax.set_xlabel('Missing rows (%)'); ax.set_ylabel('Column')
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        count_axis(ax, 'x')
        for i, value in enumerate(profile.missing_pct):
            if value > 0:
                ax.text(value+.4, i, f'{value:.1f}%', va='center', fontsize=8)
        fields = ['Date', 'Source', 'Link', 'Title']
        rates = frame.groupby('Type')[fields].agg(lambda x: x.isna().mean()*100).reindex([1,2,3])
        ax = axes[1]
        im = ax.imshow(rates, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(4), fields, rotation=35, ha='right')
        sizes = frame.Type.value_counts()
        ax.set_yticks(range(3), [f'Type {i}\nn={sizes[i]:,}' for i in [1,2,3]])
        ax.set(title='(b) Missingness within Type', xlabel='Metadata column', ylabel='Construction type')
        for r in range(3):
            for c in range(4):
                value = rates.iloc[r,c]
                ax.text(c,r,f'{value:.0f}%',ha='center',va='center',fontsize=10,
                    color='white' if value > 50 else INK)
        cbar = fig.colorbar(im, ax=ax, shrink=.65)
        cbar.set_label('Missing rows within Type (%)', fontsize=9)
        fig.supxlabel(f'KOCOH v2 | n={n:,} pairs | (a) all rows; (b) rows within each Type', fontsize=9)
        save_figure(fig, figures/'01_missingness.png')

        fig, ax = plt.subplots(figsize=(7.4,4.5),layout='constrained')
        counts = frame['Hate speech'].value_counts().reindex([0,1],fill_value=0)
        bars = ax.bar(['Non-hate (0)','Hate (1)'],counts,color=[BLUE,ORANGE],
            edgecolor=INK, linewidth=.6,width=.55)
        bars[1].set_hatch('//')
        for i,value in enumerate(counts):
            ax.text(i,value+35,f'{value:,}\n({value/n:.2%})',ha='center',va='bottom')
        ax.set(xlabel='Hate speech label',ylabel='Number of context-comment pairs',
            title='Class distribution in KOCOH v2',ylim=(0,counts.max()*1.25))
        count_axis(ax)
        fig.supxlabel(f'n={n:,} pairs | Majority-class accuracy: {counts.max()/n:.2%} | Descriptive counts',fontsize=9)
        save_figure(fig,figures/'02_class_balance.png')

        # Equal bin widths AND equal scales make the two marginal distributions comparable.
        bins = np.arange(0, int(np.ceil(lens[['Context_chars','Comment_chars']].max().max()/10))*10+11, 10)
        fig, axes = plt.subplots(1,2,figsize=(8.4,4.5),sharex=True,sharey=True,layout='constrained')
        histogram = []
        for ax, field, panel in zip(axes,['Context','Comment'],['a','b']):
            values = lens[f'{field}_chars']
            counts, _ = np.histogram(values,bins=bins)
            percentages = counts/n*100
            ax.bar(bins[:-1], percentages, width=10, align='edge', color=BLUE,
                edgecolor='white',linewidth=.5)
            median, q95 = values.quantile([.5,.95])
            ax.axvline(median,color=INK,ls='--',lw=1.2,label=f'Median: {median:g}')
            ax.axvline(q95,color=ORANGE,ls=':',lw=1.7,label=f'95th percentile: {q95:g}')
            ax.set(title=f'({panel}) {field}',xlabel='Text length (characters)',xlim=(0,bins[-1]))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
            count_axis(ax);ax.legend(loc='upper right',frameon=False,fontsize=8)
            for lo,hi,count in zip(bins[:-1],bins[1:],counts):
                histogram.append({'field':field,'bin_left_inclusive':int(lo),
                    'bin_right':int(hi),'last_bin_right_inclusive':bool(hi==bins[-1]),
                    'rows':int(count),'pct_of_all_pairs':float(count/n*100)})
        axes[0].set_ylabel('Share of context-comment pairs (%)')
        axes[1].tick_params(labelleft=True)
        fig.supxlabel(f'n={n:,} per panel | Common bin width: 10 characters | Shared axes; characters are not tokens',fontsize=8.5)
        pd.DataFrame(histogram).to_csv(output/'figure03_histogram.csv',index=False)
        save_figure(fig,figures/'03_text_lengths.png')

        cross = pd.crosstab(frame.Type,frame['Hate speech']).reindex(index=[1,2,3],columns=[0,1],fill_value=0)
        fig, ax = plt.subplots(figsize=(7.4,4.5),layout='constrained')
        x = np.arange(3);width=.31
        for offset,label,color,hatch,name in [(-width/2,0,BLUE,'','Non-hate (0)'),
                                             (width/2,1,ORANGE,'//','Hate (1)')]:
            bars=ax.bar(x+offset,cross[label],width,color=color,hatch=hatch,edgecolor=INK,linewidth=.6,label=name)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+18,f'{int(b.get_height()):,}',
                    ha='center',va='bottom',fontsize=9)
        ax.set_xticks(x,['Type 1','Type 2','Type 3'],fontsize=10)
        ax.set(xlabel='Construction type (excluded from model inputs)',ylabel='Number of context-comment pairs',
            title='Label counts by construction type',ylim=(0,cross.to_numpy().max()*1.23))
        ax.legend(loc='upper left',frameon=False);count_axis(ax)
        fig.supxlabel(f'KOCOH v2 | n={n:,} pairs | Descriptive counts; no rows excluded',fontsize=9)
        save_figure(fig,figures/'04_type_labels.png')
