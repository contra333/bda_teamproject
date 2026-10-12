"""Reproducible descriptive EDA, before any model training or test scoring."""
SEED = 42

import json
from itertools import combinations
import pandas as pd
from src.preprocess import ROOT, LABELS, load_data, load_and_split, versions
from src.plotting import draw_eda


def analyze(verbose=True):
    frame = load_data()
    splits, assignment = load_and_split(persist=True, verbose=verbose)
    output = ROOT / 'results'
    figures = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    tables = {}
    profile = pd.DataFrame({'column': frame.columns, 'dtype': frame.dtypes.astype(str).values,
        'missing_count': frame.isna().sum().values,
        'missing_pct': frame.isna().mean().values * 100,
        'unique_nonmissing': frame.nunique().values,
        'blank_count': [int(frame[c].dropna().astype(str).str.strip().eq('').sum()) for c in frame]})
    tables['column_profile'] = profile
    tables['numeric_ranges'] = frame.select_dtypes('number').agg(['min', 'max']).T.reset_index(names='column')
    tables['missing_by_type'] = frame.assign(**{c: frame[c].isna().astype(int) for c in frame.columns if c != 'Type'}).groupby('Type').sum().reset_index()
    tables['type_label'] = pd.crosstab(frame.Type, frame['Hate speech']).reset_index()
    tables['source_raw'] = frame.Source.fillna('미기재 (Type 2)').value_counts().rename_axis('source').reset_index(name='rows')
    # Parent-source attribution is a separate derived field, never fills original nulls.
    if frame.groupby('Set').Source.nunique().max() > 1:
        raise ValueError('A Set maps to multiple sources; investigate before source attribution.')
    parent_source = frame.groupby('Set').Source.first()
    tables['source_parent_attributed'] = frame.Set.map(parent_source).value_counts(dropna=False).rename_axis('source').reset_index(name='rows')
    lens = pd.DataFrame(index=frame.index)
    for field in ['Context', 'Comment']:
        lens[f'{field}_chars'] = frame[field].str.len()
        lens[f'{field}_whitespace_units'] = frame[field].str.split().str.len()
    tables['text_lengths'] = lens.describe(percentiles=[.5, .9, .95, .99]).T.reset_index(names='measure')
    tables['lengths_by_type'] = pd.concat([frame[['Type']], lens], axis=1).groupby('Type').agg(['min', 'median', 'max'])
    tables['split_summary'] = pd.DataFrame([{'split': name, 'rows': len(d), 'posts': d.Set.nunique(),
        'nonhate': int(d['Hate speech'].eq(0).sum()), 'hate': int(d['Hate speech'].eq(1).sum()),
        'hate_pct': float(d['Hate speech'].mean()*100)} for name, d in splits.items()])
    tables['split_type'] = pd.crosstab(assignment['split'], frame.Type).reset_index()
    overlaps = []
    for a, b in combinations(splits, 2):
        for col in ['Set', 'Context', 'Comment']:
            shared = set(splits[a][col]) & set(splits[b][col])
            overlaps.append({'split_a': a, 'split_b': b, 'field': col,
                'shared_unique_values': len(shared), 'affected_rows_a': int(splits[a][col].isin(shared).sum()),
                'affected_rows_b': int(splits[b][col].isin(shared).sum())})
        pa = list(zip(splits[a].Context, splits[a].Comment))
        pb = list(zip(splits[b].Context, splits[b].Comment))
        common = set(pa) & set(pb)
        overlaps.append({'split_a': a, 'split_b': b, 'field': 'Context+Comment',
            'shared_unique_values': len(common), 'affected_rows_a': sum(p in common for p in pa),
            'affected_rows_b': sum(p in common for p in pb)})
    tables['split_overlap'] = pd.DataFrame(overlaps)
    pairs = frame.groupby(['Context', 'Comment'], dropna=False).agg(rows=('Index','size'),
        label_count=('Hate speech','nunique'), set_count=('Set','nunique'))
    comments = frame.groupby('Comment').agg(rows=('Index','size'),
        label_count=('Hate speech','nunique'), set_count=('Set','nunique'))
    # Intentional Type 1/2 pairs are counted by Set + exact Comment, not CSV adjacency.
    pair_types = frame[frame.Type.isin([1,2])].groupby(['Set','Comment']).Type.agg(lambda v:set(v))
    pairs_12 = sum(v == {1,2} for v in pair_types)
    unmatched_keys = {key for key, value in pair_types.items() if value != {1,2}}
    unmatched = frame[frame.apply(lambda r: (r['Set'],r['Comment']) in unmatched_keys and r['Type'] in [1,2],axis=1)]
    tables['type12_set_exceptions'] = unmatched[['Index','Set','Type']].merge(assignment,on=['Index','Set'])
    tables['duplicate_pair_ids'] = frame[frame.duplicated(['Context','Comment'],keep=False)][['Index','Set','Type','Hate speech']].merge(assignment,on=['Index','Set'])
    targets = frame.loc[frame['Hate speech'].eq(1), LABELS[2:7]]
    tables['target_labels'] = pd.DataFrame({'target': targets.columns, 'positive_rows': targets.sum().values,
        'pct_of_hate_rows': targets.mean().values * 100})
    dates = pd.to_datetime(frame.Date.dropna(), errors='coerce')
    summary = {'rows': len(frame), 'columns': len(frame.columns), 'posts': int(frame.Set.nunique()),
        'label_counts': {str(k):int(v) for k,v in frame['Hate speech'].value_counts().sort_index().items()},
        'majority_accuracy': float(frame['Hate speech'].eq(0).mean()),
        'majority_hate_precision': 0.0, 'majority_hate_recall': 0.0, 'majority_hate_f1': 0.0,
        'type_counts': {str(k):int(v) for k,v in frame.Type.value_counts().sort_index().items()},
        'removed_rows': 0, 'duplicate_rows_excluding_index': int(frame.drop(columns='Index').duplicated().sum()),
        'duplicate_pair_excess_rows': int(frame.duplicated(['Context','Comment']).sum()),
        'conflicting_exact_pair_groups': int(pairs.label_count.gt(1).sum()),
        'comments_with_multiple_labels': int(comments.label_count.gt(1).sum()),
        'comments_across_sets': int(comments.set_count.gt(1).sum()),
        'contexts_across_sets': int(frame.groupby('Context').Set.nunique().gt(1).sum()),
        'complete_type12_comment_groups': int(pairs_12),
        'incomplete_type12_comment_groups': int(sum(v != {1,2} for v in pair_types)),
        'date_min': str(dates.min().date()), 'date_max': str(dates.max().date()),
        'invalid_dates': int(dates.isna().sum()), 'versions': versions(),
        'scope': 'Full-data structural audit; no model fit, predictions, test errors or scores inspected.'}
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=name=='lengths_by_type', encoding='utf-8-sig')
    (output / 'eda_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    draw_eda(frame, profile, lens, figures, output)
    if verbose:
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print(tables['split_overlap'].to_string(index=False))
    return summary,tables


if __name__ == '__main__':
    analyze()
