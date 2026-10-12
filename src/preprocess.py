"""Validated, fixed Set-level splits; no learned transformation is fit here."""
SEED = 42

from pathlib import Path
import argparse
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_PATH = DATA_DIR / 'KOCOH_v.2.csv'
EXPECTED_COLUMNS = ['Index', 'Set', 'Type', 'Date', 'Source', 'Link', 'Title',
    'Context', 'Comment', 'Hate speech', 'Counter speech', 'Gender', 'Disability',
    'Race/Nationality', 'Region (Korea)', 'Age', 'Profanity', 'Ikiyano-style']
LABELS = EXPECTED_COLUMNS[9:]


def versions():
    return {'python': platform.python_version(), **{n: importlib.metadata.version(n)
        for n in ['pandas', 'numpy', 'scikit-learn', 'matplotlib']}}


def load_data(path=RAW_PATH):
    path = Path(path).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise ValueError('Data must be read from the project data/ directory.')
    manifest = DATA_DIR / 'provenance.json'
    if path == RAW_PATH.resolve() and manifest.exists():
        expected = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Raw data hash mismatch; do not overwrite the fixed split.')
    frame = pd.read_csv(path, encoding='utf-8-sig')
    validate(frame)
    return frame


def validate(frame):
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError('Unexpected schema. Review the dataset version first.')
    if frame.empty or frame['Index'].isna().any() or frame['Index'].duplicated().any():
        raise ValueError('Index must be nonmissing and unique.')
    for column in ['Index', 'Set', 'Type'] + LABELS:
        values = pd.to_numeric(frame[column], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f'{column}: invalid integer values; no rows silently removed.')
    for column in ['Index', 'Set']:
        if (frame[column] <= 0).any():
            raise ValueError(f'{column} must be positive.')
    for column in LABELS:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0/1.')
    if not frame['Type'].isin([1, 2, 3]).all():
        raise ValueError('Unexpected Type.')
    for column in ['Context', 'Comment']:
        if frame[column].isna().any() or frame[column].str.strip().eq('').any():
            raise ValueError(f'{column} has missing/blank input; no automatic deletion.')


def fixed_assignment(frame):
    # Preserve pinned raw row order. Never search seeds using model performance.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(frame, frame['Hate speech'], groups=frame['Set']))
    assignment = pd.DataFrame({'Index': frame['Index'].to_numpy(),
        'Set': frame['Set'].to_numpy(), 'split': 'train'})
    assignment.loc[folds[0][1], 'split'] = 'test'
    assignment.loc[folds[1][1], 'split'] = 'valid'
    check_assignment(frame, assignment)
    return assignment


def check_assignment(frame, assignment):
    if len(assignment) != len(frame) or not assignment['Index'].is_unique:
        raise ValueError('Split does not preserve unique rows.')
    if assignment['Index'].tolist() != frame['Index'].tolist():
        raise ValueError('Split row order/IDs differ from the original.')
    if assignment['Set'].tolist() != frame['Set'].tolist():
        raise ValueError('Split Set IDs differ from the original.')
    if set(assignment['split']) != {'train', 'valid', 'test'}:
        raise ValueError('Three nonempty splits required.')
    if assignment.groupby('Set')['split'].nunique().max() != 1:
        raise ValueError('Post leakage across splits.')
    for name in ['train', 'valid', 'test']:
        if set(frame.loc[assignment['split'].eq(name).to_numpy(), 'Hate speech']) != {0, 1}:
            raise ValueError(f'{name} needs both labels.')


def model_inputs(frame, condition):
    """Return text input and target separately. Fit vectorizers on train only.

    For transformers, pass the two returned text columns as tokenizer(text,
    text_pair=...) rather than exposing metadata or gold auxiliary labels.
    """
    if condition == 'comment_only':
        columns = ['Comment']
    elif condition == 'context_comment':
        columns = ['Context', 'Comment']
    else:
        raise ValueError('Use comment_only or context_comment.')
    return frame[columns].copy(), frame['Hate speech'].copy()


def load_and_split(path=RAW_PATH, persist=False, verbose=True):
    frame = load_data(path)
    assignment = fixed_assignment(frame)
    saved = DATA_DIR / 'split_assignment.csv'
    if saved.exists():
        previous = pd.read_csv(saved)
        check_assignment(frame, previous)
        if not previous.equals(assignment):
            raise ValueError('Recomputed split changed. Check input/order/library versions; do not overwrite.')
    splits = {name: frame.loc[assignment['split'].eq(name).to_numpy()].copy()
              for name in ['train', 'valid', 'test']}
    if persist:
        assignment.to_csv(saved, index=False)
        for name, subset in splits.items():
            # These CSVs contain metadata for auditing. model_inputs selects allowed text only.
            subset.to_csv(DATA_DIR / f'{name}.csv', index=False, encoding='utf-8-sig')
        recipe = {'seed': SEED, 'algorithm': 'StratifiedGroupKFold', 'n_splits': 5,
            'shuffle': True, 'group': 'Set', 'stratify': 'Hate speech',
            'test_fold': 0, 'valid_fold': 1, 'train_folds': [2, 3, 4],
            'removed_rows': 0, 'normalization': 'none; original text retained',
            'versions': versions(), 'raw_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        (DATA_DIR / 'split_recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    if verbose:
        print(f'Original {len(frame)} rows; removed 0 rows; text unchanged.')
        for name, subset in splits.items():
            print(name, 'rows=', len(subset), 'posts=', subset.Set.nunique(),
                  'labels=', subset['Hate speech'].value_counts().sort_index().to_dict())
    return splits, assignment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without writing output files.')
    args = parser.parse_args()
    load_and_split(persist=not args.check)
