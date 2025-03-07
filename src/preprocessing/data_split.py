import pandas as pd
from src.utils.setup_logger import preprocessing_logger




def log_label_distribution(df: pd.DataFrame, label_column: str, split_name: str, summary_table: pd.DataFrame):
    label_counts = df[label_column].value_counts()
    total = len(df)

    temp_df = pd.DataFrame({
        'Split': split_name,
        'Label': label_counts.index,
        'Count': label_counts.values,
        'Percentage': (label_counts.values / total) * 100
    })

    return pd.concat([summary_table, temp_df], ignore_index=True)


def create_splits(df: pd.DataFrame, label_column: str, test_size: float = 0.2, cv: int = 5, random_state: int = 42):
    from sklearn.model_selection import train_test_split, StratifiedKFold
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df[label_column], random_state=random_state
    )

    split_df = pd.DataFrame(index=df.index)
    split_df['Respiratory cycle'] = df[label_column]
    split_df['train_test'] = split_df.index.isin(train_idx)

    summary_table = pd.DataFrame(columns=['Split', 'Label', 'Count', 'Percentage'])
    summary_table = log_label_distribution(df, label_column, "Total Data", summary_table)
    summary_table = log_label_distribution(df.loc[train_idx], label_column, "Train Split", summary_table)
    summary_table = log_label_distribution(df.loc[test_idx], label_column, "Test Split", summary_table)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(skf.split(df, df[label_column])):
        split_df[f'cv_{i + 1}'] = split_df.index.isin(train_idx)
        summary_table = log_label_distribution(df.loc[train_idx], label_column, f"CV Fold {i + 1} Train", summary_table)
        summary_table = log_label_distribution(df.loc[val_idx], label_column, f"CV Fold {i + 1} Validation",
                                               summary_table)

    preprocessing_logger.info(f"\n{summary_table.to_string(index=False)}")

    return split_df