import pandas as pd
import numpy as np
from typing import List

def load_au_file(file):
    df = pd.read_csv(file)
    au_cols = [c for c in df.columns if c.startswith("AU")]
    return df, au_cols

def load_valence_file(file):
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=[np.number])
    return numeric_cols.iloc[:, 0].values.astype(float)

def compute_au_energy(df, au_cols):
    return df[au_cols].sum(axis=1).values

def temporal_filter(x, k=3):
    return np.convolve(x, np.ones(k)/k, mode="same")

def compute_suppression_score(au_energy, valence):
    delta_valence = np.abs(np.diff(valence, prepend=valence[0]))
    return au_energy * delta_valence

def process_single_pair(au_csv, val_csv):
    """Process a single AU and valence file pair."""
    au_df, au_cols = load_au_file(au_csv)
    valence = load_valence_file(val_csv)

    au_energy = compute_au_energy(au_df, au_cols)
    au_energy = (au_energy - au_energy.min()) / (au_energy.max() - au_energy.min() + 1e-6)

    min_len = min(len(au_energy), len(valence))
    au_energy = au_energy[:min_len]
    valence = valence[:min_len]

    suppression = compute_suppression_score(au_energy, valence)
    suppression = temporal_filter(suppression, k=3)

    return suppression

def run_suppression_pipeline(au_csvs: List[str], val_csvs: List[str]):
    """Process multiple AU and valence file pairs."""
    all_suppressions = []
    file_results = []
    
    # Process each pair of files
    num_pairs = min(len(au_csvs), len(val_csvs))
    
    for i in range(num_pairs):
        suppression = process_single_pair(au_csvs[i], val_csvs[i])
        all_suppressions.append(suppression)
        
        file_results.append({
            "file_pair": i + 1,
            "mean_suppression": round(float(np.mean(suppression)), 4),
            "std_suppression": round(float(np.std(suppression)), 4),
            "frames": int(len(suppression))
        })
    
    # Combine all suppression scores
    combined_suppression = np.concatenate(all_suppressions)
    
    return {
        "overall": {
            "mean_suppression": round(float(np.mean(combined_suppression)), 4),
            "std_suppression": round(float(np.std(combined_suppression)), 4),
            "total_frames": int(len(combined_suppression)),
            "files_processed": num_pairs
        },
        "per_file": file_results
    }
