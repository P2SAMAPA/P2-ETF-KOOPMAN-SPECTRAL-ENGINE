"""
HuggingFace Hub uploader for Koopman-Spectral results.
Uploads models and signals to P2SAMAPA/p2-etf-koopman-spectral-results
"""

import os
import json
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


HF_RESULTS_REPO = "P2SAMAPA/p2-etf-koopman-spectral-results"


def get_hf_token():
    """Get HF token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    return token


def ensure_repo_exists(token: str):
    """Create repo if it doesn't exist."""
    api = HfApi()
    try:
        api.repo_info(repo_id=HF_RESULTS_REPO, repo_type="dataset", token=token)
        print(f"Repository exists: {HF_RESULTS_REPO}")
    except RepositoryNotFoundError:
        print(f"Creating repository: {HF_RESULTS_REPO}")
        create_repo(
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
            private=False,
            token=token,
            exist_ok=True
        )


def upload_model(model_path: str, training_date: str, token: str):
    """Upload trained model to HF."""
    api = HfApi()
    
    filename = Path(model_path).name
    path_in_repo = f"models/{training_date}/{filename}"
    
    print(f"Uploading {model_path} to {HF_RESULTS_REPO}/{path_in_repo}")
    
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=path_in_repo,
        repo_id=HF_RESULTS_REPO,
        repo_type="dataset",
        token=token
    )
    
    return f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/blob/main/{path_in_repo}"


def upload_signals(signals: dict, token: str):
    """Upload signals JSON to HF."""
    api = HfApi()
    
    date_str = signals['signal_date']
    filename = f"koopman_signals_{date_str}.json"
    path_in_repo = f"signals/{filename}"
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(signals, f, indent=2)
        temp_path = f.name
    
    print(f"Uploading signals to {HF_RESULTS_REPO}/{path_in_repo}")
    
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo=path_in_repo,
        repo_id=HF_RESULTS_REPO,
        repo_type="dataset",
        token=token
    )
    
    # Also upload as "latest.json" for easy access
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="signals/latest.json",
        repo_id=HF_RESULTS_REPO,
        repo_type="dataset",
        token=token
    )
    
    return f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/blob/main/{path_in_repo}"


def upload_training_history(history_path: str, training_date: str, token: str):
    """Upload training history to HF."""
    api = HfApi()
    
    filename = Path(history_path).name
    path_in_repo = f"training_history/{training_date}/{filename}"
    
    print(f"Uploading {history_path} to {HF_RESULTS_REPO}/{path_in_repo}")
    
    api.upload_file(
        path_or_fileobj=history_path,
        path_in_repo=path_in_repo,
        repo_id=HF_RESULTS_REPO,
        repo_type="dataset",
        token=token
    )
    
    return f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/blob/main/{path_in_repo}"


def upload_all_results(training_date: str = None):
    """Upload all results from current run."""
    if training_date is None:
        training_date = datetime.now().strftime("%Y-%m-%d")
    
    token = get_hf_token()
    ensure_repo_exists(token)
    
    results = {
        "training_date": training_date,
        "repo": HF_RESULTS_REPO,
        "files": []
    }
    
    # Upload model
    if Path("koopman_spectral_best.pt").exists():
        url = upload_model("koopman_spectral_best.pt", training_date, token)
        results["files"].append({"type": "model", "path": "koopman_spectral_best.pt", "url": url})
    
    if Path("koopman_spectral_final.pt").exists():
        url = upload_model("koopman_spectral_final.pt", training_date, token)
        results["files"].append({"type": "model_final", "path": "koopman_spectral_final.pt", "url": url})
    
    # Upload training history
    if Path("training_history.json").exists():
        url = upload_training_history("training_history.json", training_date, token)
        results["files"].append({"type": "history", "path": "training_history.json", "url": url})
    
    # Upload signals
    signals_dir = Path("signals")
    if signals_dir.exists():
        for signal_file in sorted(signals_dir.glob("koopman_signals_*.json"), reverse=True):
            with open(signal_file) as f:
                signals = json.load(f)
            url = upload_signals(signals, token)
            results["files"].append({"type": "signals", "path": str(signal_file), "url": url})
            break  # Only upload most recent
    
    # Save upload manifest locally
    with open("hf_upload_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Upload Complete ===")
    print(f"Results repo: https://huggingface.co/datasets/{HF_RESULTS_REPO}")
    print(f"Files uploaded: {len(results['files'])}")
    
    return results


if __name__ == "__main__":
    upload_all_results()
