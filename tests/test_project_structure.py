from pathlib import Path
 
 
def test_required_files_exist():
    required_paths = [
        "src/train.py",
        "src/validate_data.py",
        "src/version_data.py",
        "app/main.py",
        "requirements.txt",
        "Dockerfile",
    ]
    for path in required_paths:
        assert Path(path).exists(), f"Missing required file: {path}"