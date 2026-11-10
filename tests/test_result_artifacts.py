from pathlib import Path
import importlib.util


def test_existing_exported_predictions_reproduce_tables():
    p=Path(__file__).resolve().parents[1]
    script=p/'scripts'/'verify_reported_results.py'
    spec=importlib.util.spec_from_file_location('verify_results',script)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main(p)
