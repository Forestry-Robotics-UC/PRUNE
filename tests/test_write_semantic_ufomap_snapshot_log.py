from pathlib import Path
import subprocess


def test_write_semantic_ufomap_snapshot_log_copies_live_log(tmp_path):
    source = tmp_path / "run_console.log"
    out = tmp_path / "snapshot" / "run_console.log"
    source.write_text("line one\nline two\n")

    result = subprocess.run(
        [
            "python3",
            "scripts/ufomap/write_semantic_ufomap_snapshot_log.py",
            "--source",
            str(source),
            "--output",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out.read_text() == "line one\nline two\n"


def test_write_semantic_ufomap_snapshot_log_allows_same_source_and_output(tmp_path):
    source = tmp_path / "run_console.log"
    source.write_text("line one\n")

    result = subprocess.run(
        [
            "python3",
            "scripts/ufomap/write_semantic_ufomap_snapshot_log.py",
            "--source",
            str(source),
            "--output",
            str(source),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert source.read_text() == "line one\n"
