from pathlib import Path
import subprocess


def test_write_semantic_ufomap_run_params_writes_expected_yaml(tmp_path):
    out_path = tmp_path / "run_params.yaml"

    result = subprocess.run(
        [
            "python3",
            "scripts/ufomap/write_semantic_ufomap_run_params.py",
            "--output",
            str(out_path),
            "--semantic-bag",
            "/bags/semantic_input.bag",
            "--localization-bag",
            "/bags/localization_input.bag",
            "--play-rate",
            "1.0",
            "--rosbag-skip-empty-sec",
            "20",
            "--start-sec",
            "0.0",
            "--rviz",
            "false",
            "--localization-topic",
            "/localization_50hz",
            "--localization-parent-frame",
            "odom",
            "--localization-yaw-offset-deg",
            "180",
            "--localization-use-stamp-source",
            "false",
            "--localization-stamp-source-topic",
            "",
            "--localization-stamp-source-type",
            "pointcloud2",
            "--localization-stamp-source-max-age-sec",
            "0.10",
            "--ufomap-resolution",
            "0.10",
            "--ufomap-depth-levels",
            "16",
            "--ufomap-num-workers",
            "2",
            "--ufomap-color",
            "true",
            "--ufomap-max-range",
            "30",
            "--ufomap-prob-hit",
            "0.75",
            "--ufomap-prob-miss",
            "0.35",
            "--ufomap-pub-rate",
            "10.0",
            "--ufomap-export-ply",
            "true",
            "--ufomap-export-mesh",
            "true",
            "--ufomap-export-screenshots",
            "true",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    text = out_path.read_text()
    assert "semantic_bag: /bags/semantic_input.bag\n" in text
    assert "localization_bag: /bags/localization_input.bag\n" in text
    assert "rosbag_skip_empty_sec: 20\n" in text
    assert "localization_topic: /localization_50hz\n" in text
    assert "localization_parent_frame: odom\n" in text
    assert "localization_yaw_offset_deg: 180\n" in text
    assert "localization_use_stamp_source: false\n" in text
    assert "ufomap_max_range: 30\n" in text
    assert "ufomap_prob_hit: 0.75\n" in text
    assert "ufomap_prob_miss: 0.35\n" in text
    assert "ufomap_pub_rate: 10\n" in text
