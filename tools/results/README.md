# Results Tools

Utilities for reusable ablation-study result generation.

Typical workflow:

```bash
python tools/results/run_ablation_suite.py \
  --bags /bags/forest_01.bag /bags/forest_02.bag \
  --variants naive mask edge occlusion mask_edge mask_occlusion edge_occlusion full \
  --study-name forest_ablation_2026 \
  --enable-overlays \
  --duration-sec 1200

bash results/forest_ablation_2026/run_all.sh
python tools/results/summarize_metrics.py --results-dir results/forest_ablation_2026
python tools/results/make_paper_tables.py --results-dir results/forest_ablation_2026
python tools/results/plot_results.py --results-dir results/forest_ablation_2026
```

`run_ablation_suite.py` writes a shell script by default. Use `--execute` only
inside a prepared ROS Noetic environment.

The default launch target is `entfac_fusion_ros forestsphere/curt_mini.launch`
because the current reusable baseline is the ForestSphere CurtMini replay.
For another branch, platform, or launch layout, override it with `--launch-file`.

Use `--study-name icist_2026` when you want the old conference-specific output naming without changing the general workflow.

Use `--extra-bags` for localization or auxiliary bags. When a continuous replay
needs multiple ROS1 bag chunks, pass them as one quoted `--bags` value so one
`rosbag play --duration` window covers the combined timeline.
