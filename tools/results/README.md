# Results Tools

Utilities for the ICIST results pipeline.

Typical workflow:

```bash
python tools/results/run_ablation_suite.py \
  --bags /bags/forest_01.bag /bags/forest_02.bag \
  --variants naive mask_only mask_edge full \
  --results-dir results/icist_2026 \
  --enable-overlays

bash results/icist_2026/run_all.sh
python tools/results/summarize_metrics.py --results-dir results/icist_2026
python tools/results/make_paper_tables.py --results-dir results/icist_2026
python tools/results/plot_results.py --results-dir results/icist_2026
```

`run_ablation_suite.py` writes a shell script by default. Use `--execute` only
inside a prepared ROS Noetic environment.

The default launch target is `entfac_fusion_ros forestsphere/curt_mini.launch`
because the ICIST runs use the ForestSphere CurtMini bags. For another branch or
launch layout, override it with `--launch-file`.
