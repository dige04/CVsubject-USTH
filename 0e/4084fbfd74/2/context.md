# Session Context

## User Prompts

### Prompt 1

Giải thích
full_pipeline.ipynb
full_pipeline.ipynb_
Vision-First Multi-Modal Hand Gesture Recognition Pipeline
Full end-to-end pipeline for training a multi-modal HGR system on Google Colab.

This notebook is self-contained: it uses %%writefile to create each script file in the Colab filesystem, then runs them with !python. No separate repo push required.

Pipeline Overview
Phase    Description    Script(s)
01    Data Pipeline: download HaGRID, convert to YOLO, extract crops    download_hagri...

### Prompt 2

fix entire problem

### Prompt 3

Writing ml/scripts/run_ablation.py

[29]
3s
# Run ablation study
# Run ablation study
!python ml/scripts/run_ablation.py \
    --paired-csv data/paired_fusion.csv \
    --n-splits 5 \
    --output-dir results/ablation

Traceback (most recent call last):
  File "/content/CVsubject/ml/scripts/run_ablation.py", line 512, in <module>
    main()
  File "/content/CVsubject/ml/scripts/run_ablation.py", line 502, in main
    run_ablation(
  File "/content/CVsubject/ml/scripts/run_ablation.py", line 283,...

### Prompt 4

full_pipeline.ipynb
full_pipeline.ipynb_
Vision-First Multi-Modal Hand Gesture Recognition Pipeline
Full end-to-end pipeline for training a multi-modal HGR system on Google Colab.

This notebook is self-contained: it uses %%writefile to create each script file in the Colab filesystem, then runs them with !python. No separate repo push required.

Pipeline Overview
Phase    Description    Script(s)
01    Data Pipeline: download HaGRID, convert to YOLO, extract crops, extract landmarks    download_...

