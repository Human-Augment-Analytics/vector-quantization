@echo off
cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\vector-quantization
set VQ_ALPHAS=1.0
set VQ_DATASET=msmarco500k
python results\lp_joint_alloc\_lp_recall_bench.py > results\lp_joint_alloc\bench_alpha1_msmarco.log 2>&1
set VQ_DATASET=sift10m
python results\lp_joint_alloc\_lp_recall_bench.py > results\lp_joint_alloc\bench_alpha1_sift.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> results\lp_joint_alloc\bench_alpha1_sift.log
