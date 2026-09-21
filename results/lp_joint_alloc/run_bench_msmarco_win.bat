@echo off
cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\vector-quantization
set VQ_DATASET=msmarco500k
python results\lp_joint_alloc\_lp_recall_bench.py > results\lp_joint_alloc\bench_msmarco_win.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> results\lp_joint_alloc\bench_msmarco_win.log
