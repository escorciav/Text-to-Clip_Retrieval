
export PYTHONUNBUFFERED=1

for (( i=5; i<=5; i+=1 )); do
  python evaluation_recall.py \
    ../sim_iter_${i}000.p \
    -gt ../../../../preprocess/caption_gt_test.json > recall_results_${i}000.txt
done




