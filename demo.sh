mkdir ./external_out
touch ./external_out/evaluation.txt
echo -e "model\tfeature_selection\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./external_out/evaluation.txt
python deepmoi.py -f ./dataset/Test/rna.csv.gz -f ./dataset/Test/cnv.csv.gz -f ./dataset/Test/met.csv.gz  -l ./dataset/Test/label.csv -n rna -n cnv -n met --FSD --method RFE -m all -s 42 -b 16 --threshold 0.6 -o ./external_out

touch ./external_out/evaluation.txt
echo -e "model\tfeature_selection\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./external_out/evaluation_external.txt
python deepmoi_prediction.py -f ./dataset/Test/rna.csv.gz -f ./dataset/Test/cnv.csv.gz -f ./dataset/Test/met.csv.gz  -l ./dataset/Test/label.csv -n rna -n cnv -n met --FSD --method RFE -m all -s 42 -o ./external_out