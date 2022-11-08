#!/bin/bash
mkdir ./luad_multi
touch ./luad_multi/evaluation.txt
echo -e "model\tfeature_selection\tomic_group\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./luad_multi/evaluation.txt

for i in $(seq 1 10) 
do
	# FSD
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 --FSD -m all --threshold 0.6 -s $i -o ./luad_multi
	# No FSD
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 -m all --method LASSO --threshold 0.6 -s $i -o ./luad_multi
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 -m all--method RFE --threshold 0.6 -s $i -o ./luad_multi
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 -m all --method ANOVA --threshold 0.6 -s $i -o ./luad_multi
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 -m all --method PCA --threshold 0.6 -s $i -o ./luad_multi
done
