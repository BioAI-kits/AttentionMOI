#!/bin/bash
mkdir ./luad_multi_fsd_method
touch ./luad_multi_fsd_method/evaluation.txt
echo -e "model\tfeature_selection\tomic_group\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./luad_multi_fsd_method/evaluation.txt

for i in $(seq 1 10) 
do
	#FSD+method
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 --FSD --method LASSO -m all --threshold 0.6 -s $i -o ./luad_multi_fsd_method
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 --FSD --method RFE -m all --threshold 0.6 -s $i -o ./luad_multi_fsd_method
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 --FSD --method ANOVA -m all --threshold 0.6 -s $i -o ./luad_multi_fsd_method
	python deepmoi.py -f ./dataset/LUAD/LUAD_exp.csv.gz -f ./dataset/LUAD/LUAD_met.csv.gz -f ./dataset/LUAD/LUAD_logRatio.csv.gz -l ./dataset/LUAD/LUAD_label.csv -n rna -n met -n cnv -b 16 --FSD --method PCA -m all --threshold 0.6 -s $i -o ./luad_multi_fsd_method
done
