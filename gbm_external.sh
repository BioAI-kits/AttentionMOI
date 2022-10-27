
#!/bin/bash
mkdir ./output_gbm_external
touch ./output_gbm_external/evaluation.txt
echo -e "model\tfeature_selection\tomic_group\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./output_gbm_external/evaluation.txt

touch ./external_out/evaluation_external.txt
echo -e "model\tfeature_selection\tACC\tPrecision\tF1_score\tAUC\tRecall"  > ./external_out/evaluation_external.txt

# for i in {1..30}
# do
	## rna
	# FSD
	python deepmoi.py -f ./dataset/GBM/GBM_exp_intersect_cptac.csv.gz -l ./dataset/GBM/GBM_label.csv -n rna -s 1 -b 16 --FSD -m all --threshold 0.6 -o ./output_gbm_external
	# external
	python deepmoi_prediction.py -f ./dataset/GBM-external/GBM_exp_cptac.csv.gz -l ./dataset/GBM-external/gbm_cptac_label.csv -n rna --FSD -m all -s 1 -o ./output_gbm_external

	# FSD+RFE
	python deepmoi.py -f ./dataset/GBM/GBM_exp_intersect_cptac.csv.gz -l ./dataset/GBM/GBM_label.csv -n rna -s 1 -b 16 --FSD -m all --method RFE --threshold 0.6 -o ./output_gbm_external
	# external
	python deepmoi_prediction.py -f ./dataset/GBM-external/GBM_exp_cptac.csv.gz -l ./dataset/GBM-external/gbm_cptac_label.csv -n rna --FSD --method RFE -m all -s 1 -o ./output_gbm_external

	# RFE
	python deepmoi.py -f ./dataset/GBM/GBM_exp_intersect_cptac.csv.gz -l ./dataset/GBM/GBM_label.csv -n rna -s 1 -b 16 -m all --method RFE --threshold 0.6 -o ./output_gbm_external
	# external
	python deepmoi_prediction.py -f ./dataset/GBM-external/GBM_exp_cptac.csv.gz -l ./dataset/GBM-external/gbm_cptac_label.csv -n rna--method RFE -m all -s 1 -o ./output_gbm_external

	# FSD+anova
	python deepmoi.py -f ./dataset/GBM/GBM_exp_intersect_cptac.csv.gz -l ./dataset/GBM/GBM_label.csv -n rna -s 1 -b 16 --FSD -m all --method ANOVA --threshold 0.6 -o ./output_gbm_external
	# external
	python deepmoi_prediction.py -f ./dataset/GBM-external/GBM_exp_cptac.csv.gz -l ./dataset/GBM-external/gbm_cptac_label.csv -n rna --FSD --method ANOVA -m all -s 1 -o ./output_gbm_external

	# anova
	python deepmoi.py -f ./dataset/GBM/GBM_exp_intersect_cptac.csv.gz -l ./dataset/GBM/GBM_label.csv -n rna -s 1 -b 16 -m all --method ANOVA --threshold 0.6 -o ./output_gbm_external
	# external
	python deepmoi_prediction.py -f ./dataset/GBM-external/GBM_exp_cptac.csv.gz -l ./dataset/GBM-external/gbm_cptac_label.csv -n rna --method ANOVA -m all -s 1 -o ./output_gbm_external
# done
