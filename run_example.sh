# Prepare PWM and codon usage table
#bash gen_PWM.sh
#bash gen_codon_usage.sh

# Generate features
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 2
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 3
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 4
python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 5

# Model training
python script/train_model.py -feature Features_Sl_CDSATG.txt
python script/train_model.py -feature Features_Sl_5UTRATG.txt
python script/train_model.py -feature Features_Sl_CDSnonATG.txt
python script/train_model.py -feature Features_Sl_5UTRnonATG.txt

# Cross model evaluation
python script/test_model.py -feature Features_Sl_5UTRATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

# TIS prediction
python script/predict_TIS.py -fasta input_data/predict_input.fa -models Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

