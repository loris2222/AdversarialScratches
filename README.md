# AdversarialScratches
Implementation of "Adversarial Scratches: Deployable Attacks to CNN Classifiers"

## USAGE
To run an experiment, call: 
``/code/experiments.py <dataset_name> <num_samples> <batch_size> <query_limit> <target_model_name> <attack_name> <attack_params> <optimizer> <optimizer_params> <output_path> <bool_use_wandb>``

Example, to run the standard attack on imagenet:
 
``/code/experiments.py imagenet 100 100 10000 torch bezier 133-1-3 cut-ngo 0 test false``

## HOW TO SETUP TSRD DATASET
For licensing reasons, we don't include the full TSRD dataset in the repo. However, we provide our segmentation masks and instructions on how to merge them with the dataset, which you can download from http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html

In your file structure, you should have:

root_folder
|_  datasets
&nbsp;&nbsp;&nbsp;&nbsp; |_ tsrd (place here the downloaded files)
&nbsp;&nbsp;&nbsp;&nbsp; |_ tsrd_mask
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ labels.txt (provided)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ additionalmask (provided)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ test (put here the tsrd images as per the file in the folder)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ test_byclass (put here the tsrd images as per the file in the folder)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 000
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 001
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ ...
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 057
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ testmask (provided)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ train (put here the tsrd images as per the file in the folder)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 000
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 001
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ ...
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |_ 057



