# CrossSentenceMRCECIR
This is source code, which is implementation for submission: 'Exploiting Question-Context Interaction for
Machine Reading Comprehension'.
We modify this code base on Hugging Face framework for Question Answering: https://huggingface.co/.
Our study conduct the extensive experiments on 2 datasets as: Squad 2.0 vs News QA. 
To reproduce the results, follow exactly the following steps: 

1. Prepare Data 
+ For Squad 2.0: Train set vs Dev set can be easily found and download from link: https://rajpurkar.github.io/SQuAD-explorer/
+ For NewsQa: To convert NewsQa dataset to Squad 2.0 format, please follow strictly code in this link: https://github.com/amazon-research/qa-dataset-converter

2. Train 
After preparing sucessfully data in Squad 2.0 format. For NewsQa, we could have three json files: newsqa_train.json , newsqa_dev.json , newsqa_test.json. Put all of them in same path with 'run_newqa.sh' file.
Run command line: 'bash run_newsqa.sh' to start training process. 

3. Test
To evaluate performance on test set, we only need change --do_train ---> --do_eval and set --dev_set=path_to_test_set. 
Run command line: 'bash test_newqa.sh'

