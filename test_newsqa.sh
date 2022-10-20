python test_newsqa_roberta.py \
	--model_type roberta \
	--output_dir ans_roberta_large_nlock2_docstrike_400_seqlength512 \
	--evaluate_during_training \
	--do_lower_case \
	--learning_rate 1e-5 \
	--num_train_epochs 5 \
	--max_seq_length 512 \
	--max_query_length 64 \
	--max_answer_length 30 \
	--doc_stride 400\
	--per_gpu_eval_batch_size 8 \
	--gradient_accumulation_steps 4 \
	--per_gpu_train_batch_size 8 \
	--logging_steps 5000 \
	--save_steps 2000 \
	--seed 42 \
	--threads 64 \
	--train_file  newsqa_train.json \
	--model_name_or_path roberta-large \
	--overwrite_output_dir\
	--cache_dir ./cache \
	--predict_file  newsqa_dev.json \
	--n_block 2 \
	--version_2_with_negative \
	--warmup_steps 1000 \
	--word_dim 1024 \
	--sent_dim 1024  \
	--do_eval \
	--model_path ans_roberta_large_nlock2_docstrike_400_seqlength512/checkpoint-10000
	#--tokenizer_name ~/work/huggingface/vinai/phobert-base \
	# --config_name ~/work/huggingface/vinai/phobert-base \
	# --eval_all_checkpoint \
	#--version_2_with_negative
	# --null_score_diff_threshold 0.0
	#--predict_file /data/nlp/hoangnv74/question-answering/data/viquad/test_positive.json \
	# --do_eval \

