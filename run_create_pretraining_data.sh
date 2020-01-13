python create_pretraining_data.py \
  --input_file=data_vn_bert.raw \
  --output_file=tmp/tf_examples.tfrecord \
  --vocab_file=config/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345