# ALBERT for Vietnamese on word level ( apply word segmentation)
======

***************New January 13 , 2020 ***************

**[ALBERT](https://github.com/google-research/ALBERT)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.

AlBert for Vietnamese models are released.

In this version, we apply vietnamese word segmentation when tokenization. We train [ALBERT-base](https://github.com/google-research/ALBERT) for 300k steps.

Petrained using  95274 words in vocab use tokenizer from [VncoreNLP](https://github.com/vncorenlp/VnCoreNLP), on Vietnamese wikipedia corpus from Wikipedia + 400k vietnamese news (We don't release this dataset).

We don't use word sentencepiece, instead of we use basic bert tokenization with lower case input and is tokenized by VncoreNLP. we have modified tokenizer of bert for keep character _ (word segmentation char).

You can download trained model:
- [tensorflow](https://vs-insai-storage.s3-ap-southeast-1.amazonaws.com/albert/tf/albert_base.zip).
- [pytorch](https://vs-insai-storage.s3-ap-southeast-1.amazonaws.com/albert/pytorch/albert_base.zip).

``` bash

***** Eval results *****
global_step = 300000
loss = 1.6965494
masked_lm_accuracy = 0.6666374
masked_lm_loss = 1.6409736
sentence_order_accuracy = 0.990625
sentence_order_loss = 0.030532131

```

Create pretraining data :

``` bash

python create_pretraining_data.py \
  --input_file=data_vn_bert.raw \
  --output_file=tmp/tf_examples.tfrecord \
  --vocab_file=config/vocab_vn.txt \
  --do_lower_case=False \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345

```

Run pretraining with base config:

``` bash

python run_pretraining.py \
    --input_file=tmp/tf_examples.tfrecord \
    --output_dir=tmp/bert_tmp_bucket/tmp_albert/pretraining_output \
    --albert_config_file=config/albert_base.json \
    --do_train \
    --do_eval \
    --train_batch_size=4096 \
    --eval_batch_size=64 \
    --max_seq_length=256 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=125000 \
    --num_warmup_steps=3125 \
    --use_tpu = True \
    --tpu_name=test-tpu \
    --save_checkpoints_steps=5000

```
## Quick tour with pytorch transformers
### Predict miss word  with AlbertForMaskedLM
```python
from transformers import AlbertForMaskedLM, BertTokenizer
import torch
import unicodedata
from transformers import  tokenization_bert
def _is_punctuation(char):
    """Override this function to keep char '_' """
    cp = ord(char)
    if(char=='_'):
        return False
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
tokenization_bert._is_punctuation = _is_punctuation


MODEL_CLASSES = {
    'albert': (AlbertForMaskedLM,  BertTokenizer)
}
vocab_file = 'albert_base/vocab.txt'
model_class, tokenizer_class = MODEL_CLASSES["albert"]
tokenizer = BertTokenizer(vocab_file=vocab_file,do_lower_case=False,max_len=256)
model = model_class.from_pretrained("albert_base")
model.eval()
text = "trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm đối_tượng chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , chiến_sĩ công_an đã hy_sinh ."
print("Origin text : "+text)
for masked_index in [7,9,12,19,22]:
    text_mask = text.split()
    text_mask[masked_index-1] = "[MASK]"
    text_mask = " ".join(text_mask)
    input_ids = torch.tensor(tokenizer.encode(text_mask, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids)
    predictions = outputs[0]
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print("Mask text : "+text_mask)
    print("Word prediction for [MASK] : "+predicted_token)

```
``` bash

*****  results *****
Origin text : trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm đối_tượng chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , chiến_sĩ công_an đã hy_sinh .
Mask text : trong quá_trình truy_bắt , khống_chế , [MASK] nhóm đối_tượng chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , chiến_sĩ công_an đã hy_sinh .
Word prediction for [MASK] : bắt_giữ
Mask text : trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm [MASK] chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , chiến_sĩ công_an đã hy_sinh .
Word prediction for [MASK] : người
Mask text : trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm đối_tượng chống_đối đặc_biệt [MASK] nêu trên , 3 cán_bộ , chiến_sĩ công_an đã hy_sinh .
Word prediction for [MASK] : nguy_hiểm
Mask text : trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm đối_tượng chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , [MASK] công_an đã hy_sinh .
Word prediction for [MASK] : chiến_sĩ
Mask text : trong quá_trình truy_bắt , khống_chế , bắt_giữ nhóm đối_tượng chống_đối đặc_biệt nguy_hiểm nêu trên , 3 cán_bộ , chiến_sĩ công_an đã [MASK] .
Word prediction for [MASK] : bị_thương

```



### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
