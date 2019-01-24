# NACS

A simple sequence-to-sequence toolkit for doing experiments with SCAN and NACS.

## Installation

First install all requirements, e.g. `pip install -r requirements.txt`

Before use, make sure to compile the extensions:

- `python setup.py build develop`


## Usage

Example on simple data set (enc-dec with attention):

```
python -m seq2seq --mode train --root data/scan/simple --src src --trg trg --train train_wo_valid_random --validation valid.random --test test --batch_size 32 --n_iters -50 --dropout 0.2 --dim 256 --emb_dim 64 --metric exact_match --seed 1 --workdir ./seq2seq_model 
```

Or with more options:

```
python -m seq2seq --mode train --model_type encdec --enc_type birnn --dec_type rnn_att --rnn_type gru --n_enc_layers 1 --n_dec_layers 1 --root data/scan/simple --src src --trg trg --train train_wo_valid_random --validation valid.random --test test --batch_size 32 --n_iters -50 --learning_rate 0.2 --optimizer sgd --clip 5 --eval_every -1 --print_every 100 --dropout 0.2 --dim 256 --emb_dim 64 --metric exact_match --seed 1 --workdir ./seq2seq_model --pass_hidden_state 1 --n_val_examples 5 --use_visdom --unk_src 0 --unk_trg 0 --predict_from_emb 0 --predict_from_ctx 1 --predict_from_dec 1 --dec_input_emb 1 --dec_input_ctx 1
```

### Full list of options

```
usage: __main__.py [-h] --mode {train,test} [--model_type {encdec,model1}]
                   [--enc_type {rnn,birnn}] [--dec_type {rnn,rnn_att}]
                   [--src SRC] [--trg TRG] [--root ROOT] [--train TRAIN]
                   [--validation VALIDATION [VALIDATION ...]] [--test TEST]
                   [--workdir WORKDIR] [--emb_dim EMB_DIM] [--dim DIM]
                   [--batch_size BATCH_SIZE] [--n_enc_layers N_ENC_LAYERS]
                   [--n_dec_layers N_DEC_LAYERS] [--rnn_type {lstm,gru,rnn}]
                   [--use_visdom] [--n_val_examples N_VAL_EXAMPLES]
                   [--min_freq MIN_FREQ] [--tf_ratio TF_RATIO] [--seed SEED]
                   [--dropout DROPOUT] [--weight_decay WEIGHT_DECAY]
                   [--learning_rate LEARNING_RATE]
                   [--learning_rate_decay LEARNING_RATE_DECAY]
                   [--optimizer {adam,sgd}] [--clip CLIP] [--n_iters N_ITERS]
                   [--predict_from_emb PREDICT_FROM_EMB]
                   [--predict_from_ctx PREDICT_FROM_CTX]
                   [--predict_from_dec PREDICT_FROM_DEC]
                   [--dec_input_emb DEC_INPUT_EMB]
                   [--dec_input_ctx DEC_INPUT_CTX] [--save_every SAVE_EVERY]
                   [--print_every PRINT_EVERY] [--plot_every PLOT_EVERY]
                   [--eval_every EVAL_EVERY] [--resume RESUME]
                   [--max_length MAX_LENGTH]
                   [--max_length_train MAX_LENGTH_TRAIN]
                   [--metric {exact_match,accuracy,perplexity,bleu}]
                   [--ckpt CKPT] [--src_vocab SRC_VOCAB]
                   [--trg_vocab TRG_VOCAB] [--src_tags_vocab SRC_TAGS_VOCAB]
                   [--trg_tags_vocab TRG_TAGS_VOCAB] [--pointer POINTER]
                   [--save_heatmaps SAVE_HEATMAPS]
                   [--external_bleu EXTERNAL_BLEU] [--debpe DEBPE]
                   [--unk_src UNK_SRC] [--unk_trg UNK_TRG]
                   [--pass_hidden_state PASS_HIDDEN_STATE]
                   [--scan_normalize SCAN_NORMALIZE]
                   [--num_composed_commands NUM_COMPOSED_COMMANDS]
                   [--ctx_dim CTX_DIM]
```

## License

NACS is CC-NC licensed, as found in the LICENSE file.
