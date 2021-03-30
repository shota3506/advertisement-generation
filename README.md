# Advertisement Generation

## Usage instructions
### Train
```
python train.py \
--data_file <file_path> \
--spm_target_file <file_path> \
--checkpoint_path <file_path> \
--log_file <file_path> \
```
- ```data_file``` data file for training. csv format.
- ```spm_file``` SentencePiece model file.

### Generate
```
python generate.py \
--data_file <file_path> \
--spm_file <file_path> \
--checkpoint_path <file_path> \
--beam_size <num>
```

- ```data_file``` data file to generate copy. csv format. 
- ```spm_file``` SentencePiece model file.
- ```beam_size``` beam search width.
