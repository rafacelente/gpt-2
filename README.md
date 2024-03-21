## GPT-2

GPT-2 model built and trained from scratch in pure Pytorch.

### Usage

#### Inference

```
from gpt2 import GPT

gpt = GPT.build(
    max_length=1024,
    from_checkpoint="path/to/pretrained/weights",
)

# alternatively, you may pass as well
# gpt.load_checkpoint("path/to/pretrained/weights") or
# gpt.load_weights_from_hf() -> requires transformers library

gpt.generate(
    prompt="Once upon a time,  ", 
    max_len=128,
    do_sample=False,
    top_k=2,
    repetition_penalty=1.5,
    )
```

#### Training

The default datamodule assumes that the data is in the format of parquet files. The implementation of the Gepeto Dataset is found on `gpt/modules/data/text_dataset.py`.

```
from gpt2 import GPT

gpt = GPT.build(
    max_length=1024,
    from_checkpoint="path/to/pretrained/weights",
)

gpt.load_datamodule(
    data_path=data_paths, # list of paths to parquet files
    batch_size=8,
    train_test_split=0.95,
    max_length=1024
)

gpt.train(
    max_epochs=1,
)
```

The GPT module is built on top of a Lightning Module, so it supports logging with WandB, MLFlow, Tensorboard etc. 

```
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(log_model=False, project="gpt2")
gpt.train(
    max_epochs=1,
    logger=logger,
)

```

### Tips for training

#### Datasets

All of the datasets used for the training runs came from Hugging Face Datasets.

```
REPO_ID = "wikitext"
wiki_text_1 = hf_hub_download(repo_id=REPO_ID, filename="wikitext-103-v1/train-00000-of-00002.parquet", repo_type="dataset", token=HF_TOKEN)
wiki_text_2 = hf_hub_download(repo_id=REPO_ID, filename="wikitext-103-v1/train-00001-of-00002.parquet", repo_type="dataset", token=HF_TOKEN)

REPO_ID = "nampdn-ai/tiny-strange-textbooks"
tiny_strange_1 = hf_hub_download(repo_id=REPO_ID, filename="data_part_1.parquet", repo_type="dataset", token=HF_TOKEN)

data_paths = [tiny_strange_1, wiki_text_1, wiki_text_2]

gpt.load_datamodule(
    data_path=data_paths,
    batch_size=8,
    train_test_split=0.95,
    max_length=1024
)
```

#### On training efficiency

This implementation, since it is mostly for pedagogical purposes, doesn't use some common optimizations such as FlashAttention or KV cache. Training times are usually longer. Training with ~1B tokens took around 8 hours on a A100-80GB. 
