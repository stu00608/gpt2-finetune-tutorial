# GPT2 and BERT Finetune Tutorial

## Dataset

### GPT-2 fine-tune on Imdb dataset
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -zxf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
```

### BERT fine-tune on Spam dataset
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
rm smsspamcollection.zip
```

* Put them into `aclImdb` and `smss` folder in repo's root.
<img width="253" alt="CleanShot 2022-09-06 at 17 20 07@2x" src="https://user-images.githubusercontent.com/40068587/188598278-038aa396-a32a-4f4d-b65b-f538ba530250.png">


## Reference
* [🎱 GPT2 For Text Classification using Hugging Face 🤗 Transformers](https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/)
