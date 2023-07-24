# Increasing Speed 

* Integrate Flash Attention 2.0 cuda, significant speed up

* Utilize 8BIT Optimizer from BNB, big speed up weakness => bnb isn't compatible with all gpus

* Use a better tokenizer TokenMonster?

* Parallelize the transformer blocks similar to that of [PALMS](https://github.com/conceptofmind/PaLM)

* Look into MPTS config for LION for pretraining, did they use high batch size?