from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import TransformerWordEmbeddings
from torch.optim import AdamW
# from transformers.integrations import WandbCallback

columns = {0: "text", 1: "ner"}
corpus: Corpus = ColumnCorpus(
    "data/accepted_data",
    columns,
    train_file="TrainCleaned.txt",
    dev_file="TestCleaned.txt",
    test_file="TestCleaned.txt",
    comment_symbol=None,
)

tag_type = "ner"

tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

# model_name = "WhereIsAI/UAE-Large-V1"
# hidden_size = 1024
# model_name = "nomic-ai/nomic-embed-text-v1.5"
# model_name = "Alibaba-NLP/gte-base-en-v1.5"
model_name = "tner/deberta-v3-large-ontonotes5"
# model_name = "microsoft/deberta-v3-large"
hidden_size = 1024

embeddings = TransformerWordEmbeddings(
    model=model_name,
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
    trust_remote_code=True
)

tagger = SequenceTagger(
    hidden_size=hidden_size,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    dropout=0., # remove dropout after embedding
    tag_format="BIO",
    use_crf=True,
    use_rnn=True,
    rnn_layers=2, # 1
    reproject_embeddings=False, # remove reprojection
)

# flair_pretrained = "flair_finetuning/where_ai1/final-model.pt"
# tagger = SequenceTagger.load(
#     flair_pretrained,
# )

trainer = ModelTrainer(tagger, corpus)

batch_size = 12
trainer.train(
    "flair_finetuning/debertav3",
    learning_rate=1e-5,
    mini_batch_size=batch_size,
    eval_batch_size=batch_size * 2,
    # mini_batch_chunk_size=1,
    max_epochs=4,
    optimizer=AdamW,
    #   scheduler=OneCycleLR,
    embeddings_storage_mode="none",
    weight_decay=0.0001,
    min_learning_rate=1e-8,
    patience=1,
    use_final_model_for_eval=True
)

# load the trained model
# model = SequenceTagger.load("winner/final-model.pt")
# # create example sentence
# sentence = Sentence("I love Southern California and texas")
# # predict the tags
# model.predict(sentence)
# print(sentence)
