import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["WANDB_PROJECT"] = "mscft_ner"
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_NOTEBOOK_NAME"] = "gliner_finetuing.py"

import json

from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments

from utils import seed_everything


def main():
    seed_everything(41)

    location_name = "location"
    labels = [location_name]

    # model = GLiNER.from_pretrained("pretraining/checkpoint-1100")
    # model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    # model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    model = GLiNER.from_pretrained(
        "urchade/gliner_large-v2.1",
        # _attn_implementation="flash_attention_2",
        max_length=2048,
    )
    # model = GLiNER.from_pretrained("knowledgator/gliner-bi-large-v1.0")
    # model = GLiNER.from_pretrained("knowledgator/gliner-bi-llama-v1.0")
    # model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print(model)

    # use it for better performance, it mimics original implementation but it's less memory efficient
    data_collator = DataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True
    )

    # train_dataset = json.load(open("LMRData/train_location.json"))
    train_dataset = json.load(open("data/accepted_data/TrainCleaned.json"))
    for i in train_dataset:
        i["label"] = labels
        for row in i["ner"]:
            row[-1] = location_name
    test_dataset = json.load(open("data/accepted_data/TestCleaned.json"))
    for i in test_dataset:
        i["label"] = labels
        for row in i["ner"]:
            row[-1] = location_name

    print("Dataset Size:", len(train_dataset), len(test_dataset))

    print(train_dataset[:5])

    # calculate number of epochs
    batch_size = 8
    # num_steps = len(train_dataset) // batch_size
    # data_size = len(train_dataset)
    # num_batches = data_size // batch_size
    # num_epochs = max(1, num_steps // num_batches)

    num_epochs = 5

    gradient_accumulation_steps = 1
    save_steps = (
        int(len(train_dataset) * 0.5 / (batch_size * gradient_accumulation_steps))
        // (2)
    ) * 2

    training_args = TrainingArguments(
        run_name="fine_tune_gliner_large",
        output_dir="models",
        learning_rate=1e-6,
        weight_decay=0.0001,
        others_lr=0.5e-6,
        others_weight_decay=0.0001,
        lr_scheduler_type="constant",  # linear cosine
        warmup_ratio=0,  # .1
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=save_steps // 2,
        save_total_limit=10,
        dataloader_num_workers=0,
        report_to="wandb",
        metric_for_best_model="loss",
        loss_reduction="sum",
        ddp_find_unused_parameters=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # logging_steps=250,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(4)]
    )

    trainer.train()

    trainer.evaluate()


if __name__ == "__main__":
    main()
