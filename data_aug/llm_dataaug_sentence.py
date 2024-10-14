PROMPT = """
You are an AI assistant tasked with generating training data for fine-tuning a Named Entity Recognition (NER) model. The model will be trained to identify location mentions (LOC) in disaster-related sentences. Your task is to generate a labeled sentence similar to a template with proper tagging for location mentions, and ensure the text is cleaned and formatted according to specific guidelines.

Here's the template sentence to base your generation on:
<template_sentence>
{{TEMPLATE_SENTENCE}}
</template_sentence>

The location(s) mentioned in the template sentence:
<template_location>
{{TEMPLATE_LOCATION}}
</template_location>

Generate a new disaster-related sentence following these guidelines:
1. Use the provided template as inspiration.
2. Include zero, one or more locations (countries, cities, or regions) where the disaster or relief efforts are happening.
3. Ensure the sentence is coherent and relates to a disaster scenario.
4. The sentence should be different from the template but maintain a similar structure and theme.

Present your output in the following format:
<output>
<sentence>
[Your generated sentence goes here]
</sentence>
<location>
[List of location(s) mentioned in your generated sentence, in JSON array format]
</location>
</output>

Here's an example of a correctly formatted output:
<output>
<sentence>
Japanese pop star Yuki Tanaka donated ¥10 million for earthquake recovery efforts in Nepal
</sentence>
<location>
["Nepal"]
</location>
</output>

Now, generate a new disaster-related sentence based on the template provided earlier. Do not add any additional comments or explanations outside the <output> tags.
"""

# 1. Generate the sentence.
# 2. Clean and format the text.
# , apply the NER tags, clean the text, and present the formatted output

from langchain_ollama import ChatOllama
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import uuid, os
import json


llm = ChatOllama(
    model="llama3.1",
    # model="qwen2.5",
    temperature=0, # 0.3
    # num_gpu=2,
    # top_k=10,
    # top_p=0.7
    # other params...
)


def extract_ner_tag(sentence: str):
    if "<output>" in sentence and "</output>" in sentence:
        return sentence
    print(sentence)
    return ""


def make_prediction(raw):
    sentence, ner_tags= raw
    messages = [
        ("human", PROMPT.format().format(TEMPLATE_SENTENCE=sentence, TEMPLATE_LOCATION=ner_tags)),
    ]
    ai_msg = llm.invoke(messages)
    return extract_ner_tag(ai_msg.content)
    # print(ai_msg.content)


# def load_bilou_file(path):
def read_bilou_file(file_path):
    def prepare(sentence: dict[str, list[str]]):
        tokenized_text, ners = sentence["tokenized_text"], sentence["ner"]
        return " ".join(tokenized_text), [
            " ".join(tokenized_text[ner[0]: ner[1] + 1]) for ner in ners
        ]
    sentences = json.load(open(file_path,))
    sentences = list(map(prepare, sentences))
    return sentences



evaluation = read_bilou_file("LMRData/train_location.json")

texts = evaluation

thr = ThreadPoolExecutor(3)
folder = "data_augment_qwen"
os.makedirs(folder, exist_ok=True)

bs = 100
sentences_batch = [texts[i : i + bs] for i in range(0, len(texts), bs)]


def save_file(data: list[str], pos):
    with open(os.path.join(folder, str(pos) + "-" + str(uuid.uuid4())), "w") as f:
        f.write("\n\n".join(data))


for pos, texts in tqdm(enumerate(sentences_batch), total=len(sentences_batch)):
    results = list(tqdm(thr.map(make_prediction, texts), total=len(texts)))
    save_file(results, pos)
