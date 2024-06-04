"""All code below is taken directly from https://github.com/shmsw25/FActScore, with minor modifications.

Please cite FactScore if you use this code, or evaluate using this code on the FactScore dataset:
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}

FactScore repository:
https://github.com/shmsw25/FActScore
"""
import json
import numpy as np
import os
import pickle as pkl
import sqlite3
import string
import subprocess
import time
from collections import defaultdict
from tqdm import tqdm
from typing import List, Optional, Tuple

from linguistic_calibration.constants import FACTSCORE_CACHE_PATH

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256


def download_file(_id, dest, cache_dir):
    if os.path.exists(dest) or os.path.exists(os.path.join(cache_dir, dest)):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if os.path.exists(dest.replace(".zip", "")) or os.path.exists(os.path.join(cache_dir, dest.replace(".zip", ""))):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [tokenizer.decode(tokens) for tokens in passages if
                        np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(
            results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results


class Retrieval(object):
    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None):
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query],
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1

        return self.cache[cache_key]


class FactScorer(object):
    def __init__(self,
                 data_dir=FACTSCORE_CACHE_PATH,
                 cache_dir=FACTSCORE_CACHE_PATH,
                 batch_size=256):
        self.db = {}
        self.retrieval = {}
        self.batch_size = batch_size  # batch size for retrieval
        self.openai_key = os.environ.get("OPENAI_API_KEY", None)
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", None)

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def save_cache(self):
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

    def construct_prompts_with_retrieval(
        self,
        topics: List[str],
        atomic_facts: List[List[Tuple[str, str]]],
        model_name: str = 'claude-2.0',
        knowledge_source: Optional[str] = None,
        verbose: bool = True,
    ):
        # TODO(@nband): support other fact checkers
        assert model_name == 'claude-2.0', (
            'TODO: support other fact checkers. '
            'Currently our prompt construction uses Claude format (human / assistant).')

        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"

        if verbose:
            topics = tqdm(topics)

        prompts = []
        n_prompts_per_topic = []
        data_to_return = defaultdict(list)

        for topic, facts in zip(topics, atomic_facts):
            data_to_return['entity'].append(topic)
            data_to_return['generated_answer'].append(facts)
            n_prompts = 0
            answer_interpretation_prompts = []
            for atom, atom_type in facts:
                assert atom_type in {"Objective", 'Direct', 'Numerical Uncertainty', 'Linguistic Uncertainty'}
                atom = atom.strip()
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                definition = "Human: Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(
                        psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "\n\n{}\n\nInput: {} True or False?\nOutput:\n\nAssistant:".format(definition.strip(), atom.strip())

                # Flattened list of prompts
                prompts.append(prompt)

                answer_interpretation_prompts.append(prompt)

                # But track how many there are per topic/entity
                n_prompts += 1

            n_prompts_per_topic.append(n_prompts)
            data_to_return['interpretation_prompt'].append(answer_interpretation_prompts)

        return prompts, n_prompts_per_topic, data_to_return
