import os
import json

import numpy as np
import pandas as pd

import torch
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from mpstemmer import MPStemmer
from transformers import BertTokenizer, BertModel
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.tokenize import RegexpTokenizer

from others.logging import logger, init_logger

MAX_TOKENS = 50

def remove_stop_words(doc):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    res = stopword.remove(doc)
    return res

def stemming(doc):
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    # res = stemmer.stem(doc)
    stemmer = MPStemmer()
    res = stemmer.stem_kalimat(doc)
    return res

def tokenize(docs):
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    new_docs = docs.copy()
    for idx in range(len(docs)):
        new_docs[idx] = docs[idx].lower()  # Convert to lowercase.
        new_docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
        
    return new_docs

def preprocess(args):
    """Preprocess all documents before using them as topic models inputs """
    logger.info("Listing train, valid, test files...")
    
    train_files = []
    val_files = []
    test_files = []
    for file in os.listdir(args.source_path):
        if "bert.pt" in file and "train" in file:
            train_files.append(args.source_path + "/" + file)
        elif "bert.pt" in file and "valid" in file:
            val_files.append(args.source_path + "/" + file)
        elif "bert.pt" in file and "test" in file:
            test_files.append(args.source_path + "/" + file)

    logger.info("Sorting train, valid, test files...")
    
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    test_files = sorted(test_files)

    logger.info("Loading train, valid, test data...")
    
    train_docs = []
    val_docs = []
    test_docs = []
    
    i = 0
    for file in train_files:
        print(f"Loading data train {i}...")
        bert_data = torch.load(file)
        for data in bert_data:
            src = " ".join(data['src_txt'])
            train_docs.append(src)
        i = i + 1
    
    i = 0
    for file in val_files:
        print(f"Loading data val {i}...")
        bert_data = torch.load(file)
        for data in bert_data:
            src = " ".join(data['src_txt'])
            val_docs.append(src)
        i = i + 1

    i = 0
    for file in test_files:
        print(f"Loading data test {i}...")
        bert_data = torch.load(file)
        for data in bert_data:
            src = " ".join(data['src_txt'])
            test_docs.append(src)
        i = i + 1

    logger.info("Preprocess the documents...")
    
    proc_train_docs = []
    proc_val_docs = []
    proc_test_docs = []

    logger.info("Preprocess the train docs...")
            
    # Preprocess only removing stop words
    # If we also use stemming, the topic will be non-sense
    for doc in train_docs:
        doc = remove_stop_words(doc)
        proc_train_docs.append(doc)
    
    logger.info("Preprocess the valid docs...")
    
    for doc in val_docs:
        doc = remove_stop_words(doc)
        proc_val_docs.append(doc)
    
    logger.info("Preprocess the test docs...")
    
    for doc in test_docs:
        doc = remove_stop_words(doc)
        proc_test_docs.append(doc)
        
    if args.model == "lda":
        proc_train_docs = tokenize(proc_train_docs)
        proc_val_docs = tokenize(proc_val_docs)
        proc_test_docs = tokenize(proc_test_docs)
        
    return train_files, val_files, test_files, proc_train_docs, proc_val_docs, proc_test_docs

def train_lda(args, proc_train_docs, proc_val_docs, proc_test_docs):
    logger.info("Creating dictionary and corpus...")
    
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(proc_train_docs)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in proc_train_docs]
    val_bow = [dictionary.doc2bow(doc) for doc in proc_val_docs]
    test_bow = [dictionary.doc2bow(doc) for doc in proc_test_docs]
    
    # Set training parameters.
    num_topics = 200
    chunksize = 5000
    passes = 1
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    logger.info("Begin topic modeling LDA...")
    
    topic_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    topic_model.save(f'{args.topic_model_path}/lda.model')

    return topic_model, corpus, val_bow, test_bow

def train_bertopic(args, proc_train_docs):
    TOP_N_WORDS = args.top_n_words

    # Calculate embeddings
    logger.info("Calculating embeddings...")
    if args.sbert == "id":
        embedding_model = SentenceTransformer("denaya/indoSBERT-large")
    else:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(proc_train_docs, show_progress_bar=True)
    
    # Reduce dimension
    logger.info("Initialize UMAP...")
    umap_model = UMAP(n_neighbors=15, n_components=args.umap_components, min_dist=0.0, metric='cosine', random_state=42)

    # Clustering
    logger.info("Initialize HDBSCAN...")
    hdbscan_model = HDBSCAN(min_cluster_size=args.min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # Vectorizer
    logger.info("Initialize vectorizer...")
    vectorizer_model = CountVectorizer(ngram_range=(1, 2))

    # Representation model
    logger.info("Initialize representation model...")
    representation_model = MaximalMarginalRelevance(diversity=0.2)

    logger.info("Begin topic modeling...")

    topic_model = BERTopic(
      # Pipeline models
      embedding_model=embedding_model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer_model,
      representation_model=representation_model,
        
      # Hyperparameters
      top_n_words=TOP_N_WORDS,
      verbose=True
    )

    topics, probs = topic_model.fit_transform(proc_train_docs, embeddings)
    topic_model.save(args.topic_model_path, 
                     serialization="safetensors", 
                     save_ctfidf=True, 
                     save_embedding_model=embedding_model)

    return topic_model

def get_topic_tok_dist_lda(topic_model):
    """Get topics T (all topics for all tokens)"""
    
    logger.info("Acquiring the topic-token distribution (T) ...")
    
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    vocab = tokenizer.get_vocab()

    lda_topics = model.print_topics(num_topics=num_topics)
    topics_raw = {}
    for i in range(len(lda_topics)):
        topic_id, topic_components = lda_topics[i]
        # map_topic_id_to_index[topic_id] = i
        
        topic_components = topic_components.split(" + ")
        new_components = []
        for comp in topic_components:
            score, word = comp.split("*")
            word = word.replace('"', '')
            new_components.append((word, float(score)))
        topics_raw[i] = new_components

    # Convert words into tokens
    # Ex: {0: [('korea', 0.1), ('utara', 0.2)]}
    # --> {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2)]}
    topics_tokenized = {}
    for topic, words in topics_raw.items():
        if topic < 0:
            continue
        topics_tokenized[topic] = []    
        for word, score in words:
            tokens = tokenizer.tokenize(word)
            for tok in tokens:
                topics_tokenized[topic].append((tok, score))
    
    # Clean topics that have same tokens inside it and assign the highest score
    # Ex: {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2), ('ko', 0.01)]}
    # --> {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2)]}
    topics_tokenized_clean = {}
    for topic, tokens in topics_tokenized.items():
        seen_tok = []
        topics_tokenized_clean[topic] = []
        for tok, score in tokens:
            if tok in seen_tok:
                continue
            else:
                seen_tok.append(tok)
                topics_tokenized_clean[topic].append((tok, score))

    topics_T = []
    for topic in topics_tokenized_clean:
        li = []
        for token_tuple in topics_tokenized_clean[topic]:
            token, score = token_tuple
            li.append((token, score))
    
        if len(li) < MAX_TOKENS:
            add_li = [('[PAD]', 0)for _ in range(MAX_TOKENS - len(li))]
            li = li + add_li
            
        topics_T.append(li)

    return topics_T

def get_topic_tok_dist(topic_model):
    """Get topics T (all topics for all tokens)"""
    
    logger.info("Acquiring the topic-token distribution (T) ...")
    
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
    vocab = tokenizer.get_vocab()

    # Format is dictionary with key = topic number and value = list of (word, score)
    # Ex: {0: [('film', 0.1), ('sutradara', 0.2)]}
    topics_raw = topic_model.get_topics()

    # Convert words into tokens
    # Ex: {0: [('korea', 0.1), ('utara', 0.2)]}
    # --> {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2)]}
    topics_tokenized = {}
    for topic, words in topics_raw.items():
        if topic < 0:
            continue
        topics_tokenized[topic] = []    
        for word, score in words:
            tokens = tokenizer.tokenize(word)
            for tok in tokens:
                topics_tokenized[topic].append((tok, score))
    
    # Clean topics that have same tokens inside it and assign the highest score
    # Ex: {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2), ('ko', 0.01)]}
    # --> {0: [('ko', 0.1), ('##rea', 0.1), ('utara', 0.2)]}
    topics_tokenized_clean = {}
    for topic, tokens in topics_tokenized.items():
        seen_tok = []
        topics_tokenized_clean[topic] = []
        for tok, score in tokens:
            if tok in seen_tok:
                continue
            else:
                seen_tok.append(tok)
                topics_tokenized_clean[topic].append((tok, score))

    # We want to make it small
    # The original ones has topic_dist with the dimension 5 x 30521 --> too large
    # We want the 30521 is processed in the model
    # Original = 5 (topics) x 30521 (vocab_size)
    # We want to make it like 5 (topics) x 100 (token for each topic)

    topics_T = []
    for topic in topics_tokenized_clean:
        li = []
        for token_tuple in topics_tokenized_clean[topic]:
            token, score = token_tuple
            li.append((token, score))
    
        if len(li) < MAX_TOKENS:
            add_li = [('[PAD]', 0)for _ in range(MAX_TOKENS - len(li))]
            li = li + add_li
            
        topics_T.append(li)

    return topics_T

def save_topic_dist(args, corpus, files, topic_distr, topics_T):
    data_index = 0
    is_scoring = args.is_scoring
    
    for file in files:
        filename = file.split("/")[-1]
    
        # Loop only for corpus files
        if "bert.pt" in file and corpus in file:
            print(f"Processing topic dist in {filename}...")
            bert_data = torch.load(file)  # inside each files contains 2000 data
    
            # Loop through each file
            for i in range(len(bert_data)):
                if args.model == "lda":
                    d = topic_distr[i][:args.n_topics]
                else:
                    d = {} 
                    # Find related topics
                    # Loop through topic_distr to get corresponding topic for each doc
                    for j in range(len(topic_distr[data_index])):
                        if topic_distr[data_index][j] > 0:
                            d[j] = topic_distr[data_index][j]
                    d = sorted(d.items(), key=lambda x:x[1], reverse=True)[:args.n_topics]  # top 5 topic --> [(topic_id, score), (topic_id, score)]
    
                bert_data[i]['topic_dist'] = []
                distribution_over_words = []
                print("Length d:", len(d))
                # We make it smaller like this: K x MAX_TOKENS
                # [{token: token_score, token: token_score, ..., MAX_TOKENS},
                # {token: token_score, token: token_score, ..., MAX_TOKENS},
                # {token: token_score, token: token_score, ..., MAX_TOKENS}]
                if len(d) > 0:
                    # Assign topic distribution over words 
                    for item in d:
                        topic_id = item[0]
                        topic_score = item[1]                    
                        if is_scoring:
                            topics_T_scored = []
                            for token_tuple in topics_T[topic_id]:
                                new_tuple = (token_tuple[0], token_tuple[1] * topic_score) # multiply by the contribution score of the topic
                                topics_T_scored.append(new_tuple)
                            distribution_over_words.append(topics_T_scored) 
                        else:
                            distribution_over_words.append(topics_T[topic_id])
                
                # Set the dimension to be equal across documents
                empty_topic = [('[PAD]', 0) for _ in range(MAX_TOKENS)]
                distribution_over_words = distribution_over_words + [empty_topic] * (args.n_topics - len(distribution_over_words))
                
                bert_data[i]['topic_dist'] = distribution_over_words
                data_index = data_index + 1
            
            torch.save(bert_data, f"{args.save_path}/{filename}")

def get_topic_dist(args):
    # Initialize the logger and log the arguments
    init_logger(args.log_file)
    logger.info(f"Topic embedding model: {args.model}")
    logger.info(f"Topic embedding scored: {args.is_scoring}")
    
    # Determine the device to use (CPU or GPU)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device %s' % device)

    train_files, val_files, test_files, \
    proc_train_docs, proc_val_docs, proc_test_docs = preprocess(args)

    topics_T = []
    train_topic_distr = []
    val_topic_distr = []
    test_topic_distr = []
    
    if args.model == "lda":
        topic_model, corpus, val_bow, test_bow = train_lda(args, proc_train_docs, proc_val_docs, proc_test_docs)
        
        # Approximate topic distribution for each doc    
        for bow in corpus:
            distr = topic_model.get_document_topics(bow, minimum_probability=0)
            distr.sort(key=lambda x: x[1], reverse=True)
            train_topic_distr.append(distr[:5])

        for bow in val_bow:
            distr = topic_model.get_document_topics(bow, minimum_probability=0)
            distr.sort(key=lambda x: x[1], reverse=True)
            val_topic_distr.append(distr[:5])

        for bow in test_bow:
            distr = topic_model.get_document_topics(bow, minimum_probability=0)
            distr.sort(key=lambda x: x[1], reverse=True)
            test_topic_distr.append(distr[:5])
            
        topics_T = get_topic_tok_dist_lda(topic_model)
        
    else:
        topic_model = train_bertopic(args, proc_train_docs)
         
        # `topic_distr` contains the distribution of topics in each document
        train_topic_distr, _ = topic_model.approximate_distribution(proc_train_docs)
        val_topic_distr, _ = topic_model.approximate_distribution(proc_val_docs)
        test_topic_distr, _ = topic_model.approximate_distribution(proc_test_docs)

        topics_T = get_topic_tok_dist(topic_model)
    
    save_topic_dist(args, "train", train_files, train_topic_distr, topics_T)
    save_topic_dist(args, "valid", val_files, val_topic_distr, topics_T)
    save_topic_dist(args, "test", test_files, test_topic_distr, topics_T)
