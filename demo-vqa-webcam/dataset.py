from __future__ import print_function
import os
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's")
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, "wb"))
        print("dictionary dumped to %s" % path)

    @classmethod
    def load_from_file(cls, path):
        print("loading dictionary from %s" % path)
        word2idx, idx2word = cPickle.load(open(path, "rb"))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],  # image_id = id in the COCO dataset
        "image": img,  # image = internal ordered id given by the authors, useful to access self.features and self.spatials of the VQAdataset
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """

    question_path ="..\\demo-vqa-webcam\\BUTDA\\data\\v2_OpenEnded_mscoco_val2014_questions.json"
    questions = sorted(
        json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    )
    answer_path = "..\\demo-vqa-webcam\\BUTDA\\data\\cache\\val_target.pkl"
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["question_id"])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question["question_id"], answer["question_id"])
        utils.assert_eq(question["image_id"], answer["image_id"])
        img_id = question["image_id"]
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(
        self, name, dictionary, isBERT=False, dataroot="data", return_bboxes=False
    ):
        super(VQAFeatureDataset, self).__init__()
        assert name in ["train", "val"]

        ans2label_path = "..\\demo-vqa-webcam\\BUTDA\\data\\cache\\trainval_ans2label.pkl"
        label2ans_path = "..\\demo-vqa-webcam\\BUTDA\\data\\cache\\trainval_label2ans.pkl"
        with open(ans2label_path, "rb") as f:
            self.ans2label = cPickle.load(f)
        with open(label2ans_path, "rb") as f:
            self.label2ans = cPickle.load(f)
        self.num_ans_candidates = len(self.ans2label)
        self.isBERT = isBERT

        self.dictionary = dictionary
        path2="..\\demo-vqa-webcam\\BUTDA\\data\\val36_imgid2idx.pkl"
        with open(path2, "rb") as f:
            self.img_id2idx = cPickle.load(f)
        print("loading features from h5 file")
        h5_path ="..\\demo-vqa-webcam\\BUTDA\\data\\val_VGG.h5"

        print(h5_path)
        #################################################################################
        ######################!!!Comsume  too much  memory!!!############################
        #################################################################################
        # with h5py.File(h5_path, 'r') as hf:
        #     self.features = np.array(hf.get('image_features'))
        #     self.spatials = np.array(hf.get('spatial_features'))
        hf = h5py.File(h5_path, "r")
        self.features = hf.get("image_features")
        #self.spatials = hf.get("spatial_features")
        #self.bboxes = hf.get("image_bb")
        #self.return_bboxes = return_bboxes

        # this hdf5 file contains:
        #   image_bb: bounding box coordinates for each image for each of the 36 regions
        #   image_features: r-cnn features extracted for each image for each of the 36 regions
        #   spatial_features: {scaled_x, scaled_y, scaled_x + scaled_width,
        #       scaled_y + scaled_height, scaled_width, scaled_height} for each image for each of the 36 regions
        #################################################################################

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        if not self.isBERT:
            self.tokenize()  # if we use bert embeddings it's not needed anymore
        self.tensorize()

        self.v_dim = self.features.shape[2]  # length of r-cnn features for a region
        #self.s_dim = self.spatials.shape[2]  # length of spatial features for a region

        #################################################################################

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry["question"], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens

    def tensorize(self):
        for entry in self.entries:
            # not needed if we use bert
            if not self.isBERT:
                question = torch.from_numpy(np.array(entry["q_token"]))
                entry["q_token"] = question

            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = torch.from_numpy(
            self.features[entry["image"]]
        )  # r-cnn features converted to tensor
       # spatials = torch.from_numpy(
        #    self.spatials[entry["image"]]
        #)  # spatial features converted to tensor
        bboxes = torch.from_numpy(
            self.bboxes[entry["image"]]
        )  # spatial features converted to tensor

        ################################################################################

        im_id = entry["image_id"]
        question_tok = entry["q_token"]
        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)  # insert score into one-hot vector

        if self.isBERT:
            question_tok = entry["question"]

        if self.return_bboxes:
            return (
                features,
                question_tok,
                entry["question"],
                bboxes,
                im_id,
                self.label2ans[target.argmax()],
            )
        else:
            return features, question_tok, target

    def __len__(self):
        return len(self.entries)
