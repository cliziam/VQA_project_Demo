from __future__ import print_function

import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
import json
from matplotlib.patches import Rectangle
from PIL import Image
from PIL import ImageDraw

state_dict_path = "./saved_models/glove300/model.pth"  # path of the model weights
qid_to_aid_path = (
    "./assets/qid_to_aid.json"  # map id of the question -> id of the answer (label)
)
qid_to_question_path = (
    "./assets/qid_to_question.json"  # map id of the question -> question
)
aid_to_answer_path = "./assets/aid_to_answer.json"  # map label of the answer -> answer
iid_to_qids_path = "./assets/iid_to_qids.json"  # map image id -> list of questions

question_to_answer_path = "./question_to_answer.json"

EPS = 1e-7


def plot_variance(data, title):
    bins = 100
    hist = torch.histc(data, bins=bins)
    mean = round(torch.mean(data).item(), 3)
    std = round(torch.std(data).item(), 3)

    # Printing the histogram of tensor
    fig = plt.figure(figsize=(5, 3))  # (width, height)
    x = range(bins)
    plt.bar(x, hist, align="center")
    plt.title(title + "\nmean=" + str(mean) + "\nstd=" + str(std))
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()
    fig.clear()
    plt.close()


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real - expected) < EPS).all(), "%s (true) vs %s (expected)" % (
        real,
        expected,
    )


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, "jpg")
    img_ids = set()
    for img in images:
        img_id = int(img.split("/")[-1].split(".")[0].split("_")[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print("%s is not initialized." % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, "w")
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=""):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append("%s %.6f" % (key, np.mean(vals)))
        msg = "\n".join(msgs)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        print(msg)


# load the weights of our pretrained model
def load_weights(model):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()


# save the answers of the model for the first 'num_batches' batches in a json file
# remember to pass entry["question_id"] in the get_item method of the VQAFeatureDataset
def save_answers(model, dataloader, num_batches):
    data = {}
    batch_counter = 0
    for v, b, q, a, qid in tqdm(dataloader, leave=False):
        if batch_counter == num_batches:
            break
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        pred = model(v, b, q, None)
        s = nn.Softmax(dim=1)
        results = (torch.argmax(s(pred), axis=1)).tolist()
        """
        top = torch.topk(s(pred), k=3) # if we want to take the top k asnwers
        print(top[0][132]) # the first index is for the probability distribution
        print(top[1][132]) # the second one is for the labels
        """
        for i in range(len(qid)):
            data[qid[i].item()] = results[i]
        batch_counter += 1
    with open(question_to_answer_path, "w") as j:
        json.dump(data, j)


# given a question id in input, it returns the answer of the model
def get_answer(question_id):
    with open(qid_to_aid_path) as f1:
        qid_to_aid = json.load(f1)
    with open(aid_to_answer_path) as f2:
        aid_to_answer = json.load(f2)
    with open(qid_to_question_path) as f3:
        qid_to_question = json.load(f3)
    try:
        print(qid_to_question[question_id])
        print(aid_to_answer[str(qid_to_aid[question_id])])
    except:
        print("there are no answers for this question")


# given the id of an image, it returns the questions that refer to that image and their id
def get_questions_from_image(image_id):
    with open(iid_to_qids_path) as f1:
        iid_to_qids = json.load(f1)
    with open(qid_to_question_path) as f2:
        qid_to_question = json.load(f2)
    try:
        questions = []
        for qid in iid_to_qids[image_id]:
            questions.append((qid_to_question[str(qid)], str(qid)))
        print("The list of questions for this image is the following:\n", questions)
    except:
        print("There are no questions for this image ID")


def get_question_from_id(question_id):
    with open(qid_to_question_path) as f:
        qid_to_question = json.load(f)
    print(qid_to_question[question_id])


def attention_analysis(model, dataset, idx):
    with torch.no_grad():
        v, q_tok, q_str, bb, img_id, gt = dataset[idx]
        v = v.unsqueeze(0).cuda()
        q_tok = q_tok.unsqueeze(0).cuda()
        pred, att = model(v, None, q_tok, None, attention_output=True)
        pred = dataset.label2ans[pred.squeeze().argmax().item()]

    # Display the image and the bounding box
    img = Image.open("../val2014/" + str(img_id) + ".jpg")
    ax = plt.gca()

    im_a = Image.new("L", img.size, 100)  # gray mask for semi-transparency
    draw = ImageDraw.Draw(im_a)

    for box in bb[torch.topk(att.squeeze(), 1)[1].tolist()]:  # can vary k
        starting = (box[0], box[1])
        width = box[2] - box[0]
        height = box[3] - box[1]

        draw.rectangle(box.tolist(), fill=255)
        # Create a Rectangle patch and add it to the Axes
        rect = Rectangle(
            starting, width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    im_rgba = img.copy()
    im_rgba.putalpha(im_a)  # set transparency

    plt.imshow(im_rgba)  # show image
    plt.axis("off")
    plt.title("Q: " + q_str + "\nGT: " + gt + "    Our: " + pred)
    plt.show()

    return


def tokenize_single(dataset, question, max_length=14):
    tokens = dataset.dictionary.tokenize(question, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        # pad in front of the sentence
        padding = [dataset.dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    assert len(tokens) == max_length, "length not corresponding"
    return torch.tensor(tokens)


def reasoning_analysis(model, dataset, img_id, question, gt):
    v = torch.from_numpy(dataset.features[dataset.img_id2idx[img_id]])
    bb = torch.from_numpy(dataset.bboxes[dataset.img_id2idx[img_id]])
    q_tok = tokenize_single(dataset, question)

    with torch.no_grad():
        v = v.unsqueeze(0).cuda()
        q_tok = q_tok.unsqueeze(0).cuda()
        pred, att = model(v, None, q_tok, None, attention_output=True)
        top_preds_values, top_preds_labels = torch.topk(pred.squeeze(), 5)
        top_preds_values = torch.nn.functional.softmax(
            top_preds_values, dim=-1
        ).tolist()
        top_preds_labels = [dataset.label2ans[idx] for idx in top_preds_labels]
        best_pred = dataset.label2ans[pred.squeeze().argmax().item()]

    # Display the image and the bounding box
    img = Image.open("../val2014/" + str(img_id) + ".jpg")
    ax = plt.gca()

    im_a = Image.new("L", img.size, 100)  # gray mask for semi-transparency
    draw = ImageDraw.Draw(im_a)

    for box in bb[torch.topk(att.squeeze(), 1)[1].tolist()]:  # can vary k
        starting = (box[0], box[1])
        width = box[2] - box[0]
        height = box[3] - box[1]

        draw.rectangle(box.tolist(), fill=255)
        # Create a Rectangle patch and add it to the Axes
        rect = Rectangle(
            starting, width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    im_rgba = img.copy()
    im_rgba.putalpha(im_a)  # set transparency

    plt.imshow(im_rgba)  # show image
    plt.axis("off")
    plt.title("Q: " + question + "\nCommonsense: " + gt + "    Our: " + best_pred)
    plt.show()
    plt.close()

    plt.bar(top_preds_labels, top_preds_values)
    plt.title("Probability distribution of top-5 answers returned by the model")
    plt.show()

    return
