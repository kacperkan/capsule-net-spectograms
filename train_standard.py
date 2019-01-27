from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import tqdm
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import constants
import metrics
import transformers
import utils
from constants import logger
from data_standard import TrainLoader, TestLoader, generate_submission, generate_submission_with_class_confidences
from model_standard import Net

use_cuda = torch.cuda.is_available()


def get_timestamp() -> str:
    return datetime.now().strftime("%d_%m_%y_%H_%M")


def get_image_reconstruction(images, reconstructions):
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    to_show = torch.cat([images, reconstructions], dim=-1)
    x = vutils.make_grid(to_show, normalize=True, scale_each=True)
    return x


def load_train_data():
    return np.load(constants.X_DATA_PATH).astype(np.float32), np.load(constants.Y_DATA_PATH).astype(np.float32)


def train(path: str, batch_size: int, epochs: int, kfolds: int):
    path = Path(path)
    path.mkdir(exist_ok=False)
    x_data, y_data = load_train_data()
    kfolder = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=0)
    validatus_totalus_predus = []
    validatus_totalus_trus = []
    for fold_num, (train_index, val_index) in enumerate(kfolder.split(x_data, np.argmax(y_data, axis=1))):
        logger.info("Fold num: {}/{}".format(fold_num + 1, kfolds))
        model_path = path / "model_{}.pth".format(fold_num)
        best_fscore = -np.inf

        x_train, y_train = x_data[train_index], y_data[train_index]
        x_valid, y_valid = x_data[val_index], y_data[val_index]

        transforms = Compose([
            transformers.RandomWhiteNoise(-0.2, 0.2),
            transformers.MultiplicativeNoise(0.8, 1.2)
        ])
        train_dataset = DataLoader(TrainLoader(
            x_train, y_train, True, transforms
        ), batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
        valid_dataset = DataLoader(TrainLoader(
            x_valid, y_valid, False, None
        ), batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

        test_dataset = DataLoader(TestLoader(
            constants.TEST_DATA_PATH, None
        ), batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

        model = Net()

        logger.info("Model parameters: %d" % model.get_num_parameters())
        criterion = nn.CrossEntropyLoss()
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
        train_writer = SummaryWriter((path / "train_{}".format(fold_num)).as_posix())
        valid_writer = SummaryWriter((path / "valid_{}".format(fold_num)).as_posix())

        step = 0
        for epoch in range(epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, epochs))

            # Training
            model.train()
            pbar = tqdm.tqdm(total=len(train_dataset))
            for i, data in enumerate(train_dataset):
                images, labels = data
                images, labels = utils.convert_to_cuda(images, labels)
                optimizer.zero_grad()

                logits, classes, softmax = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                pred_labels, true_labels = classes, labels
                if use_cuda:
                    pred_labels = pred_labels.cpu()
                    true_labels = true_labels.cpu()

                acc = metrics.accuracy(pred_labels.numpy(), true_labels.numpy())
                fscore = metrics.fscore(pred_labels.numpy(), true_labels.numpy())

                train_writer.add_scalar("loss", loss.item(), global_step=step)
                train_writer.add_scalar("acc", acc, global_step=step)
                train_writer.add_scalar("fscore", fscore, global_step=step)

                step += 1
                pbar.update(1)
                pbar.set_postfix(OrderedDict({
                    "loss": "%.4f" % loss.item(),
                    "acc": "%.4f" % acc,
                    "f1": "%.4f" % fscore
                }))

            pbar.close()
            # Evaluation
            logger.info("Validating...")
            model.eval()
            all_pred_labels = []
            all_true_labels = []
            all_losses = []
            for i, data in enumerate(tqdm.tqdm(valid_dataset)):
                images, labels = data
                if use_cuda:
                    images, labels = utils.convert_to_cuda(images, labels)
                logits, classes, softmax = model(images)
                loss = criterion(logits, classes)

                pred_labels, true_labels = classes, labels
                pred_labels, true_labels = utils.convert_from_cuda(pred_labels, true_labels)

                all_losses.append(loss.item())
                all_true_labels.append(true_labels)
                all_pred_labels.append(pred_labels)

                validatus_totalus_trus.append(true_labels)
                validatus_totalus_predus.append(pred_labels)

            all_pred_labels = torch.cat(all_pred_labels, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            all_pred_labels, all_true_labels = utils.convert_from_cuda(all_pred_labels, all_true_labels)

            acc = metrics.accuracy(all_pred_labels, all_true_labels)
            fscore = metrics.fscore(all_pred_labels, all_true_labels)
            loss = float(np.mean(all_losses))

            if fscore > best_fscore:
                best_fscore = fscore
                torch.save(model.cpu().state_dict(), model_path.as_posix())
                model.cuda()

            valid_writer.add_scalar("loss", loss, global_step=step)
            valid_writer.add_scalar("acc", acc, global_step=step)
            valid_writer.add_scalar("fscore", fscore, global_step=step)
            logger.info("Valid scores: \nLoss: %.4f\nAcc: %.4f\nF1: %.4f" % (loss, acc, fscore))

        # Prediction
        logger.info("Predicting...")
        model = Net()
        model.eval()
        model.load_state_dict(torch.load(model_path.as_posix()))
        model = model.cuda()
        test_predictions = []
        for data in tqdm.tqdm(test_dataset):
            images = data.cuda()
            logits, classes, softmax = model(images)
            softmax = softmax.cpu()
            test_predictions.append(softmax)
        test_predictions = torch.cat(test_predictions, dim=0)
        test_predictions = test_predictions.cpu().detach().numpy()
        predicted_classes = list(np.argmax(test_predictions, axis=1).astype(np.int).tolist())

        output_path_submission = path / "{}_{:.5f}.csv".format(get_timestamp(), best_fscore)
        output_path_confidence = path / "{}_{:.5f}_{}.csv".format(get_timestamp(), best_fscore, fold_num)
        generate_submission(predicted_classes, output_path_submission)
        generate_submission_with_class_confidences(test_predictions, output_path_confidence)

    validatus_totalus_trus = torch.cat(validatus_totalus_trus, dim=0)
    validatus_totalus_predus = torch.cat(validatus_totalus_predus, dim=0)
    validatus_totalus_trus, validatus_totalus_predus = utils.convert_from_cuda(validatus_totalus_trus,
                                                                               validatus_totalus_predus)
    totalus_fscore = metrics.fscore(validatus_totalus_predus, validatus_totalus_trus)
    totalus_acc = metrics.accuracy(validatus_totalus_predus, validatus_totalus_trus)
    with open((path / 'final_res.txt').as_posix(), 'w') as f:
        f.write("""
        Fscore: {}
        Accuracy: {}
        """.format(totalus_fscore, totalus_acc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--kfolds", type=int, default=8)
    args = parser.parse_args()

    train(args.output_path, args.batch_size, args.epochs, args.kfolds)
