from configs.config_loader import Config
from torch.utils.data import DataLoader
from loader import TimeSeriesDataset
import pandas as pd
import numpy as np
import torch
import utils
import os
import hydroeval as he

def train_epoch(model, dataset, optimizer, criterion, config: Config):
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    total_loss = 0
    model.train()
    for src, tgt, in dataloader:
        src = src.to(config.device)
        tgt = tgt.to(config.device)
        output = model(src)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, valdataset, testdataset,criterion, config: Config):
    valdataloader = DataLoader(valdataset, batch_size=config.batch_size, shuffle=False)
    testdataloader = DataLoader(testdataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    total_loss = 0
    val_y_pred = []
    val_y_true = []
    with torch.no_grad():
        for src, tgt, in valdataloader:
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            output = model(src)
            loss = criterion(output, tgt)
            total_loss += loss.item()
            val_y_pred.append(output.cpu().numpy())
            val_y_true.append(tgt.cpu().numpy())
    val_y_pred = np.concatenate(val_y_pred, axis=0)
    val_y_pred = val_y_pred * valdataset.std_y.cpu().numpy() + valdataset.mean_y.cpu().numpy()
    val_y_true = np.concatenate(val_y_true, axis=0)
    val_y_true = val_y_true * valdataset.std_y.cpu().numpy() + valdataset.mean_y.cpu().numpy()

    test_y_pred = []
    test_y_true = []
    with torch.no_grad():
        for src, tgt, in testdataloader:
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            output = model(src)
            test_y_pred.append(output.cpu().numpy())
            test_y_true.append(tgt.cpu().numpy())
    test_y_pred = np.concatenate(test_y_pred, axis=0)
    test_y_pred = test_y_pred * testdataset.std_y.cpu().numpy() + testdataset.mean_y.cpu().numpy()
    test_y_true = np.concatenate(test_y_true, axis=0)
    test_y_true = test_y_true * testdataset.std_y.cpu().numpy() + testdataset.mean_y.cpu().numpy()
                
    return total_loss / len(valdataloader), val_y_pred, val_y_true, test_y_pred, test_y_true

def save_checkpoint(model, optimizer, epoch, train_loss, eval_loss, config: Config, val_y_pred, val_y_true, test_y_pred, test_y_true):
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eval_loss': eval_loss,
        'train_loss': train_loss,
        'val_y_pred': val_y_pred,
        'val_y_true': val_y_true,
        'test_y_pred': test_y_pred,
        'test_y_true': test_y_true
    }, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")

def train_full(model, optimizer, criterion, config: Config):
    
    best_eval_loss = float('inf')
    epochs_no_improve = 0
    data = pd.read_csv(config.data_path)
    train_dataset = TimeSeriesDataset(data=data, config=config, t="train")
    eval_dataset = TimeSeriesDataset(data=data, config=config, t="val")
    test_dataset = TimeSeriesDataset(data=data, config=config, t="test")
    train_loss_log = []
    eval_loss_log = []
    utils.save_yaml(config.config_path, os.path.join(config.checkpoint_dir, config.experiment_name + ".yaml"))
    for epoch in range(config.n_epochs):
        train_loss = train_epoch(model, train_dataset, optimizer, criterion, config)
        eval_loss, val_y_pred, val_y_true, test_y_pred, test_y_true = evaluate(model, eval_dataset, test_dataset, criterion, config)
        train_loss_log.append(train_loss)
        eval_loss_log.append(eval_loss)
        loss_df = pd.DataFrame({'train_loss': train_loss_log, 'eval_loss': eval_loss_log})

        loss_df.to_csv(os.path.join(config.checkpoint_dir, "loss.csv"), index=False)
        if epoch % config.save_interval == 0:
            val_nse = he.nse(val_y_pred, val_y_true)
            test_nse = he.nse(test_y_pred, test_y_true)
            print(f"Epoch: {epoch} / Train Loss: {train_loss} / Eval Loss: {eval_loss} / Val NSE: {val_nse} / Test NSE: {test_nse}")
            save_checkpoint(model, optimizer, epoch, train_loss, eval_loss, config,\
                                val_y_pred, val_y_true, test_y_pred, test_y_true)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            save_checkpoint(model, optimizer, epoch, train_loss, eval_loss, config,\
                             val_y_pred, val_y_true, test_y_pred, test_y_true)
            break