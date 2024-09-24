from configs.config_loader import Config
from model import LSTM
from loader import TimeSeriesDataset
from trainer import train_full, evaluate
import argparse
import torch
import glob



def main(arg):
    config = Config(arg)
    model = LSTM(input_size = len(config.input_variables), hidden_size=config.hidden_size, num_layers=config.num_layers, \
                 output_size=config.output_size, dropout=config.dropout, \
                 bidirectional=config.bidirectional).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    train_full(model, optimizer, criterion, config)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="./configs/test.yaml")
    args = argparser.parse_args()
    main(args.config)
    # configs = glob.glob("./configs/*.yaml")
    # configs = [arg for arg in configs if "test" not in arg]
    # for config in configs:
    #     main(config)
    # 사용법: python main.py --config ./configs/test.yaml