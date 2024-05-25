import argparse
import os
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from simple_transformer import TransformerForClassification
from dataset.uci_sentiment_dataset import uci_sentiment_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Root of training dataset')
    parser.add_argument('--wt-path', type=str, help='Path to save the trained parameters of model',
                        default='./weights/')
    parser.add_argument('--plot-path', type=str, help='Path to save the plot of the metrics in training',
                        default='./plots/')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.wt_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, vocab_size = uci_sentiment_dataloader(args.data_root, mode='train', max_seq_length=512)
    val_loader, _ = uci_sentiment_dataloader(args.data_root, mode='val', max_seq_length=512)

    model = TransformerForClassification(embed_dim=128,
                                         src_vocab_size=vocab_size,
                                         num_classes=2,
                                         seq_len=512,
                                         num_blocks=4)
    
    date_str = datetime.now().strftime('%Y%m%d-%H%M')
    save_dir = os.path.join(args.wt_path, 'simple_transformer', date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join(args.plot_path, 'simple_transformer', date_str)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.to(device)
    model.train()
    best_acc = float('-inf')
    for epoch in range(args.epoch):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch + 1}/{args.epoch}")
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(loss=loss.item())
            logger.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + i)

        print(f'Epoch {epoch + 1} -- Loss: {running_loss / len(train_loader):.3f} Train acc: {correct / total * 100:.3f} %')

        correct = 0
        total = 0
        model.eval()
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch + 1}/{args.epoch}")
        with torch.no_grad():
            for data in pbar:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(acc=correct / total)

        acc = correct / total
        logger.add_scalar('Test/Accuracy', acc, epoch)

        if best_acc < acc:
            print(f"Test accuracy improved from {best_acc:.3f} to {acc:.3f}")
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'best.pth'))

    torch.save(model.state_dict(), os.path.join(save_dir, f'last.pth'))
