import torch
from torch import nn
from torch_geometric.loader import DataLoader
from dataset import *
from model import *

def train(model, optimizer, dataloader, class_weight=None):
    model.train()
    avg_loss = 0.0 # ce for classification, mse for regression
    avg_metric = 0.0 # acc for classification, mae for regression
    n = 0

    loss_func = nn.CrossEntropyLoss(weight=class_weight)

    for data in dataloader:
        optimizer.zero_grad()

        outputs = model(data)
        targets = data.y

        preds = outputs.argmax(dim=1)
        loss = loss_func(outputs, targets)
        metric = (preds == targets).float().mean()
        avg_loss += loss.item()
        avg_metric += metric.item()

        loss.backward()
        optimizer.step()
        n += 1

    avg_loss /= n
    avg_metric /= n

    return avg_loss, avg_metric

def test(model, dataloader, class_weight=None):
    model.eval()
    avg_loss = 0.0
    avg_metric = 0.0
    n = 0

    loss_func = nn.CrossEntropyLoss(weight=class_weight)

    with torch.no_grad():
        for data in dataloader:

            outputs = model(data)
            targets = data.y

            preds = outputs.argmax(dim=1)
            loss = loss_func(outputs, targets)
            metric = (preds == targets).float().mean()
            avg_loss += loss.item()
            avg_metric += metric.item()

            n += 1

    avg_loss /= n
    avg_metric /= n

    return avg_loss, avg_metric

def main(args):
    device = torch.device(args.device)

    #### Datasets ####
    dataset = TpcGraphDataset(args.dataset_train, device=device, nevt=args.nevt, outlier_distance=args.outlier_distance)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    dataset_test = TpcGraphDataset(args.dataset_test, device=device, nevt=args.nevt_val, outlier_distance=args.outlier_distance)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch, shuffle=True)

    class_weights_train = dataset.get_class_weights() if args.enable_balanced_loss else None
    class_weights_test = dataset_test.get_class_weights() if args.enable_balanced_loss else None

    #### Model ####
    model = GraphPointTransformer(in_channels=2, 
                                  out_channels=2,
                                  debug=args.enable_debug, 
                                  dim_model=args.model_layers, 
                                  down_ratio=args.model_down_ratio, k=args.knn, 
                                  undirected=args.undirected, 
                                  reduce=args.reduce, 
                                  self_attention=args.enable_self_attention,
                                  num_heads=args.num_heads)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)


    #### Training ####
    start_epoch = 0
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    nepoch_tot = start_epoch
    for istage, nepoch in tqdm(enumerate(args.nepoch)):
        for epoch in tqdm(range(nepoch)):
            loss, metric = train(model=model, 
                                                 dataloader=dataloader, optimizer=optimizer, 
                                                 class_weight=class_weights_train)
            loss_test, metric_test = test(model=model,
                                                                    dataloader=dataloader_test, 
                                                                    class_weight=class_weights_test)

            print(f'Epoch {nepoch_tot}:')
            print(f'  Train Loss: {loss:.4f}, Train Metric: {metric:.4f}')
            print(f'  Test Loss: {loss_test:.4f}, Test Metric: {metric_test:.4f}')

            if args.save_checkpoints is not None:
                interval = np.abs(args.save_checkpoints)
                if nepoch_tot % interval == interval - 1:
                    if args.save_checkpoints < 0:
                        save_dict = {}
                        save_dict['epoch'] = nepoch_tot
                        save_dict['model_state_dict'] = model.state_dict()
                        save_dict['loss'] = loss
                        save_dict['optimizer_state_dict'] = optimizer.state_dict()
                        save_dict['scheduler_state_dict'] = scheduler.state_dict()

                        torch.save(
                            save_dict, './data/checkpoint_epoch{}_{}.pth'.format(nepoch_tot, args.tag)
                        )
                    file = open('./data/training_result_epoch{}_{}.txt'.format(nepoch_tot, args.tag), 'w')
                    file.write(f'{epoch}  {loss:.4f}  {metric:.4f}  {loss_test:.4f}  {metric_test:.4f}\n')
                    file.close()
        
            scheduler.step()
            nepoch_tot += 1

    torch.save(model, './data/model_{}.pth'.format(args.tag))
