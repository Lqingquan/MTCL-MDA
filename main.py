import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed
from trainer import NCLTrainer
from model import NCL, mlp
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.interpolate import interp1d
from util import *
import warnings
warnings.filterwarnings("ignore")
colors = list(mcolors.TABLEAU_COLORS.keys())


def run_single_model(args):
    # configurations initialization
    config = Config(
        model=NCL,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    config['seed'] = 1
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = NCL(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = NCLTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config['show_progress'])
    disease_embedding, miRNA_embedding = model.predict(dataset)
    np.savetxt(fname='data/HMDDv3/disease_embedding.txt', X=disease_embedding, newline='\n', encoding='UTF-8')
    np.savetxt(fname='data/HMDDv3/miRNA_embedding.txt', X=miRNA_embedding, newline='\n', encoding='UTF-8')


def draw(true, out):
    fpr, tpr, thresholds = roc_curve(true.cpu(), out.cpu())
    roc_auc = auc(fpr, tpr)
    x = np.linspace(0, 1, 100)
    f = interp1d(fpr, tpr)
    tpr = f(x)
    fpr = x
    plt.plot(fpr, tpr, color=mcolors.TABLEAU_COLORS[colors[times % 10]], label=f'Fold {times} ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color=mcolors.TABLEAU_COLORS[colors[times % 10]], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}-fold CV'.format(fold))
    plt.legend(loc="lower right")


def train():
    pos_data, neg_data = load_data('')
    pos_list, neg_list = split_data(pos_data, neg_data)
    pos_data = deal_embedding(pos_list)
    neg_data = deal_embedding(neg_list)
    train_data, train_label, test_data, test_label = split_train_test(pos_data, neg_data, fold, times)
    train_data, train_label, test_data, test_label = train_data.cuda(), train_label.cuda(), test_data.cuda(), test_label.cuda()

    # build model
    model = mlp(len(train_data[0]), 64, 32, 1).cuda()
    opt = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.BCELoss().cuda()

    # train
    best_loss = 1
    best_model = model
    for epoch in range(2000):
        model.train()
        out = model(train_data)
        loss = loss_fn(out, train_label.unsqueeze(-1))
        if loss < best_loss:
            best_model = model
        opt.zero_grad()
        loss.backward()
        opt.step()

    # test
    best_model.eval()
    with torch.no_grad():
        out = best_model(test_data)
    AUC = roc_auc_score(test_label.unsqueeze(-1).cpu(), out.cpu())
    temp = torch.tensor(out)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    acc, sen, pre, spe, F1 = calculate_metrics(test_label.cpu(), temp.cpu())
    print("auc:{}, acc:{}, sen:{}, pre:{}, spe:{}, F1:{}".format(AUC, acc, sen, pre, spe, F1))
    draw(test_label, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='D2R', help='datasets')
    parser.add_argument('--config', type=str, default='properties/D2R.yaml', help='config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/NCL.yaml'
    ]
    if args.config is not '':
        args.config_file_list.append(args.config)

    # run_single_model(args)

    fold = 5
    for times in range(fold):
        np.random.seed(1283)
        train()
    plt.show()

