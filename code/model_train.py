from matplotlib import pyplot as plt
from model_all import *
# from model_all_onlylitter import *
# from model_no_poi import *
# from model_GCN import *
# from model_nos import *
# from model_baseline_GAT import *
# from model_baseline_GAT_onlylitter import *
# from model_baseline_GCN_onlylitter import *
# from model_baseline_GCN import *
from utils import *

torch.manual_seed(0)


def train(day_train):
    model.train()
    # lable = y[:day_train]
    pred = []
    train_loss = []
    for i in range(day_train):
        optimizer.zero_grad()
        out = model(dataset[i], tw_train[i])
        loss = F.huber_loss(out, y[i])
        out = out.cpu().reshape(1).detach().numpy()
        pred.append(out)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.array(train_loss)
    val_loss = val(day_train)
    return np.mean(train_loss), val_loss


@torch.no_grad()
def val(day_train):
    model.eval()
    lable = y[day_train:168]
    pred = []
    loss = {}
    with torch.no_grad():
        for i in range(day_train, 168):
            out = model(dataset[i], tw_train[i])
            out = out.cpu().reshape(1).detach().numpy()
            if out < 0:
                out = 0
            pred.append(out)
        output = np.array(pred)
        label = np.array(lable)

        val_mse_loss = mse(output, label)
        val_rmse_loss = rmse(output, label)
        val_mae_loss = mae(output, label)
        val_mape_loss = mape(output, label)
        loss['val_mse_loss'] = val_mse_loss
        loss['val_rmse_loss'] = val_rmse_loss
        loss['val_mae_loss'] = val_mae_loss
        loss['val_mape_loss'] = val_mape_loss

        return loss


@torch.no_grad()
def test(day_test, path):
    model = torch.load(path)
    model.eval()
    lable = y[day_test:]
    pred = []
    loss = {}
    with torch.no_grad():
        for i in range(day_test, 210):
            out = model(dataset[i], tw_train[i])
            out = out.cpu().reshape(1).detach().numpy()
            if out < 0:
                out = 0
            pred.append(out)
        output = np.array(pred)
        label = np.array(lable)

        test_mse_loss = mse(output, label)
        test_rmse_loss = rmse(output, label)
        test_mae_loss = mae(output, label)
        test_mape_loss = mape(output, label)
        loss['test_mse_loss'] = test_mse_loss
        loss['test_rmse_loss'] = test_rmse_loss
        loss['test_mae_loss'] = test_mae_loss
        loss['test_mape_loss'] = test_mape_loss
        return loss, pred


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    model = Model(dataset[0][0][0].num_node_features, dataset[0][0][1].num_node_features,
                  dataset[0][1][0].num_node_features, dataset[0][1][1].num_node_features,
                  dataset[0][2][0].num_node_features, dataset[0][2][1].num_node_features,
                  dataset[0][3][0].num_node_features, dataset[0][3][1].num_node_features
                  ).to('cpu')
    # print(model)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    for i in range(len(dataset)):
        for j in range(4):
            for k in range(3):
                dataset[i][j][k] = dataset[i][j][k].to(device)
    tw_train = tw_train.to(device)
    model = model.to(device)

    # with torch.no_grad():  # Initialize lazy modules.
    #     # for i in range(day_train):
    #     out = model(dataset[0], tw_train[0])
    #     # model = model.double()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # 学习率衰减
    min_epochs = 10
    best_model = None
    min_rmse_loss = 100
    min_mae_loss = 100
    best = []

    best_mse_list = []
    best_rmse_list = []
    best_mae_list = []
    best_mape_list = []
    for epoch in range(200):
        # scheduler.step()
        train_loss, val_loss = train(126)
        # test_loss, pred = test(108)

        # best model
        if val_loss['val_rmse_loss'] < min_rmse_loss and val_loss['val_mae_loss'] < min_mae_loss:
            min_rmse_loss = val_loss['val_rmse_loss']
            min_mae_loss = val_loss['val_mae_loss']
            best = [val_loss['val_mse_loss'], val_loss['val_rmse_loss'], val_loss['val_mae_loss'],
                    val_loss['val_mape_loss'],
                    epoch]
            best_mae_list.append(best[2])
            best_mape_list.append(best[3])
            best_mse_list.append(best[0])
            best_rmse_list.append(best[1])
            # if epoch % 10 == 0:
            torch.save(model, 'model/model_Hoorn_basegat_001_5e4.pth')

        print(f'Epoch: {epoch:03d}, '
              f"Train loss: {train_loss:.2f}")
        print(f'Epoch: {epoch:03d}, Val loss: '
              f" MAE {val_loss['val_mse_loss']:.2f}; MAPE {val_loss['val_mape_loss']:.2f}; "
              f" MSE {val_loss['val_mse_loss']:.2f};  RMSE {val_loss['val_rmse_loss']:.2f}")

    loss, pred = test(168, 'model/model_Hoorn_basegat_001_5e4.pth')
    print(loss)
    # plt.plot(pred, label="pred")
    # plt.plot(y[168:], label="true")
    # plt.legend()
    # plt.savefig('./model/figure_Hoorn.png')
    # plt.show()

    # print(
    #     'best tesing results: \n'
    #     'MAE: {:.2f}\n'
    #     'MRE: {:.2f}\n'
    #     'RMSE: {:.2f}\n'
    #     'MAPE: {:.2f}\n'
    #     'SMAPE: {:.2f}\n'
    #     'MSE: {:.2f}\n'
    #     'R2: {:.2f}\n'.format(best[2],
    #                           best[6],
    #                           best[1],
    #                           best[3],
    #                           best[4],
    #                           best[0],
    #                           best[5]))
