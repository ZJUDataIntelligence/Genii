from matplotlib import pyplot as plt
from model_all import *
# from model_no_Heterogeneous import *
# from model_no_poi import *
# from model_no_tw import *
from utils import *


# def train(day_train) -> float:
#     model.train()
#     # train_iterations = []
#     train_loss = []
#     for i in range(day_train):
#         optimizer.zero_grad()
#         out = model(dataset[i], tw_train[i])
#         loss = F.smooth_l1_loss(out, y[i])
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.item())
#         # train_iterations.append(epoch + 1)
#     loss = np.mean(train_loss)
#     return float(loss)

def train(day_train) -> Dict[str, float]:
    model.train()
    lable = y[:day_train]
    pred = []
    train_loss = {}
    for i in range(day_train):
        optimizer.zero_grad()
        out = model(dataset[i], tw_train[i])
        loss = F.smooth_l1_loss(out, y[i])
        out = out.cpu().reshape(1).detach().numpy()
        pred.append(out)
        loss.backward()
        optimizer.step()


    output = np.array(pred)
    label = np.array(lable)

    train_mse_loss = mse(output, label)
    train_mre_loss = mre(output, label)
    train_rmse_loss = rmse(output, label)
    train_mae_loss = mae(output, label)
    train_mape_loss = mape(output, label)
    train_smape_loss = smape(output, label)
    train_r2_loss = r2_score(output, label)
    train_loss['train_mse_loss'] = train_mse_loss
    train_loss['train_mre_loss'] = train_mre_loss
    train_loss['train_rmse_loss'] = train_rmse_loss
    train_loss['train_mae_loss'] = train_mae_loss
    train_loss['train_mape_loss'] = train_mape_loss
    train_loss['train_smape_loss'] = train_smape_loss
    train_loss['train_r2_loss'] = train_r2_loss
    return train_loss


@torch.no_grad()
def test(day_train):
    model.eval()
    lable = y[day_train:]
    pred = []
    loss = {}
    with torch.no_grad():
        for i in range(day_train, 210):
            out = model(dataset[i], tw_train[i])
            out = out.cpu().reshape(1).detach().numpy()
            if out < 0:
                out = 0
            pred.append(out)
        output = np.array(pred)
        label = np.array(lable)

        test_mse_loss = mse(output, label)
        test_mre_loss = mre(output, label)
        test_rmse_loss = rmse(output, label)
        test_mae_loss = mae(output, label)
        test_mape_loss = mape(output, label)
        test_smape_loss = smape(output, label)
        test_r2_loss = r2_score(output, label)
        loss['test_mse_loss'] = test_mse_loss
        loss['test_mre_loss'] = test_mre_loss
        loss['test_rmse_loss'] = test_rmse_loss
        loss['test_mae_loss'] = test_mae_loss
        loss['test_mape_loss'] = test_mape_loss
        loss['test_smape_loss'] = test_smape_loss
        loss['test_r2_loss'] = test_r2_loss
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
    min_epochs = 10
    best_model = None
    min_rmse_loss = 200
    best = []
    # train_mse_loss_list = []
    # train_rmse_loss_list = []
    # train_mae_loss_list = []
    # train_mape_loss_list = []
    # train_smape_loss_list = []
    # train_r2_loss_list = []
    #
    # test_mse_loss_list = []
    # test_rmse_loss_list = []
    # test_mae_loss_list = []
    # test_mape_loss_list = []
    # test_smape_loss_list = []
    # test_r2_loss_list = []

    best_mse_list = []
    best_rmse_list = []
    best_r2_list = []
    best_mae_list = []
    best_mape_list = []
    best_smape_list = []
    for epoch in range(100):
        train_loss = train(160)
        test_loss, pred = test(160)
        if test_loss['test_rmse_loss'] < min_rmse_loss and epoch % 10 == 0:
            min_rmse_loss = test_loss['test_rmse_loss']
            best = [test_loss['test_mse_loss'], test_loss['test_rmse_loss'], test_loss['test_mae_loss'],
                    test_loss['test_mape_loss'], test_loss['test_smape_loss'], test_loss['test_r2_loss'], test_loss['test_mre_loss'],
                    epoch]
            best_mae_list.append(best[2])
            best_mape_list.append(best[3])
            best_smape_list.append(best[4])
            best_mse_list.append(best[0])
            best_rmse_list.append(best[1])
            best_r2_list.append(best[5])
            torch.save(model, 'model/model_epoch{}.pth'.format(epoch))

        # print(f"epoch:{epoch + 1}, loss:{loss.item()}")
        print(f'Epoch: {epoch:03d}, Train loss: '
              f" MAE {train_loss['train_mse_loss']:.2f}; MAPE {train_loss['train_mape_loss']:.2f}; R2 {train_loss['train_r2_loss']:.2f}")
        print(f'Epoch: {epoch:03d}, Test loss: '
              f" MAE {test_loss['test_mae_loss']:.2f}; MRE {test_loss['test_mre_loss']:.2f}; MAPE {test_loss['test_mape_loss']:.2f}; R2 {test_loss['test_r2_loss']:.2f}"
              f" MSE {test_loss['test_mse_loss']:.2f}; SMAPE {test_loss['test_smape_loss']:.2f}; RMSE {test_loss['test_rmse_loss']:.2f}")

        # train_mse_loss_list.append(train_loss['train_mse_loss'])
        # train_rmse_loss_list.append(train_loss['train_rmse_loss'])
        # train_mae_loss_list.append(train_loss['train_mae_loss'])
        # train_mape_loss_list.append(train_loss['train_mape_loss'])
        # train_r2_loss_list.append(train_loss['train_r2_loss'])
        #
        # test_mse_loss_list.append(test_loss['test_mse_loss'])
        # test_rmse_loss_list.append(test_loss['test_rmse_loss'])
        # test_mae_loss_list.append(test_loss['test_mae_loss'])
        # test_mape_loss_list.append(test_loss['test_mape_loss'])
        # test_r2_loss_list.append(test_loss['test_r2_loss'])

    plt.plot(pred, label="pred")
    plt.plot(y[160:], label="true")
    plt.legend()
    plt.savefig('./model/figure_all.jpg')
    plt.show()

    print(
        'best tesing results: \n'
        'MAE: {:.2f}\n'
        'MRE: {:.2f}\n'
        'RMSE: {:.2f}\n'
        'MAPE: {:.2f}\n'
        'SMAPE: {:.2f}\n'
        'MSE: {:.2f}\n'
        'R2: {:.2f}\n'.format(best[2],
                              best[6],
                              best[1],
                              best[3],
                              best[4],
                              best[0],
                              best[5]))
