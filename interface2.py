# 导入库
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from PIL import Image
from model_all import *
import os

dataset = {}
tw_train = {}
y = {}
begin_days = {'Hoorn': 168, 'Krommenie': 145, 'Hyattsville': 85}
end_days = {'Hoorn': 210, 'Krommenie': 181, 'Hyattsville': 116}

dataset['Hoorn'] = torch.load('./interface/graph.pt')
tw_data = pd.read_csv('./interface/Hoorn_ML.csv')
tw_train['Hoorn'] = tw_data.iloc[:, 1:-1].reset_index(drop=True)
tw_train['Hoorn'] = np.array(tw_train['Hoorn'])
tw_train['Hoorn'] = torch.tensor(tw_train['Hoorn'], dtype=torch.float32)
y['Hoorn'] = tw_data.iloc[:, -1][:210].reset_index(drop=True)
# y['Hoorn']= torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset['Krommenie'] = torch.load('./interface/Krommenie_litter_graph.pt')
tw_data2 = pd.read_csv('./interface/Krommenie_ML.csv')
tw_train['Krommenie'] = tw_data2.iloc[:, 1:-1].reset_index(drop=True)
tw_train['Krommenie'] = np.array(tw_train['Krommenie'])
tw_train['Krommenie'] = torch.tensor(tw_train['Krommenie'], dtype=torch.float32)
y['Krommenie'] = tw_data2.iloc[:, -1][:181].reset_index(drop=True)

dataset['Hyattsville'] = torch.load('./interface/Hyattsville_litter_graph.pt')
tw_data2 = pd.read_csv('./interface/Hyattsville_ML.csv')
tw_train['Hyattsville'] = tw_data2.iloc[:, 1:-1].reset_index(drop=True)
tw_train['Hyattsville'] = np.array(tw_train['Hyattsville'])
tw_train['Hyattsville'] = torch.tensor(tw_train['Hyattsville'], dtype=torch.float32)
y['Hyattsville'] = tw_data2.iloc[:, -1][:116].reset_index(drop=True)

# 模型对象

model_path = {'Hoorn': './interface/model_Hoorn_005.pth',
              'Krommenie': './interface/model_K_all_0055e5.pth',
              'Hyattsville': './interface/model_Hy_gcn_0055e5.pth'}
# 读取图像

image_path = {'Hoorn': './interface/Hoorn_litter.png',
              'Krommenie': './interface/Krommenie_litter.png',
              'Hyattsville': './interface/Hyattsville_litter.png'}


@torch.no_grad()
def test(day_test, city):
    model = torch.load(model_path[city])
    model.eval()
    # label = y[day_test:]
    pred = []
    with torch.no_grad():
        for i in range(begin_days[city], begin_days[city] + day_test):
            out = model(dataset[city][i], tw_train[city][i])
            out = out.cpu().reshape(1).detach().numpy()
            if out < 0:
                out = 0
            pred.append(out)
        # output = np.array(pred)

        return pred


# 定义函数
def predict(city, impath, days):
    # model = model_dic[city]
    # device = torch.device('cpu')
    # if city == 'Hoorn':
    #     model = model_dic[city]
    #
    #     for i in range(len(dataset['Hoorn'])):
    #         for j in range(4):
    #             for k in range(3):
    #                 dataset['Hoorn'][i][j][k] = dataset['Hoorn'][i][j][k].to(device)
    #     tw_train['Hoorn'] = tw_train['Hoorn'].to(device)
    #     model = model.to(device)

    pred = test(day_test=days, city=city)
    # 绘制折线图
    # plt.style.use('dark_background')
    plt.plot(y[city][begin_days[city]:begin_days[city] + days].tolist(), label="True")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Day")
    plt.ylabel("Litter Amount")
    plt.legend()
    plt.title(f"Litter Prediction for {city}")
    plt.savefig('./interface/{}_prediction.png'.format(city))
    plt.show()
    img = cv.imread('./interface/{}_prediction.png'.format(city))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # 返回图像对象
    return img


css = """
    body {
        font-size: 20px;
    }
    
"""

# 创建界面
demo = gr.Interface(
    fn=predict,  # 函数对象
    inputs=[gr.Dropdown(["Hoorn", "Krommenie", "Hyattsville"], label="City"),  # 下拉菜单输入
            gr.Image(shape=(700, 700), label="Litter Distribution").style(height=500),  # 图像输入
            gr.Slider(1, 30, default=7, step=1, label="Days")],  # 数字滑块输入
    outputs=gr.Image(shape=(200, 200)).style(height=700),  # 图像输出

    examples=[
        ["Hoorn", "E:/桌面/实验室/plastic litter/数据集/code/interface/Hoorn_litter.png"],
        ["Krommenie", "E:/桌面/实验室/plastic litter/数据集/code/interface/Krommenie_litter.png"],
        ["Hyattsville", "E:/桌面/实验室/plastic litter/数据集/code/interface/Hyattsville_litter.png"]

    ],
    css=css,
    theme='bethecloud/storj_theme'  # 'bethecloud/storj_theme' or 'freddyaboulton/dracula_revamped'

)

# 启动界面
demo.launch()
