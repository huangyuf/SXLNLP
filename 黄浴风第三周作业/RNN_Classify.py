import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModule, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias = False, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()   #交叉熵函数作为损失函数


    def forward(self, x, y=None):
        x, _ = self.layer(x)
        x = self.fc(x[:, -1, :])  # 使用最后一个时间步的输出
        if y is not None:
            return self.loss(x, y)
        else:
            return x

#输入一个五维向量，哪个位置最大则返回哪个类型

def build_sample():
    x = np.random.random(5)
    max = 0
    for i in range(5):
        if x[i] > max:
            max = x[i]
            pos = i
    return x, pos

def build_dataset(sample_num):
    dataset_X = []
    dataset_Y = []
    for i in range(sample_num):
        x, y = build_sample()
        dataset_X.append(x)
        dataset_Y.append(y)
    return torch.FloatTensor(dataset_X), torch.LongTensor(dataset_Y)

def evaluate(model):
    model.eval()
    test_sample = 1000
    test_x, test_y = build_dataset(test_sample)
    test_x = test_x.unsqueeze(1)  # 添加 batch 维度
    correct, wrong = 0,0
    with torch.no_grad():
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == test_y).sum().item()
        wrong = test_sample - correct
    print("正确的个数：%d,错位的个数：%d,正确率：%f" % (correct, wrong, correct/(correct + wrong)))
    return correct / (correct + wrong)

def main():
    #初始化参数
    batch_size = 20 #一次训练样本的个数
    epoch_num = 20  #训练轮数
    input_size = 5  #输入维度
    output_size = 5
    hidden_size = 100
    train_sample = 5000 #训练样本个数
    learining_rate = 0.01   #学习率
    log = []
    #建立模型
    model = RNNModule(input_size, hidden_size, output_size)
    train_x, train_y = build_dataset(train_sample)  #建立训练数据
    train_x = train_x.unsqueeze(1)  # 添加 batch 维度
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learining_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward() #反向传播
            optimizer.step()   #梯度更新
            optimizer.zero_grad()  #梯度清零
            watch_loss.append(loss.item())
        print("-----------第%d轮：平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        print("----------第%d轮，准确率为：%f" % (epoch + 1, acc))
        log.append([acc, float(np.mean(watch_loss))])
        torch.save(model.state_dict(), "model.pt")
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    input_size = 5
    model = RNNModule(input_size, hidden_size=100, output_size=5)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict)
    model.eval()
    input_tensor = torch.FloatTensor(input_vec).unsqueeze(1)  # 添加 batch 维度
    with torch.no_grad():
        result = model(torch.FloatTensor(input_tensor))
        for vec, res in zip(input_vec, result):
            print("输入向量为：", vec)
            print("预测结果：", res)
            print("预测类别：", torch.argmax(res, dim=0).item())
            print("预测概率：", torch.max(torch.softmax(res, dim=0)).item())
            print("--------------------")


if __name__ == "__main__":
    #main()
    # 示例输入向量
    input_vec = [
        np.random.random(5),
        np.random.random(5),
        np.random.random(5)
    ]
    predict("model.pt", input_vec)






