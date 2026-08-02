import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_tsne(embeedings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=0)
    X_tsne = tsne.fit_transform(embeedings)
    return X_tsne


def vis_embeedings(embeedings, labels=None, save_path=None):
    if labels is not None:
        assert embeedings.shape[0] == len(labels)
    
    # 使用t-SNE进行降维
    X_tsne = get_tsne(embeedings, n_components=2)

    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # 绘制t-SNE可视化图
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 图中文字体设置为Times New Roman
    
    # 遍历所有标签种类
    if labels is not None:
        marker_list = ['o', 'v', '1', '<', '2', 's', '3', '*', '4', '+', 'x', 'D', '^', 'P']  # 设置不同类别的形状
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'purple', 'yellow', 'brown', 'pink', 'orange']  # 设置不同类别的颜色
        labels = np.array(labels, dtype=int)
        unique_labels = np.unique(labels)
        unique_markers = [marker_list[i % len(marker_list)] for i in range(len(unique_labels))]
        unique_colors = [color_list[i % len(color_list)] for i in range(len(unique_labels))]
        for i in range(len(unique_labels)):
            plt.scatter(X_norm[labels == unique_labels[i], 0], X_norm[labels == unique_labels[i], 1], color=unique_colors[i],
                        marker=unique_markers[i], s=25, label=f'{unique_labels[i]}', alpha=0.5)   # 绘制散点图,s=150表示形状大小, alpha=0.5表示图形透明度(越小越透明)
    else:
        plt.scatter(X_norm[:, 0], X_norm[:, 1], color='b', marker='o', alpha=0.5)

    # 添加图例，并设置字体大小
    # plt.legend(loc='right', fontsize=16)

    plt.xticks(fontsize=20)  # 定义坐标轴刻度
    plt.yticks(fontsize=20)

    # plt.xlabel('x', fontsize=20)  # 定义坐标轴标题
    # plt.ylabel('y', fontsize=20)
    # plt.title('t-SNE Visualization', fontsize=24)  # 定义图题

    if save_path is not None:
        plt.savefig(save_path)


if __name__ == '__main__':
    # 定义三个类别的均值和协方差矩阵
    mean1 = [0, 1]
    cov1 = [[1, 0.3], [0.3, 1]]
    mean2 = [3, 3]
    cov2 = [[1, -0.2], [-0.2, 3]]
    mean3 = [-10, 10]
    cov3 = [[1, 0], [0, 0.5]]
    mean4 = [-4, 2]
    cov4 = [[0.5, 0.2], [0.2, 2]]

    # 生成三个类别的样本数据
    data1 = np.random.multivariate_normal(mean1, cov1, 100)
    data2 = np.random.multivariate_normal(mean2, cov2, 100)
    data3 = np.random.multivariate_normal(mean3, cov3, 100)
    data4 = np.random.multivariate_normal(mean4, cov4, 100)

    # 生成三个类别的样本数据的标签
    label1 = np.zeros(data1.shape[0]) + 0
    label2 = np.zeros(data1.shape[0]) + 1
    label3 = np.zeros(data1.shape[0]) + 2
    label4 = np.zeros(data1.shape[0]) + 3

    # 将三个类别的数据合并
    data = np.concatenate((data1, data2, data3, data4))
    labels = np.concatenate((label1, label2, label3, label4))
    
    vis_embeedings(data, labels, save_path='/home/ericxhzou/Code/InCloud/exp/Ours/Q.png')
