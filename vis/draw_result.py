# -*- coding: utf-8 -*-
import os.path
import random
import csv
import math

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from PIL import Image
from matplotlib import pyplot as plt
plt.rc('font', family='Times New Roman')

from misc.utils import get_files, load_pc, normalize_point_cloud


# get color by value
def get_color_by_value(value, min_value=-1, max_value=1):
    norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
    rgb = list(matplotlib.cm.jet(norm(value), bytes=True))[:3]
    color = '#'
    for i in range(len(rgb)):
        num = int(rgb[i])
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


# draw point cloud in matplot
def draw_pc(pc, save_filepath=None, title_info='', pt_size=3, show_fig=False):
    if not show_fig:
        matplotlib.use('Agg')
    if pc.shape[0] > 4096:
        sample_idxs = np.random.choice(pc.shape[0], 4096, replace=False)
        pc = pc[sample_idxs]
    pc,_,_ = normalize_point_cloud(pc)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
               cmap='rainbow')
    ax.set_title(title_info, fontsize=30)
    ax.axis()
    # set init view
    ax.view_init(elev=65.0, azim=-45.0)
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close('all')


# draw point cloud with colors
def draw_pc_with_color(pc, color_values, pt_size=3, save_filepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_value, max_value = np.min(color_values), np.max(color_values)
    for i in range(pc.shape[0]):
        color = get_color_by_value(color_values[i], min_value, max_value)
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2], s=pt_size, color=color)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    plt.close('all')


def draw_pcs(pcs, extra_infos, save_filepath=None, pt_size=20, show_fig=False):
    if not show_fig:
        matplotlib.use('Agg')
    fig = plt.figure(figsize=(75, len(pcs)*75))
    gs = fig.add_gridspec(1, len(pcs))
    for i in range(len(pcs)):
        pc,_,_ = normalize_point_cloud(pcs[i])
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        ax = fig.add_subplot(gs[:, i], projection='3d')
        ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
                   cmap='rainbow')
        ax.set_title(extra_infos[i], fontsize=30)
        ax.axis()
    plt.tight_layout()
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close('all')


# draw pcs in dir
def draw_pcs_in_dir(in_dir, pt_size=3, show_fig=False):
    # get files in in_dir
    pc_files = get_files(in_dir, '.bin')
    for pc_f in pc_files:
        svg_f = os.path.join(in_dir, os.path.splitext(os.path.basename(pc_f))[0] + '.svg')
        pc = load_pc(pc_f)
        draw_pc(pc, svg_f, pt_size=pt_size, show_fig=show_fig)


# add border to image
def add_img_border(src, loc='a', width=3, color=(0, 0, 0, 255)):
    """
        src: (str) 需要加边框的图片
        loc: (str) 边框添加的位置, 默认是'a'(
            四周: 'a' or 'all'
            上: 't' or 'top'
            右: 'r' or 'rigth'
            下: 'b' or 'bottom'
            左: 'l' or 'left' )
        width: (int) 边框宽度 (默认是3)
        color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
    """
    # 读取图片
    w = src.size[0]
    h = src.size[1]

    # 添加边框
    dst = None
    if loc in ['a', 'all']:
        w += 2 * width
        h += 2 * width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (width, width))
    elif loc in ['t', 'top']:
        h += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, 0, w - width, h))
    elif loc in ['b', 'bottom']:
        h += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, 0, w, h - width))
    elif loc in ['l', 'left']:
        w += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (width, 0, w, h))
    else:
        pass
    return dst


# draw features with t-SNE
def draw_features_with_tsne(features, labels, pc_idxs, pcs, out_path, rewrite_img=True):
    # draw pcs and save to files
    image_files = []
    temp_path = os.path.join(os.path.dirname(out_path), 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    for i in range(len(pcs)):
        save_filepath = os.path.join(temp_path, '{}.png'.format(i))
        image_files.append(save_filepath)
        if not os.path.exists(save_filepath) and rewrite_img:
            draw_pc(pcs[i], save_filepath, 'id: {}'.format(pc_idxs[i]))
    # draw t-sne
    feat_tsne = TSNE(perplexity=5).fit_transform(features)  # high dim -> 2 dim
    tx, ty = feat_tsne[:, 0], feat_tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    bk_width, bk_height, max_dim = 4000, 3000, 175
    background = Image.new('RGBA', (bk_width, bk_height), (255, 255, 255, 255))
    for x, y, img_file, lbl in zip(tx, ty, image_files, labels):
        img = Image.open(img_file)
        if int(lbl) > 1:  # query
            border_color = (128, 0, 128, 255)  # purple
        elif int(lbl) == 1:
            border_color = (0, 255, 0, 255)  # green
        else:
            border_color = (255, 0, 0, 255)  # red
        img = add_img_border(img, width=5, color=border_color)
        rs = max(1, img.width / max_dim, img.height / max_dim)
        img = img.resize((int(img.width / rs), int(img.height / rs)), Image.Resampling.LANCZOS)
        background.paste(img, (int((bk_width - max_dim) * x), int((bk_height - max_dim) * y)), img)
    background.save(out_path)


# draw curve
def draw_line_chart(data_list, title='', xlabel='', ylabel='', xrange=[0, 25], xstep=1,
                    yrange=[50, 100], ystep=10, xtick_delta=1, xtick_step=5, font_size=15, legend_loc='lower right', save_filepath=None):
    # 如果要显示中文字体,则在此处设为：SimHeiplt.rcParams['axes.unicode_minus'] = False # 显示负号
    plt.rcParams['font.sans-serif'] = ['Arial']
    x = np.arange(xrange[0], xrange[1], step=xstep)
    y = np.arange(yrange[0], yrange[1] + ystep, step=ystep)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：- -- -. : ,# marker：. , o v < * + 1
    plt.figure(figsize=(10, 8))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    for data_i in data_list:
        plt.plot(x, data_i.data, marker=data_i.fig_marker, color=data_i.fig_color, label=data_i.network, linewidth=1.5)

    x = np.arange(xrange[0], xrange[1] + xstep, step=xtick_step)  # xrange[0] - xtick_delta
    plt.xticks(x, fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(y, fontsize=font_size, fontweight='bold')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel=xlabel, fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontsize=font_size, fontweight='bold')
    plt.xlim(xrange[0]-xtick_delta, xrange[1]+xtick_delta)  # 设置x轴的范围
    plt.ylim(yrange[0], yrange[1])

    plt.legend()  # 显示各曲线的图例
    plt.legend(loc=legend_loc, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细

    if save_filepath:
        plt.savefig(save_filepath, format='svg') # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中plt.show()
    plt.close('all')


# draw two clouds with point pairs
def draw_pc_pps(src_pc, src_kpt, tgt_pcs, tgt_kpt, tgt_states=None, offset_x=90.0, pt_size=3, title=None, save_filepath=None):
    # if save_filepath is None:
    #     matplotlib.use('Agg')
    # centralize clouds
    src_center = np.mean(src_kpt, axis=0, keepdims=True)
    src_pc = src_pc - src_center
    src_kpt = src_kpt - src_center
    tgt_center = src_center - np.array([[offset_x, 0.0, 0.0]])
    tgt_kpt = tgt_kpt - tgt_center
    # src/tgt clouds
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], s=pt_size, color='purple')
    if isinstance(tgt_pcs, list):
        for i in range(len(tgt_pcs)):
            tgt_pc = tgt_pcs[i] - tgt_center
            tgt_state = 1 if tgt_states is None else tgt_states[i]
            tgt_color = 'green' if tgt_state else 'red'
            ax.scatter(tgt_pc[:, 0], tgt_pc[:, 1], tgt_pc[:, 2], s=pt_size, color=tgt_color)
    else:
        tgt_pcs = tgt_pcs - tgt_center
        ax.scatter(tgt_pcs[:, 0], tgt_pcs[:, 1], tgt_pcs[:, 2], s=pt_size, color='green')
    # point pairs
    color_groups = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'chocolate', 'cyan', 'lime', 'olive']
    for i in range(len(src_kpt)):
        pps_color = random.choice(color_groups)
        ax.plot([src_kpt[i, 0], tgt_kpt[i, 0]], [src_kpt[i, 1], tgt_kpt[i, 1]],
                [src_kpt[i, 2], tgt_kpt[i, 2]], linewidth=2, color=pps_color)
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
    plt.close('all')


# draw two clouds without point pairs
def draw_two_pc(src_pc, tgt_pcs, tgt_states=None, use_diff_center=False, pt_size=3, title=None, save_filepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # src
    center = np.mean(src_pc, axis=0, keepdims=True)
    src_pc = src_pc - center
    # tgt
    if use_diff_center:
        center = np.mean(tgt_pcs.reshape(-1,3), axis=0, keepdims=True) - np.array([[75.0, 0.0, 0.0]])
    tgt_pcs = tgt_pcs - center
    if len(tgt_pcs.shape) == 2:
        ax.scatter(tgt_pcs[:, 0], tgt_pcs[:, 1], tgt_pcs[:, 2], s=pt_size, color='red')
    else:  # len(tgt_pcs.shape) == 3
        for i in range(tgt_pcs.shape[0]):
            tgt_pc = tgt_pcs[i]
            if tgt_states is None:
                tgt_color = 'red'
            else:
                tgt_color = 'green' if tgt_states[i] else 'red'
            ax.scatter(tgt_pc[:, 0], tgt_pc[:, 1], tgt_pc[:, 2], s=pt_size, color=tgt_color)
    ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], s=pt_size*3, color='purple')
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
    plt.close('all')


# recall / precision @topN struct
class RecallPrecisionTopN:
    def __int__(self):
        self.network = 'unknown'
        self.fig_color = 'blue'
        self.fig_marker = '.'
        self.data = np.array()


# draw recall/precision @top1~25 curve
def draw_recall_precision_curve(csv_files, save_dir):
    """ csv_files: evaluation reults in csv files
    """
    # load data
    data_list, methods, envs = [], set(), set()
    for i in range(len(csv_files)):
        filename = os.path.basename(csv_files[i])
        if 'PointNetVLAD' in filename:
            method = 'PointNetVLAD'
        elif 'PPTNet' in filename:
            method = 'PPTNet'
        elif 'MinkLoc3D' in filename:
            method = 'MinkLoc3D'
        elif 'EgoNN' in filename:
            method = 'EgoNN'
        elif 'LoGG3DNet' in filename:
            method = 'LoGG3D-Net'
        else:
            continue
        methods.add(method)
        df = pd.read_csv(csv_files[i])
        for idx, row in df.iterrows():
            data = {'method': method,
                    'env': row['env'],
                    'Recall': [row[f'R@{x}'] for x in range(1, 26)],
                    'Precision': [row[f'P@{x}'] for x in range(1, 26)]
                   }
            data_list.append(data)
            envs.add(data['env'])
    
    def get_data(method, env, data_type):
        if data_type != 'Recall' and data_type != 'Precision':
            return None
        for i in range(len(data_list)):
            if data_list[i]['method'] == method and data_list[i]['env'] == env:
                return data_list[i][data_type]
    
    # draw
    colors = ['blue', 'green', 'chocolate', 'red', 'orange', 'purple', 'pink']
    markers = ['v', 's', '^', 'd', 'x', 'o', '-']
    data_types = ['Recall', 'Precision']
    for e in envs:
        for t in data_types:
            rp_datas = []
            for m in methods:
                rp_data = RecallPrecisionTopN()
                rp_data.network = m
                rp_data.fig_color = colors[len(rp_datas)]
                rp_data.fig_marker = markers[len(rp_datas)]
                rp_data.data = np.array(get_data(m, e, t))
                rp_datas.append(rp_data)
            save_filepath = os.path.join(save_dir, f'PR_{t}_on_{e}.svg')
            legend_loc = 'lower right' if t == 'Recall' else 'upper right'
            draw_line_chart(rp_datas, title=f'{t} on {e}', xlabel='N-Number of top database candidates',
                            ylabel=f'{t}@N (%)', yrange=[0, 100], ystep=20, legend_loc=legend_loc,
                            save_filepath=save_filepath)


# draw recall/precision @top1~25 curve
def draw_recall_precision_curve_rerank(csv_files, save_dir):
    """ csv_files: evaluation reults in csv files
    """
    # load data
    data_list, rerank_methods, pr_backbones, envs = [], set(), set(), set()
    for i in range(len(csv_files)):
        filename = os.path.basename(csv_files[i])
        
        if 'PPTNet' in filename:
            pr_backbone = 'PPTNet'
        elif 'EgoNN' in filename:
            pr_backbone = 'EgoNN'
        elif 'LoGG3DNet' in filename:
            pr_backbone = 'LoGG3D-Net'
        else:
            continue
        pr_backbones.add(pr_backbone)
        
        if 'aQE' in filename:
            rerank_method = 'aQE'
        elif 'avgQE' in filename:
            rerank_method = 'avgQE'
        elif 'SGV' in filename:
            rerank_method = 'SGV'
        elif 'RPR' in filename:
            rerank_method = 'RPR'
        elif 'RANSAC' in filename:
            rerank_method = 'RANSAC'
        else:
            continue
        rerank_methods.add(rerank_method)
        
        df = pd.read_csv(csv_files[i])
        for idx, row in df.iterrows():
            data = {'rerank_method': rerank_method,
                    'pr_backbone': pr_backbone,
                    'env': row['env'],
                    'Recall': [row[f'R@{x}'] for x in range(1, 26)],
                    'Precision': [row[f'P@{x}'] for x in range(1, 26)]
                   }
            data_list.append(data)
            envs.add(data['env'])
    
    def get_data(rerank_method, pr_backbone, env, data_type):
        if data_type != 'Recall' and data_type != 'Precision':
            return None
        for i in range(len(data_list)):
            if data_list[i]['rerank_method'] == rerank_method and data_list[i]['pr_backbone'] == pr_backbone and data_list[i]['env'] == env:
                return data_list[i][data_type]
    
    # draw
    colors = ['blue', 'green', 'chocolate', 'red', 'orange', 'purple', 'pink'][:len(rerank_methods)]
    markers = ['v', 's', '^', 'd', 'x', 'o', '-'][:len(pr_backbones)]
    data_types = ['Recall', 'Precision']
    for e in envs:
        for t in data_types:
            rp_datas = []
            for m_idx, m in enumerate(rerank_methods):
                for b_idx, b in enumerate(pr_backbones):
                    rp_data = RecallPrecisionTopN()
                    rp_data.network = f'{m}+{b}'
                    rp_data.fig_color = colors[m_idx]
                    rp_data.fig_marker = markers[b_idx]
                    rp_data.data = np.array(get_data(m, b, e, t))
                    rp_datas.append(rp_data)
            save_filepath = os.path.join(save_dir, f'Rerank_{t}_on_{e}.svg')
            legend_loc = 'lower right' if t == 'Recall' else 'upper right'
            draw_line_chart(rp_datas, title=f'{t} on {e}', xlabel='N-Number of top database candidates',
                            ylabel=f'{t}@N (%)', yrange=[0, 100], ystep=20, legend_loc=legend_loc,
                            save_filepath=save_filepath)


# draw recall/precision @top1~25 curve
def draw_recall_precision_curve_viewpoint(csv_files, save_dir):
    """ csv_files: evaluation reults in csv files
    """
    # load data
    data_list, methods, rotations, envs = [], set(), set(), set()
    for i in range(len(csv_files)):
        filename = os.path.basename(csv_files[i])
        
        if 'PointNetVLAD' in filename:
            method = 'PointNetVLAD'
        elif 'PPTNet' in filename:
            method = 'PPTNet'
        elif 'MinkLoc3D' in filename:
            method = 'MinkLoc3D'
        elif 'EgoNN' in filename:
            method = 'EgoNN'
        elif 'LoGG3DNet' in filename:
            method = 'LoGG3D-Net'
        else:
            continue
        methods.add(method)
        
        if 'rotate0.0' in filename:
            rotation = '0.0'
        elif 'rotate30.0' in filename:
            rotation = '30.0'
        elif 'rotate60.0' in filename:
            rotation = '60.0'
        elif 'rotate90.0' in filename:
            rotation = '90.0'
        elif 'rotate120.0' in filename:
            rotation = '120.0'
        elif 'rotate150.0' in filename:
            rotation = '150.0'
        elif 'rotate180.0' in filename:
            rotation = '180.0'
        else:
            continue
        rotations.add(rotation)
        
        df = pd.read_csv(csv_files[i])
        for idx, row in df.iterrows():
            data = {'method': method,
                    'rotation': rotation,
                    'env': row['env'],
                    'R@1': [row[f'R@1']],
                    'R@1%': [row[f'R@1%']],
                    'P@1': [row[f'P@1']]
                   }
            data_list.append(data)
            envs.add(data['env'])
    
    def get_data(method, env, data_type):
        if data_type != 'R@1' and data_type != 'R@1%' and data_type != 'P@1':
            return None
        data = []
        for r in range(0, 210, 30):
            found = False
            for i in range(len(data_list)):
                if data_list[i]['method'] == method and data_list[i]['rotation'] == f'{r}.0' and data_list[i]['env'] == env:
                    data.append(data_list[i][data_type])
                    found = True
                    break
            assert found
        return data
    
    # draw
    colors = ['blue', 'green', 'chocolate', 'red', 'orange', 'purple', 'pink']
    markers = ['v', 's', '^', 'd', 'x', 'o', '-']
    data_types = ['R@1', 'R@1%', 'P@1']
    for e in envs:
        for t in data_types:
            rp_datas = []
            for m_idx, m in enumerate(methods):
                    rp_data = RecallPrecisionTopN()
                    rp_data.network = m
                    rp_data.fig_color = colors[m_idx]
                    rp_data.fig_marker = markers[m_idx]
                    rp_data.data = np.array(get_data(m, e, t))
                    rp_datas.append(rp_data)
            save_filepath = os.path.join(save_dir, f'Viewpoint_{t}_on_{e}.svg')
            draw_line_chart(rp_datas, title=f'{t} on {e}', xlabel='Rotation angle', ylabel=f'{t} (%)',
                            xrange=[0, 180], xstep=30, xtick_delta=5, xtick_step=30, yrange=[0, 100], ystep=20,
                            legend_loc='upper right', save_filepath=save_filepath)


# draw place recognition result: success / failure cases
def draw_pr_cases(case_path, num_query, num_top, show_fig=False):
    if not os.path.exists(case_path):
        print('Invalid case path: ', case_path)
    else:
        for i in range(num_query):
            # PatchAugNet success case-query
            pc_file = os.path.join(case_path, 'case{}-query.bin'.format(i))
            pc = load_pc(pc_file)
            svg_file = os.path.join(case_path, 'case{}-query.svg'.format(i))
            draw_pc(pc, svg_file, show_fig)
            # PatchAugNet success case-top
            for j in range(num_top):
                pc_file = os.path.join(case_path, 'case{}-top{}.bin'.format(i, j))
                pc = load_pc(pc_file)
                svg_file = os.path.join(case_path, 'case{}-top{}.svg'.format(i, j))
                draw_pc(pc, svg_file, show_fig)


def draw_demo_pic(mls_traj_file, pls_traj_file, result_file, demo_out_dir, demo_title, show_fig=False):
    if not show_fig:
        matplotlib.use('Agg')
    history_query_xs = []
    history_query_ys = []
    history_query_states = []

    # 从文本文件中读取MLS轨迹数据
    mls_data = []
    if os.path.splitext(os.path.basename(mls_traj_file))[1] == ".csv":
        pls_hgt = 100.0
        with open(mls_traj_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mls_data.append([float(row[1]), float(row[2]), 0.0])
    else:
        pls_hgt = 1.0
        with open(mls_traj_file, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            count = 0
            for line in lines:
                line_strs = line.split(' ')
                if count % 50 == 0:
                    mls_data.append([float(line_strs[8]), float(line_strs[7]), 0.0])
                count = count + 1
    mls_data = np.array(mls_data)
    mls_x, mls_y, mls_z = mls_data[:, 1], mls_data[:, 0], mls_data[:, 2]

    # 从文本文件中读取PLS轨迹数据
    pls_data = []
    with open(pls_traj_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pls_data.append([float(row[1]), float(row[2]), pls_hgt])
    pls_data = np.array(pls_data)
    pls_x, pls_y, pls_z = pls_data[:, 1], pls_data[:, 0], pls_data[:, 2]

    # get query files and found files
    with open(result_file, 'r') as f:
        lines = f.readlines()
        lines.pop(0)
    num_lines = len(lines)
    num_units = num_lines // 7
    item_dict = {
            'state': [],
            'name': [],
            'query_file': [],
            'query_x': [],
            'query_y': [],
            'found_file': [],
            'found_x': [],
            'found_y': []
        }
    for i in range(num_units):
        unit_lines = lines[i*7 : (i+1)*7]
        line0_strs = unit_lines[0].split(' ')
        query_submap_file = line0_strs[2]
        query_x = float(line0_strs[6])
        query_y = float(line0_strs[8])
        found_submap_file = None
        found_submap_state = False
        found_x = found_y = None
        for j in range(2, 7):
            unit_lines_j_strs = unit_lines[j].split(' ')
            found_submap_file = unit_lines_j_strs[1]
            found_submap_state = unit_lines_j_strs[3]
            found_x = float(unit_lines_j_strs[5])
            found_y = float(unit_lines_j_strs[7])
            if found_submap_state == 'True':
                found_submap_state = True
                break
            else:
                found_submap_state = False

        dist = math.sqrt((query_x - found_x)**2 + (query_y - found_y)**2)
        if dist > 30.0 and found_submap_state:
            continue

        item_dict['state'].append(found_submap_state)
        item_dict['name'].append(float(os.path.splitext(os.path.basename(query_submap_file))[0]))
        item_dict['query_file'].append(query_submap_file)
        item_dict['query_x'].append(query_x)
        item_dict['query_y'].append(query_y)
        item_dict['found_file'].append(found_submap_file)
        item_dict['found_x'].append(found_x)
        item_dict['found_y'].append(found_y)

    item_pd = pd.DataFrame(item_dict)
    #item_pd = item_pd.sort_values(by=['name'])
    item_pd = item_pd.sample(frac=1.0)
    count = 0
    for index, row in item_pd.iterrows():
        if count == 250:
            break
        # 创建画布和子图
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3)

        # 绘制MLS&PLS轨迹，添加标题
        axs_left = fig.add_subplot(gs[:, :2], projection='3d')
        axs_left.scatter(mls_x, mls_y, mls_z, s=2, color='gray', zorder=1)
        axs_left.scatter(pls_x, pls_y, pls_z, s=2, color='black', zorder=1)
        # axs_left.set_title('MLS and PLS Trajectory')
        if not os.path.splitext(os.path.basename(mls_traj_file))[1] == ".csv":
            axs_left.view_init(elev=60.0, azim=-90)
        axs_left.axis('off')

        # 连接加粗的轨迹点
        line_color = 'limegreen'
        if not row['state']:
            line_color = 'red'
        history_query_xs.append(row['query_x'])
        history_query_ys.append(row['query_y'])
        history_query_states.append(row['state'])
        for j in range(len(history_query_xs)):
            if not history_query_states[j]:
                axs_left.scatter(history_query_xs[j], history_query_ys[j], pls_hgt, marker='o', s=75, color='red', zorder=3)
            else:
                axs_left.scatter(history_query_xs[j], history_query_ys[j], pls_hgt, marker='o', s=75, color='limegreen',zorder=3)
        axs_left.scatter(row['found_x'], row['found_y'], 0.0, marker='o', s=25, color='black', zorder=3)
        axs_left.plot([row['query_x'], row['found_x']], [row['query_y'], row['found_y']], [pls_hgt, 0.0], linewidth=4, color=line_color, zorder=2)

        # 绘制找到的query子图点云，添加标题
        query_pc = load_pc(row['query_file'])
        query_pc,_,_ = normalize_point_cloud(query_pc)
        axs11 = fig.add_subplot(gs[0, 2], projection='3d')
        axs11.scatter(query_pc[:, 0], query_pc[:, 1], query_pc[:, 2], s=3, c=query_pc[:, 2], cmap='rainbow')
        axs11.set_title('Query Submap: {}'.format(row['name']))
        axs11.axis()

        # 绘制找到的reference子图点云，添加标题
        ref_pc = load_pc(row['found_file'])
        ref_pc,_,_ = normalize_point_cloud(ref_pc)
        axs01 = fig.add_subplot(gs[1, 2], projection='3d')
        axs01.scatter(ref_pc[:, 0], ref_pc[:, 1], ref_pc[:, 2], s=3, c=ref_pc[:, 2], cmap='rainbow')
        axs01.set_title('Matched Submap: {}'.format(os.path.splitext(os.path.basename(row['found_file']))[0]))
        axs01.axis()

        # 添加整个图的标题
        fig.suptitle(demo_title)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

        # 保存图片
        out_file = os.path.join(demo_out_dir, str(count) + ".png")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close('all')
        count = count + 1


def draw_radar_chart(data_series, labels, categories, title='', normalization=False, save_filepath=None):
    # 归一化
    if normalization:
        for i in range(len(data_series)):
            max = np.max(data_series[i])
            data_series[i] = data_series[i] / max
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_series)))
    
    # 绘制每个系列
    for i, (data, label, color) in enumerate(zip(data_series, labels, colors)):
        data = list(data)
        ax.plot(angles, data + data[:1], 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, data + data[:1], alpha=0.1, color=color)
    
    # 设置类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=24)
    
    # 设置径向网格
    # ax.set_rgrids([20, 40, 60, 80, 100])
    plt.ylim(0, 100)
    
    # 添加标题和图例
    plt.title(title, size=24, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=24)
    
    # 保存图片
    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    # #----------test Draw point pairs
    # src_pc = np.random.randint(-30, 30, (100, 3))
    # src_kpt = np.random.randint(-30, 30, (10, 3))
    # tgt_pc = np.random.randint(-30, 30, (100, 3))
    # tgt_kpt = np.random.randint(-30, 30, (10, 3))
    # pps_state = np.random.randint(0, 2, (10, 1))
    # draw_pc_pps(src_pc, src_kpt, tgt_pc, tgt_kpt, pps_state)

    # #----------test Draw tsne
    # features = np.random.randint(0, 100, (25, 32))
    # labels = list(np.random.randint(0, 2, (25, 1)))
    # pc_idxs = list(range(25))
    # pc = np.random.randint(-30, 30, (100, 3))
    # pcs = [pc] * 25
    # draw_features_with_tsne(features, labels, pc_idxs, pcs, '/home/ericxhzou/Code/PCGL-Benchmark/exp/pr/tsne.png')
    
    # #----------test Draw radar chart
    # categories = ['Speed', 'Power', 'Skill', 'Endurance', 'Agile', 'Intelligence']
    # players_data = [
    #     [85, 90, 75, 80, 85, 70],  # 球员A
    #     [70, 95, 80, 85, 75, 65],  # 球员B
    #     [80, 85, 90, 75, 80, 85]   # 球员C
    # ]
    # labels = ['Player A', 'Player B', 'Player C']
    # draw_radar_chart(players_data, labels, categories, title='Comparison of basketball players abilities',
    #                  save_filepath='/home/ericxhzou/Code/PCGL-Benchmark/exp/pr/radar.png',
    #                  normalization=False)
    
    #----------PR: recall / precision curves
    exp_dir = '/home/ericxhzou/Code/PCGL-Benchmark/exp/exp_with_dynamics/pr'
    csv_files = ['PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv'
                ]
    csv_files = [os.path.join(exp_dir, x) for x in csv_files]
    draw_recall_precision_curve(csv_files, save_dir=exp_dir)
    
    #----------Rerank: recall / precision curves
    exp_dir = '/home/ericxhzou/Code/PCGL-Benchmark/exp/exp_with_dynamics/rerank'
    csv_files = ['aQE/PPTNet/aQE_PPTNet_rerank.csv',
                 'aQE/EgoNN/aQE_EgoNN_rerank.csv',
                 'aQE/LoGG3DNet/aQE_LoGG3DNet_rerank.csv',
                 'SGV/PPTNet/SGV_PPTNet_rerank.csv',
                 'SGV/EgoNN/SGV_EgoNN_rerank.csv',
                 'SGV/LoGG3DNet/SGV_LoGG3DNet_rerank.csv',
                 'RPR/PPTNet/RPR_PPTNet_rerank.csv',
                 'RPR/EgoNN/RPR_EgoNN_rerank.csv',
                 'RPR/LoGG3DNet/RPR_LoGG3DNet_rerank.csv'
                ]
    csv_files = [os.path.join(exp_dir, x) for x in csv_files]
    draw_recall_precision_curve_rerank(csv_files, save_dir=exp_dir)
    
    #----------Viewpoint sensitive: recall / precision curves
    exp_dir = '/home/ericxhzou/Code/PCGL-Benchmark/exp/exp_with_dynamics/pr'
    csv_files = ['PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate30.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate60.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate90.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate120.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate150.0_on_Hankou_1_2.csv',
                 'PointNetVLAD/eval/PointNetVLAD_train-aug0_eval-aug2_rotate180.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate30.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate60.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate90.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate120.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate150.0_on_Hankou_1_2.csv',
                 'PPTNet/eval/PPTNet_train-aug0_eval-aug2_rotate180.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate30.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate60.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate90.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate120.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate150.0_on_Hankou_1_2.csv',
                 'MinkLoc3D/eval/MinkLoc3D_train-aug0_eval-aug2_rotate180.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate30.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate60.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate90.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate120.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate150.0_on_Hankou_1_2.csv',
                 'EgoNN/eval/EgoNN_train-aug0_eval-aug2_rotate180.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate0.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate30.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate60.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate90.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate120.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate150.0_on_Hankou_1_2.csv',
                 'LoGG3DNet/eval/LoGG3DNet_train-aug0_eval-aug2_rotate180.0_on_Hankou_1_2.csv'
                ]
    csv_files = [os.path.join(exp_dir, x) for x in csv_files]
    draw_recall_precision_curve_viewpoint(csv_files, save_dir=exp_dir)
