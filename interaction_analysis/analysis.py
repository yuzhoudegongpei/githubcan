import argparse
import os
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from skimage.morphology import convex_hull_image
from predict import classification
import pandas
#设置一个解析器，添加参数解析参数

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/bio/data_lwj/er_mito_interaction_analysis/er_mito_interact_datasets')###############
parser.add_argument('--dst-dir', default='/home/bio/data_lwj/er_mito_interaction_analysis/result_20230629')
options = parser.parse_args()


def write_to_excel(data, dst):                 
    props = [v for _, v in data.items()]
    props = np.array(props)
    df = pd.DataFrame(
            {'MITO_ID': list(data.keys()),
             '细胞凸边行面积': props[:, 0],
             '线粒体面积密度': props[:, 1],
             '线粒体个数密度': props[:,2],
             '线粒体形态复杂度': props[:, 3],
             '内质网面积占比': props[:,4],
             '内质网节点密度':props[:,5],
             '内质网边长度': props[:, 6],
             '内质网网路属性': props[:, 7],
             '互作发生密度': props[:, 8],
             '互作面积占比': props[:, 9],
             '互作面积/线粒体面积':props[:, 10],
             '互作面积/内质网面积': props[:, 11],
             '互作在每类线粒体上的个数': props[:, 12],
             '互作发生个数/内质网复杂度属性': props[:, 13]
            })

    with pd.ExcelWriter(dst, mode='w+') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)  ######################################################################


def calculate_area(img):       #计算某个区域面积
    pts=np.where(img!=0)   #where 遍历图像像素，找到所有满足条件的像素点
    return len(pts[0])


def calculate_closest_node(img,nodes): #计算区域到最近节点(node)的距离
    # get center point
    pts = np.where(img != 0)                                                      #输出的元组类型
    y_mean,x_mean=np.mean(pts[0]),np.mean(pts[1])
    # print(x_mean,y_mean)

    dis_list=[]
    for i in range(len(nodes)):
        distance=np.sqrt((x_mean-nodes[i][0])**2 + (y_mean-nodes[i][1])**2)#sqrt平方根
        dis_list.append(distance)

    dis_list=np.array(dis_list)
    index=np.argmin(dis_list) #沿轴里最小值的索引

    return nodes[index],np.min(dis_list)



def point_distance_line(point,line_point1,line_point2):#计算点线之间的距离
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance


def calculate_closest_line(img,lines_list):#计算区域到最近边的距离
    # get center point
    pts = np.where(img != 0)
    y_mean,x_mean=np.mean(pts[0]),np.mean(pts[1])
    # print(x_mean,y_mean)
    center_point=np.array([x_mean,y_mean])

    dis_list=[]
    for i in range(len(lines_list)):
        distance=point_distance_line(center_point,lines_list[i][0],lines_list[i][1])   ###################################################
        dis_list.append(distance)

    dis_list=np.array(dis_list)
    index=np.argmin(dis_list)

    return lines_list[index]



def extract_feats(chull_area,inter_segm,net_segm, xlt_segm, G, prefix="mito", save_dir=None):#图像的特征，主要计算图中节点和边的属性
    density=nx.density(G)
    closeness_centrality=nx.closeness_centrality(G)#节点的中心系数
    # betweenness_centrality=nx.betweenness_centrality(G)
    net_propertity=(density,closeness_centrality)                                          #######

    # degree=nx.degree(G)
    # degree_centrality=nx.degree_centrality(G)

    # print(degree)
    # print(degree_centrality)
    # print(closeness_centrality)
    # print(betweenness_centrality)


    # label each mito
    inter_num,segm_label=cv2.connectedComponents(inter_segm,connectivity=8)#连通域的计算
    xlt_num, xlt_label = cv2.connectedComponents(xlt_segm, connectivity=8)

    # plt.imshow(xlt_label)
    plt.show()



    # xlt classification
    label_distribution=[]
    for i in range(1,xlt_num):
        show_map = np.zeros_like(xlt_label.copy())#输出和a相同shape和数据类型的子0数组
        pts=np.where(xlt_label==i)  #输出符合条件的元组
        show_map[pts]=255                                          ########################


        y_min,y_max=min(pts[0]),max(pts[0])
        x_min, x_max = min(pts[1]), max(pts[1])
        map=show_map[y_min:y_max,x_min:x_max]
        map_pad=np.pad(map,((1,1),(1,1)),'constant',constant_values=0)
        # plt.imshow(map_pad)
        # plt.show()
        img_path="1.png"
        cv2.imwrite(img_path, map_pad) #写入保存图片
        predicted_label=classification(img_path)
        label_distribution.append(predicted_label.item())#item提高精度
    print("label",label_distribution)
    cate1,cate2,cate3,cate4 = 0,0,0,0
    for i in label_distribution:
        if i==0:
            cate1+=1
        elif i==1:
            cate2+=1
        elif i==2:
            cate3+=1
        else:
            cate4+=1
    fuzadu = dict()                    #复杂度
    fuzadu['0']= cate1 / len(label_distribution)
    fuzadu['1'] = cate2 / len(label_distribution)
    fuzadu['2'] = cate3 / len(label_distribution)
    fuzadu['3'] = cate4 / len(label_distribution)

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)] ###########################################################3
    # all nodes list
    nodes = []
    lines=[]
    lines_length=[]
    for sub_G in S:
        for x, y in sub_G.nodes():
            nodes.append(np.array([x,y]))
        for l in sub_G.edges():
            lines.append(l)
            length=np.sqrt((l[0][0]-l[1][0])**2 + (l[0][1]-l[1][1])**2 )
            lines_length.append(length)


    feats = dict()
    xlt_area=calculate_area(xlt_segm)                          
    inter_area = calculate_area(inter_segm)
    net_area = calculate_area(net_segm)
    num_node=len(nodes)
    num_lines=len(lines)
    lines_length=np.array(lines_length)
    lines_aver_length=np.mean(lines_length)
    lines_max_length = np.max(lines_length)
    lines_min_length = np.min(lines_length)
    line_length=(lines_max_length,lines_min_length,lines_aver_length)


    dis_list=[]
    for i in range(1,inter_num):
        pts=np.where(segm_label==i)
        single_interaction=np.zeros_like(segm_label)
        single_interaction[pts]=1
        # calculate area of single_interaction
        closest_node,dis=calculate_closest_node(single_interaction,nodes)
        dis_list.append(dis)

    dis_list=np.array(dis_list)
    ave_dis=np.mean(dis_list)

    save_name=prefix
    feats[save_name] = [
            chull_area,
            xlt_area/chull_area,
            xlt_num/chull_area,
            fuzadu,
            net_area/chull_area,
            num_node/net_area,
            num_lines/net_area,
            line_length,
            net_propertity,
            inter_num/chull_area,
            inter_area/chull_area,
            inter_area/xlt_area,
            inter_area/net_area,0,0]

    return feats


def main():                                                        ###################################
    exp_list = os.listdir(options.data_dir)#读取文件和文件夹名字形成列表
    exp_list.sort()#进行排序

    mito_sample_dir = os.path.join(options.dst_dir, 'new_images_v2')#拼接路径           #################################
    os.makedirs(mito_sample_dir, exist_ok=True)#创建目录

    feats = dict()
    for exp_dir in exp_list: 
        print("experiment: %s" % (exp_dir))                                  ###########################
        mask_dir = os.path.join(options.data_dir, exp_dir, 'interaction_viz/merge_viz')
        graph_dir = os.path.join(options.data_dir, exp_dir, 'mito_networks/graph_gpickle')

        mask_list = os.listdir(mask_dir)#返回指定的文件夹包含的文件或文件夹的名字的列表
        mask_list.sort()#排序

        for filename in mask_list:
            print("mask %s" % (filename))                                
            mito_segm_dir = os.path.join(mask_dir, filename)
            mito_graph_dir = os.path.join(graph_dir, filename)

            img_list = os.listdir(mito_segm_dir)  #返回文件夹中文件或文件夹中的名称的列表
            img_list.sort()

            # feats = dict()
            for frame, img_name in enumerate(img_list):
                img_id = filename + '-' + img_name
                img_path = os.path.join(mito_segm_dir, img_name)
                graph_path = os.path.join(mito_graph_dir, img_name.replace('png', 'gpickle'))

                print("**********",img_name)
                img = cv2.imread(img_path, 0)#读入一个灰度图片
                with open(graph_path, 'rb') as f: #以二进制格式打开文件只读
                    G = pickle.load(f)#将二进制文件对象转换成 Python 对象 
                # G = nx.read_gpickle(graph_path)


                pixels=np.unique(img)   # 0, 121(interaction), 154(network),230 #去除重复进行排序

                inter_label_map=np.zeros_like(img)
                inter_pts=np.where(img==121)
                inter_label_map[inter_pts]=1


                net_label_map = np.zeros_like(img)
                net_pts = np.where(img == 154)
                net_label_map[net_pts] = 1

                xlt_label_map = np.zeros_like(img)
                xlt_pts = np.where((img == 230) + (img == 121))
                xlt_label_map[xlt_pts] = 1

                # plt.imshow(xlt_label_map)
                # plt.show()


                chull=convex_hull_image(net_label_map)
                chull_pts=np.where(chull==1)
                chull_area=len(chull_pts[0])

                mito_feat = extract_feats(chull_area,inter_label_map,net_label_map,xlt_label_map, G, img_id.replace('.png', ''), mito_sample_dir)
                feats.update(mito_feat)

        # break
    write_to_excel(feats, os.path.join(options.dst_dir, 'mito_analysis_v2.xlsx'))





if __name__ == "__main__":
    '''
    1. extract and save all mitos and thir features for the 1st frame of each video
    '''
    main()
