import torch
import cv2
import numpy as np
import torch.nn.functional as F
import time

# define loss function
class HammingLoss(torch.nn.Module):

    def __init__(self, static_ratio=1, dynamic_ratio=1, triplet_ratio=1, 
                 semantic_ratio=1, threhold=36, ratio_threshold=0.90):
        """
        :param eta:
            0: ratio of static objects in semantic lss
            1: ratio of dynamic objects in semantic loss
            2: ratio of triplet loss
        """
        super(HammingLoss, self).__init__()
        self.static_ratio = static_ratio
        self.dynamic_ratio = dynamic_ratio
        self.triplet_ratio = triplet_ratio
        self.semantic_ratio = semantic_ratio
        self.ratio_threshold = ratio_threshold
        # number of top attention value for feature

        # threhold for triplet loss
        self.threhold = threhold

    def hamming_instance(self, positive, negative):
        """
        calculate hamming instance for positive and negative
        args:
            positive[batch, number, size]
            negative[batch, numbei, size]
        return instance[number]
        """
        instance = ((positive//128) != (negative//128)).int()
        positive = positive%128
        negative = negative%128
        for i in range(6):
            instance += ((positive//2**(6-i))!=(negative//2**(6-i))).int()
            positive = positive%2**(6-i)
            negative = negative%2**(6-i)
        instance += (positive != negative).int()
        return instance.sum(dim=-1)
    


    def forward(self, predictions, labels, indices, features):
        """
        :param mask:
            output of attentive feature filter network
        :param predictions:
            attention map from net[batch*3, 1, height, width]
        :param labels:
            seman_image[batch*3, 1, height, width]
        param indices:
            feature indices [batch*3, 1, number, 1]
        :param features:
            feature discripitors [batch*3, size, number]
        :return:
            semantic loss + triplet descriptor loss
        """
        batch_num = features.size(0)//3
        feature_number = indices.size(2)
        discrip_size = features.size(1)
        img_width =  predictions.size(-1)

        # # semantic loss
        weight = torch.ones_like(predictions) * self.static_ratio
        weight[labels==0] = self.dynamic_ratio       
        semantic_loss = F.binary_cross_entropy(predictions, labels, weight=weight, reduction='mean')

        # # triplet descriptor loss

        # prediction_sel is feature location's attention [batch*3, number]
        prediction_sel = torch.squeeze(torch.cat(tuple([torch.gather(predictions.view(batch_num*3, 1, -1), 
                                    dim=2, index=indices[:,:,i,:].long()) for i in range(feature_number)]), dim=-1), dim=1)

        # calculate pos of each position
        # x or y is [batch, number]
        location_positive = indices[batch_num:2*batch_num, 0, :, 0]
        location_negative = indices[2*batch_num:, 0, :, 0]
        location_origin = indices[0:batch_num, 0, :, 0]

        # depart ori pos and neg[batch, number, 32] ---> [batch,number,self.top_num,32]
        # ori expand in dim 1
        # neg and pos expand in dim 2
        ori_feature = features[0:batch_num].permute(0,2,1).unsqueeze(dim=1).repeat(1,feature_number,1,1)
        pos_feature = features[batch_num:2*batch_num].permute(0,2,1).unsqueeze(dim=2).repeat(1,1,feature_number,1)
        neg_feature = features[2*batch_num:3*batch_num].permute(0,2,1).unsqueeze(dim=2).repeat(1,1,feature_number,1)

        # calculate pos instance and neg instance
        positive_instance = self.hamming_instance(ori_feature, pos_feature)
        negative_instance = self.hamming_instance(ori_feature, neg_feature)
        
        # Get a best feature point in pos and neg for every feature_point in origin
        # sort [batch, number, 2]
        sort_pos_value,sort_pos_indice = torch.topk(positive_instance.float(), k=2, dim=2, largest=False)
        sort_neg_value,sort_neg_indice = torch.topk(negative_instance.float(), k=2, dim=2, largest=False)

        distance_pos = torch.tensor(0.).cuda()
        distance_neg = torch.tensor(0.).cuda()

        for i in range(batch_num):
            """
            calculate positive matrix 
            """

            # position is [3,num]
            # pred is [num]
            judge = sort_pos_value[:,:,0]<self.ratio_threshold*sort_pos_value[:,:,1]
            position = torch.cat((torch.unsqueeze(location_positive[0, sort_pos_indice[i, judge[i, :]][:,0]]//img_width, dim=-1), 
                                  torch.unsqueeze(location_positive[0, sort_pos_indice[i, judge[i, :]][:,0]]%img_width, dim=-1)), dim=-1).float()
            position = torch.cat((position, torch.ones(position.shape[0], 1).cuda()), dim=-1).t()
            pred_pos = prediction_sel[batch_num+i, sort_pos_indice[i, judge[i, :]][:,0]]

            position_ori = torch.cat((torch.unsqueeze(location_origin[i, judge[i, :]]//img_width, dim=-1), 
                                      torch.unsqueeze(location_origin[i, judge[i, :]]%img_width, dim=-1)), dim=-1).float()
            position_ori = torch.cat((position_ori, torch.ones(position_ori.shape[0], 1).cuda()), dim=-1).t()
            pred_pos_ori = prediction_sel[i, judge[i, :]]


            # 将得到的坐标进行归一化
            # position and position_ori is [num, 2]
            S = (2*position_ori.shape[1])**0.5 / (((position_ori - torch.tensor([[torch.mean(position_ori, dim=1)[0]], [torch.mean(position_ori, dim=1)[1]], [0]]).cuda())**2).sum())**0.5
            H_normal = torch.tensor([[1, 0, -torch.mean(position_ori, dim=1)[0]],
                                     [0, 1, -torch.mean(position_ori, dim=1)[1]],
                                     [0, 0, S]]).cuda()

            position_ori = torch.mm(H_normal, position_ori)
            position_ori = position_ori[0:2, :].t()

            S = (2*position.shape[1])**0.5 / (((position - torch.tensor([[torch.mean(position, dim=1)[0]], [torch.mean(position, dim=1)[1]], [0]]).cuda())**2).sum())**0.5
            H_normal = torch.tensor([[1, 0, -torch.mean(position, dim=1)[0]],
                                     [0, 1, -torch.mean(position, dim=1)[1]],
                                     [0, 0, S]]).cuda()

            position = torch.mm(H_normal, position)
            position = position[0:2, :].t()


            # https://www.cnblogs.com/wangguchangqing/p/8287585.html
            # calculate Homography with best squares
            A1_cal = torch.zeros(position.shape[0], 16).cuda()
            A1_cal[:, 0:2] = position_ori
            A1_cal[:, 2] = 1
            A1_cal[:, 6:8] = -position_ori*position[:,0:1].repeat(1,2)

            A1_cal[:, 11:13] = position_ori
            A1_cal[:, 13] = 1
            A1_cal[:, 14:16] = -position_ori*position[:,1:2].repeat(1,2)

            A1_cal = A1_cal.view(-1,8)

            # 最小二乘法的结果
            solve_pos = torch.cat((torch.mm(torch.mm(torch.mm(A1_cal.t(), A1_cal).inverse(), A1_cal.t()), position.contiguous().view(-1,1)), torch.ones(1,1).cuda()), dim=0).view(3,3)

            # 计算平均距离作为loss
            position = torch.cat((position, torch.ones(position.shape[0],1).cuda()),dim=1)
            position_ori = torch.cat((position_ori, torch.ones(position_ori.shape[0],1).cuda()),dim=1)
            scale = torch.mm(solve_pos, position_ori.t()).t()
            distance_pos += (((position - (scale/scale[:,2:3].repeat(1,3)))**2).sum(dim=1)**0.5*pred_pos_ori*pred_pos).mean()

            """
            calculate negative matrix 
            """
            # position is [3,num]
            # pred is [num]
            judge = sort_neg_value[:,:,0]<self.ratio_threshold*sort_neg_value[:,:,1]
            position = torch.cat((torch.unsqueeze(location_negative[0, sort_neg_indice[i, judge[i, :]][:,0]]//img_width, dim=-1), 
                                  torch.unsqueeze(location_negative[0, sort_neg_indice[i, judge[i, :]][:,0]]%img_width, dim=-1)), dim=-1).float()
            position = torch.cat((position, torch.ones(position.shape[0], 1).cuda()), dim=-1).t()
            pred_neg = prediction_sel[batch_num+i, sort_neg_indice[i, judge[i, :]][:,0]]

            position_ori = torch.cat((torch.unsqueeze(location_origin[i, judge[i, :]]//img_width, dim=-1), 
                                      torch.unsqueeze(location_origin[i, judge[i, :]]%img_width, dim=-1)), dim=-1).float()
            position_ori = torch.cat((position_ori, torch.ones(position_ori.shape[0], 1).cuda()), dim=-1).t()
            pred_neg_ori = prediction_sel[i, judge[i, :]]


            # 将得到的坐标进行归一化
            # position and position_ori is [num, 2]
            S = (2*position_ori.shape[1])**0.5 / (((position_ori - torch.tensor([[torch.mean(position_ori, dim=1)[0]], [torch.mean(position_ori, dim=1)[1]], [0]]).cuda())**2).sum())**0.5
            H_normal = torch.tensor([[1, 0, -torch.mean(position_ori, dim=1)[0]],
                                     [0, 1, -torch.mean(position_ori, dim=1)[1]],
                                     [0, 0, S]]).cuda()

            position_ori = torch.mm(H_normal, position_ori)
            position_ori = position_ori[0:2, :].t()

            S = (2*position.shape[1])**0.5 / (((position - torch.tensor([[torch.mean(position, dim=1)[0]], [torch.mean(position, dim=1)[1]], [0]]).cuda())**2).sum())**0.5
            H_normal = torch.tensor([[1, 0, -torch.mean(position, dim=1)[0]],
                                     [0, 1, -torch.mean(position, dim=1)[1]],
                                     [0, 0, S]]).cuda()

            position = torch.mm(H_normal, position)
            position = position[0:2, :].t()



            # https://www.cnblogs.com/wangguchangqing/p/8287585.html
            # calculate Homography with best squares
            A1_cal = torch.zeros(position.shape[0], 16).cuda()
            A1_cal[:, 0:2] = position_ori
            A1_cal[:, 2] = 1
            A1_cal[:, 6:8] = -position_ori*position[:,0:1].repeat(1,2)

            A1_cal[:, 11:13] = position_ori
            A1_cal[:, 13] = 1
            A1_cal[:, 14:16] = -position_ori*position[:,1:2].repeat(1,2)

            A1_cal = A1_cal.view(-1,8)

            # 最小二乘法的结果
            solve_neg = torch.cat((torch.mm(torch.mm(torch.mm(A1_cal.t(), A1_cal).inverse(), A1_cal.t()), position.contiguous().view(-1,1)), torch.ones(1,1).cuda()), dim=0).view(3,3)

            # 计算平均距离作为loss
            position = torch.cat((position, torch.ones(position.shape[0],1).cuda()),dim=1)
            position_ori = torch.cat((position_ori, torch.ones(position_ori.shape[0],1).cuda()),dim=1)
            scale = torch.mm(solve_pos, position_ori.t()).t()

            distance_neg += (((position - (scale/scale[:,2:3].repeat(1,3)))**2).sum(dim=1)**0.5*pred_neg_ori*pred_neg).mean()

        distance_pos, distance_neg = distance_pos/batch_num, distance_neg/batch_num
	
        # triplet_loss = distance_pos
        # print("distance_pos", distance_pos)
        triplet_loss = max(distance_pos.float() - distance_neg.float() + self.threhold, torch.tensor(0.).cuda())
        overall_loss = self.semantic_ratio*semantic_loss + self.triplet_ratio*triplet_loss

        return overall_loss, semantic_loss, triplet_loss

        
        # # test
        # H_create = torch.tensor([[1,0.,-7.],[0.,1.,-8.],[0.,0.,1]])
        # position_ori = torch.cat(((torch.rand(2,46)*500).floor(), torch.ones(1,46)), dim=0)
        # scale = torch.mm(H_create, position_ori)
        # position = (scale/scale[2:3,:].repeat(3,1)).floor()
        # print("position",position)
        # print("postion_ori", position_ori)
        
    
        # # 将得到的坐标进行归一化
        # # position and position_ori is [num, 2]
        # S = (2*position_ori.shape[1])**0.5 / (((position_ori - torch.tensor([[torch.mean(position_ori, dim=1)[0]], [torch.mean(position_ori, dim=1)[1]], [0]]))**2).sum())**0.5
        # H_normal = torch.tensor([[1, 0, -torch.mean(position_ori, dim=1)[0]],
        #                          [0, 1, -torch.mean(position_ori, dim=1)[1]],
        #                          [0, 0, S]])
        # position_ori = torch.mm(H_normal, position_ori)
        # position_ori = position_ori[0:2, :].t()
        # S = (2*position.shape[1])**0.5 / (((position - torch.tensor([[torch.mean(position, dim=1)[0]], [torch.mean(position, dim=1)[1]], [0]]))**2).sum())**0.5
        # H_normal = torch.tensor([[1, 0, -torch.mean(position, dim=1)[0]],
        #                          [0, 1, -torch.mean(position, dim=1)[1]],
        #                          [0, 0, S]])
        # position = torch.mm(H_normal, position)
        # position = position[0:2, :].t()
