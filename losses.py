import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import threading
import time
from itertools import product

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)#取出logits关系第0个，也就是阈值
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)#取出每个实体对中前四的分数
            top_v = top_v[:, -1]#取出每一个实体对前四中分数最低的那个分数
            mask = (logits >= top_v.unsqueeze(1)) & mask#将前四之外以及不满足阈值的都pass掉
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)#将没预测出关系的实体对关系第0位设为1
        return output

    def global_inference(self, logits, num_labels=-1,rules=None,T=None,k=None,pr=None,typeids=None,hts=None):
        output = torch.zeros_like(logits).to(logits)
        R = output[0].size()[0]
        # pr[0] = 1.0/(R-1)
        # pr = [(R-1)*i for i in pr]
        th_logit = logits[:, 0].unsqueeze(1)#取出logits关系第0个，也就是阈值
        gr = torch.sigmoid(logits-th_logit)
        #Irmask = torch.zeros_like(logits).to(logits).bool()
        before = torch.log(gr)
        pr = T*(-torch.log(torch.tensor(pr).to(logits))).pow(k)
        after = torch.log(1-gr)#乘不乘pr需要测试
        Ir = (before > after).int()
        loss_select = before #新方式

        #mask掉每个实体对关系loss中排前四之后的关系，再mask掉不满足阈值的关系
        mask = (Ir.bool())
        if num_labels > 0:
            top_v, _ = torch.topk(loss_select, num_labels, dim=1)#取出每个实体对中前四的分数
            top_v = top_v[:, -1]#取出每一个实体对前四中分数最低的那个分数
            mask = (loss_select >= top_v.unsqueeze(1)) & mask#将前四之外以及不满足阈值的都pass掉
        output[mask] = 1.0
        #从目前的output中抽取出为1的位置，方便后面直接从中校验，不用遍历所有实体对，是简化版本的核心内容
        silver_label = []
        #通过规则校验output
        if len(silver_label) > 0:
            output[:, 0] = (output.sum(1) == 0.).to(logits)#将没预测出关系的实体对关系第0位设为1
            return output
        new_output = []
        e0= 0
        for elist in typeids:
            el = e0 + len(elist)*len(elist)
            bodyout = torch.reshape(output[e0:el],(len(elist),len(elist),R))
            silver_label_one = np.argwhere(bodyout.cpu()).t().tolist()
            for label in silver_label_one:#这块儿应该是要去掉头实体等于尾实体的部分
                if label[0] == label[1]:
                    silver_label_one.remove(label)
                    if output[e0:el][label[0]*len(elist)+label[1]][label[2]] == 0:
                        print('error:去除头尾实体相同标签时错误')
                    output[e0:el][label[0]*len(elist)+label[1]][label[2]] = 0.0
            new_output.append(output[e0:el])
            silver_label.append(silver_label_one)     
            e0 = el
        #多线程
        thread_list = []
        for head in rules :
            for body in rules[head]:
                for i in range(len(new_output)):
                    outone = new_output[i]
                    typeid = typeids[i]
                    l = len(typeid)
                    labels = silver_label[i]
                    t = threading.Thread(target=inferance, args=(outone,typeid,labels,head,body,l,R,threading.Lock()))
                    thread_list.append(t)
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()
        output = torch.cat(tuple(new_output))
        output[:, 0] = (output.sum(1) == 0.).to(logits)#将没预测出关系的实体对关系第0位设为1
        return output
        # th_logit = logits[:, 0].unsqueeze(1)#取出logits关系第0个，也就是阈值
        # output = torch.zeros_like(logits).to(logits)
        # gr = torch.sigmoid(logits-torch.logit(th_logit))
        # Irmask = torch.zeros_like(logits).to(logits).bool()
        # before = torch.log(gr)
        # after = T*(-torch.log(pr)).pow(k)*torch.log(1-gr)
        # loss = torch.masked_fill(input=before,mask=Irmask,value=0)+torch.masked_fill(input=after,mask=1-Irmask,value=0)
        # rev_loss = torch.masked_fill(input=before,mask=1-Irmask,value=0)+torch.masked_fill(input=after,mask=Irmask,value=0)
        # loss_all = torch.stack(loss,rev_loss,dim=0)
        # Ir = torch.argmax((loss_all),dim=0).to(logits)
        # loss_select = torch.masked_fill(input=before,mask=Ir,value=0)+torch.masked_fill(input=after,mask=1-Ir,value=0)

        # if num_labels > 0:
        #     top_v, _ = torch.topk(loss_select, num_labels, dim=1)#取出每个实体对中前四的分数
        #     top_v = top_v[:, -1]#取出每一个实体对前四中分数最低的那个分数
        #     mask = loss_select >= top_v.unsqueeze(1)#将前四之外以及不满足阈值的都pass掉
        # output[mask] = 1.0
        # output[:, 0] = (output.sum(1) == 0.).to(logits)#将没预测出关系的实体对关系第0位设为1
        # return output

class ourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, rulesconf,rulestobetaconf,ruleshead,rulestobetahead,typeids,labels,hts):
        lossconf = 0
        losshead = 0
        R = logits.size()[1]-1

        labelhits = []
        for h in hts:
            loc00 = h.index([0,0])
            labelhits.append(h[:loc00])

        for head in rulesconf:
            for i in range(len(rulesconf[head])):
                body = rulesconf[head][i]
                lossbetaconf = torch.log(torch.tensor(rulestobetaconf[head][i])).item()
                if len(body) == 1:
                    hits = []
                    revhits = []
                    count = 0
                    if body[0] > R-1:
                        for d in range(len(typeids)):
                            arrayt = np.array(typeids[d])
                            loc1 = np.where(arrayt == head[1])[0].tolist()
                            loc2 = np.where(arrayt == head[2])[0].tolist()
                            hit = list(product(loc1,loc2))
                            hit = [h for h in hit if h[0] != h[-1]]
                            if len(hit) == 0:
                                continue
                            for t in hit:
                                hits.append(count + hts[d].index(list(t)))
                                revhits.append(count + hts[d].index(list(t[::-1])))
                            count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(hits).to(labels).long())
                        rev_rulelogit = logits.index_select(0,torch.tensor(revhits).to(labels).long())
                        lossconf += ((F.logsigmoid(rev_rulelogit[:,body[0]-R+1])-F.logsigmoid(rulelogit[:,head[0]+1])).clamp(min=0.0) * (rev_rulelogit[:,body[0]-R+1] > rev_rulelogit[:,0]).float()).sum().item()+lossbetaconf
                    else:
                        for d in range(len(typeids)):
                            arrayt = np.array(typeids[d])
                            loc1 = np.where(arrayt == head[1])[0].tolist()
                            loc2 = np.where(arrayt == head[2])[0].tolist()
                            hit = list(product(loc1,loc2))
                            hit = [h for h in hit if h[0] != h[-1]]
                            if len(hit) == 0:
                                continue
                            for t in hit:
                                hits.append(count + hts[d].index(list(t)))
                            count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(hits).to(labels).long())
                        lossconf += ((F.logsigmoid(rulelogit[:,body[0]+1])-F.logsigmoid(rulelogit[:,head[0]+1])).clamp(min=0.0) * (rulelogit[:,body[0]+1] >rulelogit[:,0]).float()).sum().item()+lossbetaconf
                if len(body) == 2:
                    body1hits = []
                    body2hits = []
                    headhits = []
                    count = 0  
                    for d in range(len(typeids)):
                        arrayt = np.array(typeids[d])
                        loc1 = np.where(arrayt == head[1])[0].tolist()
                        loc2 = [i for i in range(len(typeids[d]))]
                        loc3 = np.where(arrayt == head[2])[0].tolist()
                        hit = list(product(loc1,loc2,loc3))
                        hit = [h for h in hit if h[0] != h[1] and h[1] != h[2] and h[0] != h[2] and ([h[0],h[1]] in labelhits[d] or [h[1],h[2]] in labelhits[d])]
                        if len(hit) == 0:
                            continue
                        for t in hit:
                            headhits.append(count + hts[d].index([t[0],t[2]]))
                            if body[0] > R-1:
                                body1hits.append(count + hts[d].index([t[1],t[0]]))
                            else:
                                body1hits.append(count + hts[d].index([t[0],t[1]]))
                            if body[1] > R-1:
                                body2hits.append(count + hts[d].index([t[2],t[1]]))
                            else:
                                body2hits.append(count + hts[d].index([t[1],t[2]]))
                        count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(headhits).to(labels).long())
                        body1logit = logits.index_select(0,torch.tensor(body1hits).to(labels).long())
                        body2logit = logits.index_select(0,torch.tensor(body2hits).to(labels).long())
                        if body[0] > R-1:
                            lossbase1 = F.logsigmoid(body1logit[:,body[0]-R+1])
                        else:
                            lossbase1 = F.logsigmoid(body1logit[:,body[0]+1])
                        losstheta1 = F.logsigmoid(body1logit[:,0])
                        if body[1] > R-1:
                            lossbase2 = F.logsigmoid(body2logit[:,body[1]-R+1])
                        else:
                            lossbase2 = F.logsigmoid(body2logit[:,body[1]+1])
                        losstheta2 = F.logsigmoid(body2logit[:,0])
                        losstheta = torch.stack((losstheta1,losstheta2))
                        lossbase = torch.min(lossbase1,lossbase2)
                        lossbase -= F.logsigmoid(rulelogit[:,head[0]+1])
                        losssum = torch.stack((lossbase1,lossbase2))
                        bodyloc = torch.argmin(losssum,dim=0)
                        bodytheta = losstheta.gather(0,bodyloc.unsqueeze(0))
                        lossconf = (lossbase.clamp(min=0.0)*(torch.min(lossbase1,lossbase2) > bodytheta).float()).sum().item() + lossbetaconf
        
        
        for head in ruleshead:
            for i in range(len(ruleshead[head])):
                body = ruleshead[head][i]
                lossbetahead = torch.log(torch.tensor(rulestobetahead[head][i])).item()
                if len(body) == 1:
                    hits = []
                    revhits = []
                    count = 0
                    if body[0] > R-1:
                        for d in range(len(typeids)):
                            arrayt = np.array(typeids[d])
                            loc1 = np.where(arrayt == head[1])[0].tolist()
                            loc2 = np.where(arrayt == head[2])[0].tolist()
                            hit = list(product(loc1,loc2))
                            hit = [h for h in hit if h[0] != h[-1]]
                            if len(hit) == 0:
                                continue
                            for t in hit:
                                hits.append(count + hts[d].index(list(t)))
                                revhits.append(count + hts[d].index(list(t[::-1])))
                            count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(hits).to(labels).long())
                        rev_rulelogit = logits.index_select(0,torch.tensor(revhits).to(labels).long())
                        losshead += ((-F.logsigmoid(rev_rulelogit[:,body[0]-R+1])+F.logsigmoid(rulelogit[:,head[0]+1])).clamp(min=0.0) * (rev_rulelogit[:,body[0]-R+1] <= rev_rulelogit[:,0]).float()).sum().item() + lossbetahead
                    else:
                        for d in range(len(typeids)):
                            arrayt = np.array(typeids[d])
                            loc1 = np.where(arrayt == head[1])[0].tolist()
                            loc2 = np.where(arrayt == head[2])[0].tolist()
                            hit = list(product(loc1,loc2))
                            hit = [h for h in hit if h[0] != h[-1]]
                            if len(hit) == 0:
                                continue
                            for t in hit:
                                hits.append(count + hts[d].index(list(t)))
                            count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(hits).to(labels).long())
                        losshead += ((lossbetahead-F.logsigmoid(rulelogit[:,body[0]+1])+F.logsigmoid(rulelogit[:,head[0]+1])).clamp(min=0.0) * (rulelogit[:,body[0]+1] <= rulelogit[:,0]).float()).sum().item() + lossbetahead
                if len(body) == 2:
                    body1hits = []
                    body2hits = []
                    headhits = []
                    count = 0  
                    for d in range(len(typeids)):
                        arrayt = np.array(typeids[d])
                        loc1 = np.where(arrayt == head[1])[0].tolist()
                        loc2 = [i for i in range(len(typeids[d]))]
                        loc3 = np.where(arrayt == head[2])[0].tolist()
                        hit = list(product(loc1,loc2,loc3))
                        hit = [h for h in hit if h[0] != h[1] and h[1] != h[2] and h[0] != h[2] and [h[0],h[2]] in labelhits[d]]
                        if len(hit) == 0:
                            continue
                        for t in hit:
                            headhits.append(count + hts[d].index([t[0],t[2]]))
                            if body[0] > R-1:
                                body1hits.append(count + hts[d].index([t[1],t[0]]))
                            else:
                                body1hits.append(count + hts[d].index([t[0],t[1]]))
                            if body[1] > R-1:
                                body2hits.append(count + hts[d].index([t[2],t[1]]))
                            else:
                                body2hits.append(count + hts[d].index([t[1],t[2]]))
                        count += len(hts[d])
                        rulelogit = logits.index_select(0,torch.tensor(headhits).to(labels).long())
                        body1logit = logits.index_select(0,torch.tensor(body1hits).to(labels).long())
                        body2logit = logits.index_select(0,torch.tensor(body2hits).to(labels).long())
                        if body[0] > R-1:
                            lossbase1 = F.logsigmoid(body1logit[:,body[0]-R+1])
                        else:
                            lossbase1 = F.logsigmoid(body1logit[:,body[0]+1])
                        losstheta1 = F.logsigmoid(body1logit[:,0])
                        if body[1] > R-1:
                            lossbase2 = F.logsigmoid(body2logit[:,body[1]-R+1])
                        else:
                            lossbase2 = F.logsigmoid(body2logit[:,body[1]+1])
                        losstheta2 = F.logsigmoid(body2logit[:,0])
                        losstheta = torch.stack((losstheta1,losstheta2))
                        lossbase = -torch.min(lossbase1,lossbase2)
                        lossbase += F.logsigmoid(rulelogit[:,head[0]+1])
                        losssum = torch.stack((lossbase1,lossbase2))
                        bodyloc = torch.argmin(losssum,dim=0)
                        bodytheta = losstheta.gather(0,bodyloc.unsqueeze(0))
                        losshead = (lossbase.clamp(min=0.0)*(torch.min(lossbase1,lossbase2) <= bodytheta).float()).sum().item() + lossbetahead


        loss = lossconf + losshead
        return loss


class CNSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, rules,rulestoconf,typeids,labels,hts,Eta1,Eta2):
        #typehts = [[typeid[ht[0]],typeid[ht[1]]] for ht in hts]
        newlabels = []
        newlogits = []
        newhts = []
        R = logits.size()[1]-1#关系的数量，不包括逆关系和nota
        e0= 0
        count = 0
        for elist in typeids:
            el = e0 + len(elist)*len(elist)
            split = labels[e0:el]
            newlabels.append(split[split[:,0] == 0,:])
            newlogits.append(logits[e0:el][split[:,0] == 0,:])
            newhts.append(hts[count][:(split[:,0] == 0).sum()])
            e0 = el
            count += 1
        loss = 0
        for head in rules:
            for b in range(len(rules[head])):
                body = rules[head][b]
                bodyconf = rulestoconf[head][b]
                for i in range(len(newlabels)):
                    label = newlabels[i]
                    logit = newlogits[i]
                    ht = newhts[i]
                    typeid = typeids[i]
                    for item in ht:
                        if typeid[item[0]] == head[1] and typeid[item[1]] == head[2]:
                            headloc = ht.index(item)
                            #若规则体长度为1，那么实体路径就是规则头路径
                            if len(body) == 1:
                                lossrule = torch.log(Eta1*bodyconf).to(labels)
                                if body[0] > R-1:
                                    if item[::-1] not in ht:
                                        continue
                                    loc = ht.index(item[::-1])
                                    lossrule += F.logsigmoid(logit[loc][body[0]-R+1]) - F.logsigmoid(logit[headloc][head[0]+1])
                                else:
                                    lossrule += F.logsigmoid(logit[headloc][body[0]+1]) - F.logsigmoid(logit[headloc][head[0]+1])
                                loss += torch.max(torch.tensor(0.0, dtype=torch.float).to(labels),lossrule)
                            else:#若规则体长度大于1，那么依次组合头尾，生成实体路径列表 [[2, 1], [1, 3], [3, 5], [5, 6]] 》 [[2, 1, 3], [1, 3, 5], [3, 5, 6]]
                                paths = ht
                                if body[0] > R-1:
                                    paths = [path[::-1] for path in paths]
                                for j in range(1,len(body)):
                                    if body[j] > R-1:
                                        paths = [m+[n[0]] for m in paths for n in ht if m[-1] == n[1]]
                                    else:
                                        paths = [m+[n[1]] for m in paths for n in ht if m[-1] == n[0]]
                                paths = [path for path in paths if path[0] == item[0] and path[-1] == item[1]]
                                if len(paths) == 0:#路径中没有满足规则路径的实体路径，直接掠过
                                    continue
                                for path in paths:
                                    lossrule = torch.log(Eta2*bodyconf).to(labels)
                                    for c in range(len(path)-1):
                                        if body[c] > R-1:
                                            loc = ht.index([path[c+1],path[c]])
                                            lossrule += F.logsigmoid(logit[loc][body[c]-R+1])
                                        else:
                                            loc = ht.index([path[c],path[c+1]])
                                            lossrule += F.logsigmoid(logit[loc][body[c]+1])
                                    lossrule -= F.logsigmoid(logit[loc][head[0]+1])
                                    loss += torch.max(torch.tensor(0.0, dtype=torch.float).to(labels),lossrule)
                    
        return loss

def train(logit,ht,labels,R,typeid,head,body,bodyconf,Eta1,Eta2,loss,lk):
    for item in ht:
        if typeid[item[0]] == head[1] and typeid[item[1]] == head[2]:
            headloc = ht.index(item)
            #若规则体长度为1，那么实体路径就是规则头路径
            if len(body) == 1:
                lossrule = torch.log(Eta1*bodyconf).to(labels)
                if body[0] >= R-1:
                    if item[::-1] not in ht:
                        continue
                    loc = ht.index(item[::-1])
                    lossrule += F.logsigmoid(logit[loc][body[0]-R+2]) - F.logsigmoid(logit[headloc][head[0]+1])
                else:
                    lossrule += F.logsigmoid(logit[headloc][body[0]+1]) - F.logsigmoid(logit[headloc][head[0]+1])
                lk.acquire()
                loss[0] += torch.max(torch.tensor(0.0, dtype=torch.float).to(labels),lossrule)
                lk.release()
            else:#若规则体长度大于1，那么依次组合头尾，生成实体路径列表 [[2, 1], [1, 3], [3, 5], [5, 6]] 》 [[2, 1, 3], [1, 3, 5], [3, 5, 6]]
                paths = ht
                if body[0] >= R-1:
                    paths = [path[::-1] for path in paths]
                for j in range(1,len(body)):
                    if body[j] >= R-1:
                        paths = [m+[n[0]] for m in paths for n in ht if m[-1] == n[1]]
                    else:
                        paths = [m+[n[1]] for m in paths for n in ht if m[-1] == n[0]]
                paths = [path for path in paths if path[0] == item[0] and path[-1] == item[1]]
                if len(paths) == 0:#路径中没有满足规则路径的实体路径，直接掠过
                    continue
                for path in paths:
                    lossrule = torch.log(Eta2*bodyconf).to(labels)
                    for c in range(len(path)-1):
                        if body[c] >= R-1:
                            loc = ht.index([path[c+1],path[c]])
                            lossrule += F.logsigmoid(logit[loc][body[c]-R+2])
                        else:
                            loc = ht.index([path[c],path[c+1]])
                            lossrule += F.logsigmoid(logit[loc][body[c]+1])
                    lossrule -= F.logsigmoid(logit[loc][head[0]+1])
                    lk.acquire()
                    loss[0] += torch.max(torch.tensor(0.0, dtype=torch.float).to(labels),lossrule)
                    lk.release()

def inferance(outone,typeid,labels,head,body,l,R,lk):
    if len(body) == 1:
        for label in labels:
            if body[0] >= R-1:
                if typeid[label[0]] == head[2] and typeid[label[1]] == head[1] and label[2] == body[0]-R+2:
                    lk.acquire()
                    outone[label[0]*l+label[1]][head[0]+1] = 1.0
                    lk.release()
            else:
                if typeid[label[0]] == head[1] and typeid[label[1]] == head[2] and label[2] == body[0]+1:
                    lk.acquire()
                    outone[label[0]*l+label[1]][head[0]+1] = 1.0
                    lk.release()
    elif len(body) == 2:
        if body[0] >= R-1:
            paths = [path[0:2][::-1] for path in labels if path[-1] == body[0]-R+2]
        else:
            paths = [path[0:2] for path in labels if path[-1] == body[0]+1]
        for j in range(1,len(body)):
            if body[j] >= R-1:
                paths = [m+[n[0]] for m in paths for n in labels if m[-2] == n[1] and n[2] == body[j]-R+2]
            else:
                paths = [m+[n[1]] for m in paths for n in labels if m[-2] == n[0] and n[2] == body[j]+1]
        paths = [path for path in paths if typeid[path[0]] == head[0] and typeid[path[-1]] == head[1]]

        if len(paths) == 0:#路径中没有满足规则路径的实体路径，直接掠过
            return
        else:
            for path in paths:
                lk.acquire()
                outone[path[0]*l+path[-1]][head[0]+1] = 1.0
                lk.release()