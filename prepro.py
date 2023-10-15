from collections import Counter, defaultdict
import os
from tqdm import tqdm
import ujson as json
import torch
from scipy.stats import beta
import numpy as np

docred_rel2id = json.load(open('dataset/dwie/meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

class OurRuleReader(object):
    'read text feature'
    """read and store DocRED data"""
    def __init__(self, data_dir, save_dir, max_step=3) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.rel2id = {k: v-1 for k,v in json.load(open(os.path.join(data_dir, 'meta/rel2id.json'))).items()}
        self.id2rel = {k:v for v, k in self.rel2id.items()}
        self.R = len(self.rel2id) - 1
        self.type2id = json.load(open(os.path.join(data_dir, 'meta/ner2id.json')))
        self.id2type = {k:v for v, k in self.type2id.items()}
        self.betatheta = 0.9

        self.data_paths = {
            'rtrain': os.path.join(data_dir, 'rtrain.json'),
            'train': os.path.join(data_dir, 'train_annotated.json'),
            'dist': os.path.join(data_dir, 'train_distant.json'),
            'dev': os.path.join(data_dir, 'dev.json'),
            'test': os.path.join(data_dir, 'test.json')
        }
        self.bin_paths = {
            'rtrain': os.path.join(save_dir, 'cooccur-rtrain.pth'),
            'train': os.path.join(save_dir, 'cooccur-train.pth'),
            'dist': os.path.join(save_dir, 'cooccur-dist.pth'),
            'dev': os.path.join(save_dir, 'cooccur-dev.pth'),
            'test': os.path.join(save_dir, 'cooccur-test.pth')
        }
        self.max_step = max_step

    def read(self, split='train',betatheta=0.9):
        bin_path = self.bin_paths[split]
        if os.path.exists(bin_path):
            return torch.load(bin_path)
        else:
            self.betatheta = betatheta
            features = self.read_raw(split)
            torch.save(features, bin_path)
            return features

    def read_raw(self, split='train'):
        """count co-occurence info"""
        max_step = self.max_step
        r2epair = self.get_r2epair()
        rule_counter = {(i, h, t): Counter() for i in range(self.R) for (h, t) in r2epair[i]}

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            paths = {}
            meta_paths = {1: paths}

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                if h not in paths:
                    paths[h] = {t: [([r], [t])]}
                elif t not in paths[h]:
                    paths[h][t] = [([r], [t])]
                else:
                    paths[h][t].append(([r], [t]))

                if t not in paths:
                    paths[t] = {h: [([r + self.R], [h])]}
                elif h not in paths[t]:
                    paths[t][h] = [([r + self.R], [h])]
                else:
                    paths[t][h].append(([r + self.R], [h]))

            for step in range(2, max_step + 1):
                prev_paths = meta_paths[step - 1]
                paths = {}
                for h in prev_paths:
                    for inode, prev_chain in prev_paths[h].items():#inode是尾实体，chain是关系加尾实体的元组
                        if inode in meta_paths[1]:
                            for t, rs in meta_paths[1][inode].items():
                                if h == t:
                                    continue
                                new_chain = append_chain(prev_chain, rs)
                                if not new_chain:
                                    continue
                                if h not in paths:
                                    paths[h] = {t: new_chain}
                                elif t not in paths[h]:
                                    paths[h][t] = new_chain
                                else:
                                    paths[h][t].extend(new_chain)
                meta_paths[step] = paths

            for h in meta_paths[1]:
                for t, rs in meta_paths[1][h].items():
                    c_meta_paths = set()
                    for step in range(1, max_step + 1):
                        if h in meta_paths[step] and t in meta_paths[step][h]:
                            for path in meta_paths[step][h][t]:
                                c_meta_paths.add(tuple(path[0]))
                    for r in rs:
                        if r[0][0] >= self.R:
                            continue
                        triple = (r[0][0], entity_types[h], entity_types[t])
                        rule_counter[triple].update(c_meta_paths)
        
        triples = []
        triple2rules = {}
        triple2probs = {}
        lens = [len(epair) for epair in r2epair]
        for ri, epairs in enumerate(r2epair):
            for epair in epairs:
                triple = (ri, epair[0], epair[1])
                total = sum(rule_counter[triple].values())
                rules, probs = [], []
                for rule in rule_counter[triple]:
                    #改动：取出规则集合中规则体长度为1，且与规则头相同，又非逆关系的关系
                    if len(rule) == 1 and rule[0] == triple[0]:
                        continue
                    rules.append(rule)
                    probs.append(rule_counter[triple][rule] / total)

                triples.append(triple)
                triple2rules[triple] = rules
                triple2probs[triple] = probs
        ##改动，求置信度，而且不是规则概率，得到规则集合后，在每个文档中用规则集合来验证
        triple2counter = {}
        triple2support = {}
        triple2head = {}
        for item in tqdm(data,desc='calculate the confidence'):
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]
            label_facts = []
            #取出单个文件中的标签事实，当作已有的三元组，查看这些三元组是否满足规则集合中的规则体，又是否真的映射到了规则头
            for fact in item['labels']:
                #h, t, r = entity_types[fact['h']], entity_types[fact['t']], self.rel2id[fact['r']] #这里转换类型会导致后续无法确定是否在一个文档中规则体和规则头是否指向同一个实体对
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']] 
                label_facts.append((r,h,t))
                label_facts.append((r+self.R,t,h))
            #长度为1，2，3的规则分别计算，这里要求首尾实体类型一致，相邻俩个规则体谓词的相接处实体类型一致，谓词顺序也要与规则中相同，这里是对于每个文档偏离所有规则进行支持度和counter计算
            for head in triples:
                for i in range(len(triple2rules[head])):
                    body = triple2rules[head][i]
                    if head not in triple2counter:
                        triple2counter[head] = [0 for _ in range(len(triple2rules[head]))]
                        triple2support[head] = [0 for _ in range(len(triple2rules[head]))]
                        triple2head[head] = [0 for _ in range(len(triple2rules[head]))]
                    if len(body) == 1:
                        for t in label_facts:
                            if (t[0],entity_types[t[1]],entity_types[t[2]]) == (body[0],head[1],head[2]):
                                if (head[0],t[1],t[2]) in label_facts:
                                    triple2support[head][i] = triple2support[head][i]+1
                                else:
                                    triple2counter[head][i] = triple2counter[head][i]+1
                            if (t[0],entity_types[t[1]],entity_types[t[2]]) == head:
                                if (body[0],t[1],t[2]) not in label_facts:
                                    triple2head[head][i] = triple2head[head][i]+1
                            
                    if len(body) == 2:
                        label_facts_2 = [(m[1],n[-1]) for m in label_facts for n in label_facts if m[-1] == n[1] and m[0] == body[0] and n[0] == body[1]]
                        for t in label_facts_2:
                            if entity_types[t[0]] == head[1] and entity_types[t[1]] == head[2]:
                                if (head[0],t[0],t[1]) in label_facts:
                                    triple2support[head][i] = triple2support[head][i] +1
                                else:
                                    triple2counter[head][i] = triple2counter[head][i] +1
                        for t in label_facts:
                            if (t[0],entity_types[t[1]],entity_types[t[2]]) == head:
                                if (t[1],t[2]) not in label_facts_2:
                                    triple2head[head][i] = triple2head[head][i]+1
                    #暂时用不到长度为3的规则
                    # if len(body) == 3:
                    #     for t in label_facts:
                    #         if t[0] == body[0] and t[1] == head[1]:
                    #             for t2 in label_facts:
                    #                 if t2[0] == body[1] and t2[1] == t[2] and (body[2],t2[2],head[2]) in label_facts:
                    #                     if head in label_facts:
                    #                         triple2support[head][i] = triple2support[head][i] +label_facts.count((body[2],t2[2],head[2]))
                    #                     else:
                    #                         triple2counter[head][i] = triple2counter[head][i] +label_facts.count((body[2],t2[2],head[2]))
        #置信度计算，conf = supp/(supp+counter)
        triple2conf = {}
        triple2betaconf = {}
        triple2betahead = {}
        x = np.linspace(0, 1, 100)[1:-1]
        for head in triples:
            conf = []
            betaconf = []
            betahead = []
            for i in range(len(triple2rules[head])):
                y1 = beta.sf(x,triple2support[head][i]+1,triple2counter[head][i]+1) 
                y2 = beta.sf(x,triple2support[head][i]+1,triple2head[head][i]+1)
                betatheta = self.betatheta*100
                betaconf.append(y1[int(betatheta)])
                betahead.append(y2[int(betatheta)])
                conf.append(triple2support[head][i] / (triple2support[head][i]+triple2counter[head][i]))
            triple2conf[head] = conf
            triple2betaconf[head] = betaconf
            triple2betahead[head] = betahead

        features = {
            'triples': triples,
            'sections': lens,
            'triple2rules': triple2rules,
            'triple2probs': triple2probs,
            'triple2conf':triple2conf,
            'triple2betaconf':triple2betaconf,
            'triple2betahead':triple2betahead
        }

        return features

    def get_r2epair(self):
        r2epair = [[] for _ in range(len(self.rel2id)-1)]
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = entity_types[fact['h']], entity_types[fact['t']], self.rel2id[fact['r']]
                if (h,t) not in r2epair[r]:
                    r2epair[r].append((h, t))

        return r2epair

    def get_epair2r(self):
        e_pair2r = torch.zeros(len(self.type2id), len(self.type2id), len(self.rel2id)-1).bool()
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                e_pair2r[entity_types[h], entity_types[t], r] = 1
        print(e_pair2r.size(), e_pair2r.sum())
        return e_pair2r

    def get_type_mask(self, triples, sections, split='train'):
        ntypes = len(self.type2id)
        rpair2id = [{} for _ in sections]
        tid = 0
        for section in sections:
            for sid in range(section):
                r, e1, e2 = triples[tid]
                rpair2id[r][(e1, e2)] = sid
                tid += 1

        triple2sid = torch.CharTensor(ntypes, ntypes, self.R).fill_(-1)
        for ei in range(ntypes):
            for ej in range(ntypes):
                for r in range(self.R):
                    triple2sid[ei, ej, r] = rpair2id[r].get((ei, ej), -1)

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        type_masks = []
        for item in data:
            entities = item['vertexSet']
            N = len(entities)
            entity_types = torch.tensor([self.type2id[e[0]['type']] for e in entities])
            type_indices = (entity_types.unsqueeze(1).repeat(1, N), entity_types.unsqueeze(0).repeat(N, 1))
            type_mask = triple2sid[type_indices[0], type_indices[1]]
            type_masks.append(type_mask)
        
        return type_masks

    def get_dist(self, split='train'):
        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        dists = []
        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            N = len(entities)
            entities_pos = []
            for entity in entities:
                s = entity[0]['pos'][0]
                e = entity[0]['pos'][1]
                entities_pos.append([s, e])
            dist = torch.zeros(N, N)
            for h in range(N):
                for t in range(N):
                    sh, eh = entities_pos[h]
                    st, et = entities_pos[t]
                    dist[h,t] = min(abs(sh - et), abs(st - eh))
            dists.append(dist)
        return dists

class ERuleReader(object):
    'read text feature'
    """read and store DocRED data"""
    def __init__(self, data_dir, save_dir, max_step=3) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.rel2id = {k: v-1 for k,v in json.load(open(os.path.join(data_dir, 'meta/rel2id.json'))).items()}
        self.id2rel = {k:v for v, k in self.rel2id.items()}
        self.R = len(self.rel2id) - 1
        self.type2id = json.load(open(os.path.join(data_dir, 'meta/ner2id.json')))
        self.id2type = {k:v for v, k in self.type2id.items()}

        self.data_paths = {
            'rtrain': os.path.join(data_dir, 'rtrain.json'),
            'train': os.path.join(data_dir, 'train_annotated.json'),
            'dist': os.path.join(data_dir, 'train_distant.json'),
            'dev': os.path.join(data_dir, 'dev.json'),
            'test': os.path.join(data_dir, 'test.json')
        }
        self.bin_paths = {
            'rtrain': os.path.join(save_dir, 'cooccur-rtrain.pth'),
            'train': os.path.join(save_dir, 'cooccur-train.pth'),
            'dist': os.path.join(save_dir, 'cooccur-dist.pth'),
            'dev': os.path.join(save_dir, 'cooccur-dev.pth'),
            'test': os.path.join(save_dir, 'cooccur-test.pth')
        }
        self.max_step = max_step

    def read(self, split='train'):
        bin_path = self.bin_paths[split]
        if os.path.exists(bin_path):
            return torch.load(bin_path)
        else:
            features = self.read_raw(split)
            torch.save(features, bin_path)
            return features

    def read_raw(self, split='train'):
        """count co-occurence info"""
        max_step = self.max_step
        r2epair = self.get_r2epair()
        rule_counter = {(i, h, t): Counter() for i in range(self.R) for (h, t) in r2epair[i]}

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            paths = {}
            meta_paths = {1: paths}

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                if h not in paths:
                    paths[h] = {t: [([r], [t])]}
                elif t not in paths[h]:
                    paths[h][t] = [([r], [t])]
                else:
                    paths[h][t].append(([r], [t]))

                if t not in paths:
                    paths[t] = {h: [([r + self.R], [h])]}
                elif h not in paths[t]:
                    paths[t][h] = [([r + self.R], [h])]
                else:
                    paths[t][h].append(([r + self.R], [h]))

            for step in range(2, max_step + 1):
                prev_paths = meta_paths[step - 1]
                paths = {}
                for h in prev_paths:
                    for inode, prev_chain in prev_paths[h].items():
                        if inode in meta_paths[1]:
                            for t, rs in meta_paths[1][inode].items():
                                if h == t:
                                    continue
                                new_chain = append_chain(prev_chain, rs)
                                if not new_chain:
                                    continue
                                if h not in paths:
                                    paths[h] = {t: new_chain}
                                elif t not in paths[h]:
                                    paths[h][t] = new_chain
                                else:
                                    paths[h][t].extend(new_chain)
                meta_paths[step] = paths

            for h in meta_paths[1]:
                for t, rs in meta_paths[1][h].items():
                    c_meta_paths = set()
                    for step in range(1, max_step + 1):
                        if h in meta_paths[step] and t in meta_paths[step][h]:
                            for path in meta_paths[step][h][t]:
                                c_meta_paths.add(tuple(path[0]))
                    for r in rs:
                        if r[0][0] >= self.R:
                            continue
                        triple = (r[0][0], entity_types[h], entity_types[t])
                        rule_counter[triple].update(c_meta_paths)
        
        triples = []
        triple2rules = {}
        triple2probs = {}
        lens = [len(epair) for epair in r2epair]
        for ri, epairs in enumerate(r2epair):
            for epair in epairs:
                triple = (ri, epair[0], epair[1])
                total = sum(rule_counter[triple].values())
                rules, probs = [], []
                for rule in rule_counter[triple]:
                    #改动：取出规则集合中规则体长度为1，且与规则头相同，又非逆关系的关系
                    if len(rule) == 1 and rule[0] == triple[0]:
                        continue
                    rules.append(rule)
                    probs.append(rule_counter[triple][rule] / total)

                triples.append(triple)
                triple2rules[triple] = rules
                triple2probs[triple] = probs
        ##改动，求置信度，而且不是规则概率，得到规则集合后，在每个文档中用规则集合来验证
        triple2counter = {}
        triple2support = {}
        for item in tqdm(data,desc='calculate the confidence'):
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]
            label_facts = []
            #取出单个文件中的标签事实，当作已有的三元组，查看这些三元组是否满足规则集合中的规则体，又是否真的映射到了规则头
            for fact in item['labels']:
                #h, t, r = entity_types[fact['h']], entity_types[fact['t']], self.rel2id[fact['r']]
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']] 
                label_facts.append((r,h,t))
                label_facts.append((r+self.R,t,h))
            #长度为1，2，3的规则分别计算，这里要求首尾实体类型一致，相邻俩个规则体谓词的相接处实体类型一致，谓词顺序也要与规则中相同，这里是对于每个文档偏离所有规则进行支持度和counter计算
            for head in triples:
                for i in range(len(triple2rules[head])):
                    body = triple2rules[head][i]
                    if head not in triple2counter:
                        triple2counter[head] = [0 for _ in range(len(triple2rules[head]))]
                        triple2support[head] = [0 for _ in range(len(triple2rules[head]))]
                    if len(body) == 1:
                        for t in label_facts:
                            if (t[0],entity_types[t[1]],entity_types[t[2]]) == (body[0],head[1],head[2]):
                                if (head[0],t[1],t[2]) in label_facts:
                                    triple2support[head][i] = triple2support[head][i]+1
                                else:
                                    triple2counter[head][i] = triple2counter[head][i]+1
                            
                    if len(body) == 2:
                        label_facts_2 = [(m[1],n[-1]) for m in label_facts for n in label_facts if m[-1] == n[1] and m[0] == body[0] and n[0] == body[1]]
                        for t in label_facts_2:
                            if entity_types[t[0]] == head[1] and entity_types[t[1]] == head[2]:
                                if (head[0],t[0],t[1]) in label_facts:
                                    triple2support[head][i] = triple2support[head][i] +1
                                else:
                                    triple2counter[head][i] = triple2counter[head][i] +1
                    # if len(body) == 1:
                    #     for t in label_facts:
                    #         if t == (body[0],head[1],head[2]):
                    #             if head in label_facts:
                    #                 triple2support[head][i] = triple2support[head][i]+1
                    #             else:
                    #                 triple2counter[head][i] = triple2counter[head][i]+1
                    # if len(body) == 2:
                    #     for t in label_facts:
                    #         if t[0] == body[0] and t[1] == head[1] and (body[1],t[2],head[2]) in label_facts:
                    #             if head in label_facts:
                    #                 triple2support[head][i] = triple2support[head][i] +label_facts.count((body[1],t[2],head[2]))
                    #             else:
                    #                 triple2counter[head][i] = triple2counter[head][i] +label_facts.count((body[1],t[2],head[2]))
                    if len(body) == 3:
                        for t in label_facts:
                            if t[0] == body[0] and t[1] == head[1]:
                                for t2 in label_facts:
                                    if t2[0] == body[1] and t2[1] == t[2] and (body[2],t2[2],head[2]) in label_facts:
                                        if head in label_facts:
                                            triple2support[head][i] = triple2support[head][i] +label_facts.count((body[2],t2[2],head[2]))
                                        else:
                                            triple2counter[head][i] = triple2counter[head][i] +label_facts.count((body[2],t2[2],head[2]))
        #置信度计算，conf = supp/(supp+counter)
        triple2conf = {}
        for head in triples:
            conf = []
            for i in range(len(triple2rules[head])):
                conf.append(triple2support[head][i] / (triple2support[head][i]+triple2counter[head][i]))
            triple2conf[head] = conf

        features = {
            'triples': triples,
            'sections': lens,
            'triple2rules': triple2rules,
            'triple2probs': triple2probs,
            'triple2conf':triple2conf,
        }

        return features

    def get_r2epair(self):
        r2epair = [[] for _ in range(len(self.rel2id)-1)]
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = entity_types[fact['h']], entity_types[fact['t']], self.rel2id[fact['r']]
                if (h,t) not in r2epair[r]:
                    r2epair[r].append((h, t))

        return r2epair

    def get_epair2r(self):
        e_pair2r = torch.zeros(len(self.type2id), len(self.type2id), len(self.rel2id)-1).bool()
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                e_pair2r[entity_types[h], entity_types[t], r] = 1
        print(e_pair2r.size(), e_pair2r.sum())
        return e_pair2r

    def get_type_mask(self, triples, sections, split='train'):
        ntypes = len(self.type2id)
        rpair2id = [{} for _ in sections]
        tid = 0
        for section in sections:
            for sid in range(section):
                r, e1, e2 = triples[tid]
                rpair2id[r][(e1, e2)] = sid
                tid += 1

        triple2sid = torch.CharTensor(ntypes, ntypes, self.R).fill_(-1)
        for ei in range(ntypes):
            for ej in range(ntypes):
                for r in range(self.R):
                    triple2sid[ei, ej, r] = rpair2id[r].get((ei, ej), -1)

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        type_masks = []
        for item in data:
            entities = item['vertexSet']
            N = len(entities)
            entity_types = torch.tensor([self.type2id[e[0]['type']] for e in entities])
            type_indices = (entity_types.unsqueeze(1).repeat(1, N), entity_types.unsqueeze(0).repeat(N, 1))
            type_mask = triple2sid[type_indices[0], type_indices[1]]
            type_masks.append(type_mask)
        
        return type_masks

    def get_dist(self, split='train'):
        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        dists = []
        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            N = len(entities)
            entities_pos = []
            for entity in entities:
                s = entity[0]['pos'][0]
                e = entity[0]['pos'][1]
                entities_pos.append([s, e])
            dist = torch.zeros(N, N)
            for h in range(N):
                for t in range(N):
                    sh, eh = entities_pos[h]
                    st, et = entities_pos[t]
                    dist[h,t] = min(abs(sh - et), abs(st - eh))
            dists.append(dist)
        return dists

def append_chain(chains, rs):
    ret = []
    for chain, chain_nodes in chains:
        for r, rnode in rs:
            if rnode[0] not in chain_nodes:
                ret.append((chain + r, chain_nodes + rnode))
    return ret

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def Rule_Miner(data_dir,conf,length):
    rules = {}
    rulestoconf = {}
    rg_reader = ERuleReader(
            data_dir,
            os.path.join(data_dir, 'MILR/cooccur-data'),
            max_step=length
        )
    rule_data = rg_reader.read()
    for head in rule_data['triples']:
        body = []
        ruleconf = []
        for i in range(len(rule_data['triple2conf'][head])):
            if rule_data['triple2conf'][head][i] > conf:
                body.append(rule_data['triple2rules'][head][i])
                ruleconf.append(rule_data['triple2conf'][head][i])
        #如果该规则头三元组下没有符合要求的规则体，那么就不用加入这个规则头了
        if len(body) == 0:
            continue
        rules[head] = body
        rulestoconf[head] = ruleconf
    return rules,rulestoconf

def Rule_Miner_our(data_dir,minbetaconf,minbetahead,length,betatheta):
    ruleshead = {}
    rulesconf = {}
    rulestobetaconf = {}
    rulestobetahead = {}
    rg_reader = OurRuleReader(
            data_dir,
            os.path.join(data_dir, 'our/cooccur-data_'+str(betatheta)),
            max_step=length
        )
    rule_data = rg_reader.read(betatheta=betatheta)
    for head in rule_data['triples']:
        bodyconf = []
        ruleconf = []
        bodyhead = []
        rulehead = []
        for i in range(len(rule_data['triple2conf'][head])):
            if rule_data['triple2betaconf'][head][i] > minbetaconf:
                bodyconf.append(rule_data['triple2rules'][head][i])
                ruleconf.append(rule_data['triple2betaconf'][head][i])
            if rule_data['triple2betahead'][head][i] > minbetahead:
                bodyhead.append(rule_data['triple2rules'][head][i])
                rulehead.append(rule_data['triple2betahead'][head][i])
        #如果该规则头三元组下没有符合要求的规则体，那么就不用加入这个规则头了
        if len(bodyconf) != 0:
            rulesconf[head] = bodyconf
            rulestobetaconf[head] = ruleconf
        if len(bodyhead) != 0:
            ruleshead[head] = bodyhead
            rulestobetahead[head] = rulehead
    return rulesconf,rulestobetaconf,ruleshead,rulestobetahead

def read_docred(file_in, tokenizer, max_seq_length=1024,data_dir=None):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    #pr就是关系的频率，
    pr = [0 for i in range(len(docred_rel2id))]
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    # COUNT = 0
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        # #测试取四个样例
        # COUNT += 1
        # if COUNT == 4: break
        entities = sample['vertexSet']
        type2id = json.load(open(os.path.join(data_dir, 'meta/ner2id.json')))
        entity_types = [type2id[e[0]['type']] for e in entities]
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                pr[r] = pr[r]+1
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts, count = [], [], 0
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if [h, t] not in hts:#删去了h != t这个条件让所有自反关系也参与进来，没有自反关系就设为反例
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * len(entities)#这里后面不用减一了

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'type_id':entity_types
                   }
        features.append(feature)

    ll = sum(pr)
    if ll != 0:
        pr = [i/ll for i in pr]
    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features,pr


def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
