f'''
# 数据探索

## 

标注字段的长度，最大最小平均 各种标注的数量
字频统计，是不是可以找到一些停用词？
各种标注内的字频统计

标注的位置统计，开头，结尾，前半部后半部
标注共现情况
标注所在的句子的字的情况，所在句子有相应的标注 vs. 没有相应的标注


规则特征？
涉及标注的字的 标签 vs. 非标签 的比例
'''
import numpy as np
from collections import defaultdict


tag_to_chrs = defaultdict(list)
wrd_to_num = defaultdict(int)


with open('train.txt', 'r') as f:
    for line in f:
        chucks = line.split('  ')
        for ck in chucks:
            chrs, tag = ck.split('/')
            chars = chrs.split('_')
            # 标注字段长度特征
            tag_to_chrs[tag].append(chars)

            # 字频统计
            for c in chars:
                wrd_to_num[c] += 1

with open('test.txt', 'r') as f:
    for line in f:
        chars = line.split('_')
        # 字频统计
        for c in chars:
            wrd_to_num[c] += 1

with open('corpus.txt', 'r') as f:
    for line in f:
        chars = line.split('_')
        # 字频统计
        for c in chars:
            wrd_to_num[c] += 1

# 标注字段长度特征
for tg in tag_to_chrs.keys():
    lens = [len(chrs) for chrs in tag_to_chrs[tg]]
    info = 'tag: {} | count {} | max {} | min {} | mean {:.2f} | std {:.2f}'.format(tg, max(lens), min(lens), np.mean(lens), np.std(lens))
    print(info)

# 各种标注内的字频统计
wrd_to_tag = defaultdict(defaultdict(int))
for tg in tag_to_chrs.keys():
    for chrs in tag_to_chrs[tg]:
        for c in chrs:
            wrd_to_tag[c][tg] += 1
wrd_tagfreq = [sum(x[k] for k in x.keys()) for x in wrd_to_tag.items()]
wrd_tagfreq = sorted(wrd_tagfreq, reverse=True)

# 涉及标注的字的 标签 vs. 非标签 的比例
wrd_tagratio = [(wt[0], wt[1] / wrd_to_num[wt[0]]) for wt in wrd_tagfreq]

# 字频统计，是不是可以找到一些停用词？
wrd_freq = sorted(wrd_to_num.items(), key=lambda x: x[1], reverse=True)


# 标注所在的句子的字的情况，所在句子有相应的标注 vs. 没有相应的标注
wrdtag_YN = {c: {t: {'Y': 0, 'N': 0} for t in ['a', 'b', 'c']} for c in wrd_to_num.keys()}
with open('train.txt', 'r') as f:
    for line in f:
        chucks = line.split('  ')
        tags = []
        chars = []
        for ck in chucks:
            chrs, tag = ck.split('/')
            chars += chrs.split('_')
            tags.append(tag)

        for c in chars:
            for t in ['a', 'b', 'c']:
                if t in tags:
                    wrdtag_YN[c][t]['Y'] += 1
                else:
                    wrdtag_YN[c][t]['N'] += 1

wrd_tfreq_rank = [t[0] for t in wrd_tagfreq]
wrd_yn_freq = sorted(wrdtag_YN.items(), key=lambda x: wrd_tfreq_rank.index(x[0]))
wrd_ynratio = []
for tp in wrd_yn_freq:
    yn = wrd_yn_freq[tp[0]]
    ynratio = []
    for t in ['a', 'b', 'c']:
        r = yn[t]['Y'] / (yn[t]['Y'] + yn[t]['N'])
        ynratio.append((t, r))
    wrd_ynratio.append((tp[0], ynratio))


# 标注的位置统计，开头，结尾，前半部后半部


# 标注共现情况，自己共现和别人共现，单次？多次？