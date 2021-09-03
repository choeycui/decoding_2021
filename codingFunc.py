# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:55:12 2021

@author: cuch9001
"""


illigalWords = ['(A类)', '(PT)', '(T淘汰)', '(下)', '作废', '(删除)', '(剪角)', \
                '(换购）', '(整件)', '(整箱)', '(整箱石家庄)', '(自采)', '(赠品)', '#', '-', '@', '_']

greekConvert = {'β':'b', 'Β':'B', 'α':'a', 'Α':'A'}

import re
pattern_imdb = re.compile('^@+\d+/?\d+/?\d+,?|^\[[A-Z]*\]')
# pattern that indentify the split signal
pattern_cut = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9^\d^\.^\+]')
pattern_retailer = re.compile('^[A-Z](?=[^a-zA-Z0-9])|^([\\w^旺^Q^花])\\1+|^\*|^#|^\[[A-Z]*\]|^\([^\)]+\)\*?')
pattern_web = re.compile('^[\[\(](?=[^【^】]).*[\]\)]')

import jieba
def jiebaPrework(imdb):
    for brand in imdb['BRAND_DESC_CN']:
        jieba.add_word(str(brand))
    for manu in imdb['MANU_DESC_CN']:
        jieba.add_word(str(manu))
    for sBrand in imdb['SUBBRAND_DESC_CN']:
        jieba.add_word(str(sBrand))
    '''
    for sDesc in imdb['SHORRDESC']:
        jieba.add_word(sDesc)
    '''
    wordList = ['舒化', '富硒', '高钙', '未来星', '臻浓','臻享','臻醇','悦鲜活','谷粒多','红养','活力','颗颗','QQ星','奶特','榛果',\
                '天山雪','优舒','量贩','天喔','致优','味可滋','麦香','紫薯','弗里生','龙丹','醇壹','德拉米尔',\
                '旺仔','特仑苏','雪兰','东方多鲜','萨乌什金','生榨','手挤','滋浓','布呐呐','净纯','爵品','果粒','人人和','宁兰','抹茶',]
    for word in wordList:
        jieba.add_word(word)

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def descProcessing(itemDesc):
    itemDesc = itemDesc.replace('优+','优加')
    itemDesc = itemDesc.replace('0脂肪','零脂肪')
    itemDesc = itemDesc.replace('0乳糖','零乳糖')
    for k, v in greekConvert.items():
        itemDesc = itemDesc.replace(k, v)
    return itemDesc

def imdbDesc(itemDesc):
    itemDesc = descProcessing(itemDesc)
    itemDesc = re.sub('GM(?=\*)|GM$','G', itemDesc.upper())
    def descClean(itemDesc):
        result = pattern_imdb.search(itemDesc)
        substr = result.group() if result else ''
        return itemDesc.replace(substr, '').replace('@','')
    itemDesc1 = descClean(itemDesc).replace('+', '加')
    return re.sub(pattern_cut, ' ', itemDesc1)

def retailerDesc(itemDesc):
    itemDesc = descProcessing(itemDesc)
    def descClean1(itemDesc):
        itemDesc = itemDesc.upper()
        for word in illigalWords:
            itemDesc = itemDesc.replace(word, '')
        descList = itemDesc.split('^')
        if len(descList)==1:
            result =  descList[0]
        if len(descList)>1:
            result = descList[0].replace(descList[-1], '')
            result += descList[-1] if descList[-1] in descList[0] else ''
        return re.sub(pattern_retailer, '', result)
    itemDesc = descClean1(itemDesc)
    return re.sub(pattern_cut, ' ', itemDesc)

def webDesc(itemDesc):
    itemDesc = strQ2B(itemDesc)
    if re.search('^【(?=[^【^】]).*】$', itemDesc):
        itemDesc = itemDesc.replace('【', '').replace('】', '')
    itemDesc = itemDesc.replace('【', '[').replace('】', ']')
    itemDesc = itemDesc.replace("（", "(").replace("）",')')
    re.sub(pattern_web, '', itemDesc)
    itemDesc = retailerDesc(itemDesc)
    itemDesc = imdbDesc(itemDesc)
    return itemDesc

import Levenshtein
from difflib import SequenceMatcher
import numpy as np
from collections import Counter

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def similarity(desc, brand, itemDict):
    subDict = itemDict.loc[itemDict['BRAND_DESC_CN']==brand, ['NANKEY', 'ITEM_SEQ']].copy()
    size = len(subDict['NANKEY'].unique())
    if size < 1:
        return np.nan
    else:
        poolSize = min(3, size)
        subDict['JaroRatio'] = subDict['ITEM_SEQ'].apply(lambda x: Levenshtein.ratio(x,desc))
        subDict['Jaro'] = subDict['ITEM_SEQ'].apply(lambda x: Levenshtein.jaro(x,desc))
        subDict['JaroWinkler'] = subDict['ITEM_SEQ'].apply(lambda x: Levenshtein.jaro_winkler(x,desc))
        subDict['Ratio'] = subDict['ITEM_SEQ'].apply(lambda x: similar(x,desc))

        JaroRatioWinner = subDict.groupby('NANKEY')['JaroRatio'].max().sort_values(ascending=False).index[:poolSize].to_list()
        JaroWinner = subDict.groupby('NANKEY')['Jaro'].max().sort_values(ascending=False).index[:poolSize].to_list()
        JaroWinklerWinner = subDict.groupby('NANKEY')['JaroWinkler'].max().sort_values(ascending=False).index[:poolSize].to_list()
        RatioWinner = subDict.groupby('NANKEY')['Ratio'].max().sort_values(ascending=False).index[:poolSize].to_list()

        itemSet = JaroRatioWinner + JaroWinner + JaroWinklerWinner + RatioWinner
        return Counter(itemSet).most_common(1)[0][0]
