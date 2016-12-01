# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 13:32:23 2016

@author: L
"""

''''
CNN 文本分类之中文预处理(分词)
E:\doc\data\SogouC.reduced.20061127\SogouC.reduced\Reduced\C000024 军事

E:\doc\data\SogouC.reduced.20061127\SogouC.reduced\Reduced\C000020 教育

分词提取实词后，词汇量仍然相对较大，在样本和计算资源有限的情况下，可以考虑选取频率较高的特征作为原始特征
'''
#encoding=utf-8
import jieba
import jieba.posseg as pseg
import os

flag_list = ['t','q','p','u','e','y','o','w','m']
def jiebafenci(all_the_text):
    re = ""
    relist = ""
    words = pseg.cut(all_the_text)
    count = 0
    for w in words:
        flag = w.flag
        tmp = w.word
        #print "org: "+tmp
        if len(tmp)>1 and len(flag)>0 and flag[0] not in flag_list and  tmp[0]>=u'/u4e00' and tmp[0]<=u'\u9fa5':
            re = re + " " + w.word
            count = count +1
        if count%100 == 0:
            print re
            re = re.replace("\n"," ")
            relist = relist + "\n" + re
            re = ""
            count = count +1
    re = re.replace("\n"," ").replace("\r"," ")   
    if len(relist)>1 and len(re)>40:
        relist = relist + "\n" + re
    elif len(re)>40:
        relist = re
    relist = relist + "\n"
    relist = relist.replace("\r\n","\n").replace("\n\n","\n")

    return relist

def getTrainData(inpath,outfile):
    fw = open(outfile,"a") 
    for filename in os.listdir(inpath):
        print filename
        file_object = open(inpath+"\\"+filename)
        try:
            all_the_text = file_object.read()
            all_the_text = all_the_text.decode("gb2312").encode("utf-8")
            pre_text = jiebafenci(all_the_text)
            if len(pre_text)>30:
                fw.write(pre_text.encode("utf-8"))
        except:
            pass
        finally:
            file_object.close()
    fw.close()
    
inpath = 'E:\doc\data\SogouC.reduced.20061127\SogouC.reduced\Reduced\C000013'
outfile = 'E:\doc\data\SogouC.reduced.20061127\SogouC.reduced\Reduced\C000013_pre.txt'
getTrainData(inpath,outfile)