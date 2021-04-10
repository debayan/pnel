from __future__ import division
import sys
import json
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz
import requests
#from annoy import AnnoyIndex
import re,random
from nltk.util import ngrams
from textblob import TextBlob
#import urllib2
from multiprocessing import Pool
#import redis
import random

#redisdb = redis.Redis()

def mean(a):
    return sum(a) / len(a)

postags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
es = Elasticsearch(port=9200)
es9800 = Elasticsearch(port=9800)

writef = open(sys.argv[2], 'w') 

cache = {}

def gettextembedding(text):
    if text in cache:
        return cache[text]
    try:
        inputjson = {'chunks':[text]}
        response = requests.post('http://localhost:8887/ftwv', json=inputjson)
        labelembedding = response.json()[0]
        cache[text] = labelembedding
        return labelembedding
    except Exception as e:
        print("getdescriptionsembedding err: ",e)
        return [0]*300
    return [0]*300

def getembedding(enturl):
    res = es.search(index="freebaseembedindex01", body={"query":{"term":{"key":{"value":enturl}}}})
    try:
        embedding = res['hits']['hits'][0]['_source']['embedding']
        return embedding
    except Exception as e:
        #print(enturl,' not found')
        return None
    return None

def gettextmatchmetric(label,word):
    return [fuzz.ratio(label,word)/100.0,fuzz.partial_ratio(label,word)/100.0,fuzz.token_sort_ratio(label,word)/100.0] 

fail = 0
def givewordvectors(inputtuple):#(id,question,entities):
    id = inputtuple[0]
    question = inputtuple[1]
    entities = [x.replace('ns:','') for x in inputtuple[2]]
    relations = [x.replace('ns:','') for x in inputtuple[3]]
    if not question:
        return []
    q = question.replace('"','')
    q = re.sub("\s*\?", "", q.strip())
    print(q)
    pos = TextBlob(q)
    chunks = pos.tags
    candidatevectors = []
    #questionembedding
    tokens = [token[0] for token in chunks]
    r = requests.post("http://localhost:8887/ftwv",json={'chunks': tokens})
    questionembeddings = r.json()
    print(question,len(questionembeddings))
    questionembedding = list(map(lambda x: sum(x)/len(x), zip(*questionembeddings)))
    true = []
    false = []
    found = set()
    for idx,chunk in enumerate(chunks):
        #n
        word = chunk[0]
        tokenembedding = questionembeddings[idx]
        posonehot = len(postags)*[0.0]
        posonehot[postags.index(chunk[1])] = 1
        esresult = es9800.search(index="freebaseentitylabelindex01", body={"query":{"multi_match":{"query":chunks[idx][0]}},"size":50})
        esresults = esresult['hits']['hits']
        if len(esresults) > 0:
            for entidx,esresult in enumerate(esresults):
                entityembedding = getembedding(esresult['_source']['uri'])
                label = esresult['_source']['freebaseLabel']
                descembedding = gettextembedding(label)
                textmatchmetric = gettextmatchmetric(label, word)
                if not all(x>0.5 for x in textmatchmetric):
                    continue
                if entityembedding and questionembedding :
                    if esresult['_source']['uri'] in entities:
                        found.add(esresult['_source']['uri'])
                        candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,1],esresult['_source']['uri'],1.0])
                    else:
                        candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,1],esresult['_source']['uri'],0.0])
        #n,n+1
        if idx < len(chunks) - 1:
            word = chunks[idx][0]+' '+chunks[idx+1][0]
            esresult = es9800.search(index="freebaseentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":50})
            esresults = esresult['hits']['hits']
            if len(esresults) > 0:
                for entidx,esresult in enumerate(esresults):
                    entityembedding = getembedding(esresult['_source']['uri'])
                    label = esresult['_source']['freebaseLabel']
                    descembedding = gettextembedding(label)
                    textmatchmetric = gettextmatchmetric(label, word)
                    if not all(x>0.5 for x in textmatchmetric):
                        continue
                    if entityembedding and questionembedding :
                        if esresult['_source']['uri'] in entities:
                            found.add(esresult['_source']['uri'])
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,2],esresult['_source']['uri'],1.0])
                        else:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,2],esresult['_source']['uri'],0.0])
        #n,n+1,n+2
        if idx < len(chunks) - 2:
            word = chunks[idx][0]+' '+chunks[idx+1][0]+' '+chunks[idx+2][0]
            esresult = es9800.search(index="freebaseentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":50})
            esresults = esresult['hits']['hits']
            if len(esresults) > 0:
                for entidx,esresult in enumerate(esresults):
                    entityembedding = getembedding(esresult['_source']['uri'])
                    label = esresult['_source']['freebaseLabel']
                    descembedding = gettextembedding(label)
                    textmatchmetric = gettextmatchmetric(label, word)
                    if not all(x>0.5 for x in textmatchmetric):
                        continue
                    if entityembedding and questionembedding :
                        if esresult['_source']['uri'] in entities:
                            found.add(esresult['_source']['uri'])
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,3],esresult['_source']['uri'],1.0])
                        else:
                            candidatevectors.append([questionembedding+tokenembedding+entityembedding+descembedding+posonehot+textmatchmetric+[entidx,idx,3],esresult['_source']['uri'],0.0])
    writef.write(json.dumps([id,item['entities'],item['relations'],candidatevectors])+'\n')
    return (id,entities,candidatevectors,found)

d = json.loads(open(sys.argv[1]).read())
#random.shuffle(d)
labelledcandidates = []
inputcandidates = []
totalents = 0
totalfound = 0
for idx,item in enumerate(d):
    print(idx,item['question'])
    inputcandidates.append((item['ID'],item['question'],item['entities'],item['relations']))
    candidatevectors = givewordvectors((item['ID'],item['question'],item['entities'],item['relations']))
    totalents += len(set(item['entities']))
    totalfound += len(candidatevectors[3])
    print("candveclen: ",len(candidatevectors[2]))
    print("gold ents: ",totalents)
    print("foundents: ",totalfound)
#pool = Pool(10)
#responses = pool.imap(givewordvectors,inputcandidates)
#count = 0
#redisdb.delete(sys.argv[3])
#for response in responses:
#    print("count = ",count)
#    count += 1
#    redisdb.rpush(sys.argv[3],json.dumps([response[0],response[1],response[2]])+'\n')

