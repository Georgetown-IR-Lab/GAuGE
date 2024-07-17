import pandas as pd
import pyterrier as pt
pt.init()
from time import sleep
import json
from tqdm import tqdm
import ir_datasets
import openai
import os
from openai.error import RateLimitError  
import pyterrier_dr
import pyterrier_pisa
import heapq
import random

from pyterrier_t5 import MonoT5ReRanker
from rouge_score import rouge_scorer

model, prefix = 'text-davinci-edit-001', 'davinci'
LAMBDA = 10 # scaling parameter
ROUGETYPE = 'rouge2'

monoT5 = MonoT5ReRanker()  # loads castorini/monot5-base-msmarco by default
scorer = rouge_scorer.RougeScorer([ROUGETYPE], use_stemmer=True)

openai.api_key = ''

bm25 = pyterrier_pisa.PisaIndex.from_dataset('msmarco_passage').bm25(num_results=1000)
electra = pyterrier_dr.ElectraScorer(verbose=False)

pipeline = bm25 >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text') >> electra


def query_rewrite(docid, passage, query, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=passage,
                instruction='Re-write the passage to better answer the question: ' + query,
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        if 'text' not in g.keys():
            text = ''
            print('empty')
        else:
            text = g['text'].strip().replace('\n', ' ').strip()
        yield {'docno': f'{docid}-qr{idx}', 'text': text}


def rewrite(docid, passage, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=passage,
                instruction='Re-write the passage',
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        if 'text' not in g.keys():
            text = ''
            print('empty')
        else:
            text = g['text'].strip().replace('\n', ' ').strip()
        yield {'docno': f'{docid}-r{idx}', 'text': text}


def combine(docno1, docno2, passage1, passage2, query, count=1):
    while True:
        try:
            result = openai.Edit.create(
                engine='text-davinci-edit-001',
                input=f'Passage1: {passage1}\n\nPassge2: {passage2}\n\nAnswer:',
                instruction='Combine ideas from both Passage1 and Passage2 to answer the question: ' + query,
                api_key=os.getenv('OPENAI'),
                temperature=0.7,
                top_p=1,
                n=count,
            )
            break
        except openai.error.RateLimitError:
            print('RATE LIMIT, sleeping for 10 seconds')
            sleep(10)
    for idx, g in enumerate(result['choices']):
        text = ''
        if 'text' in g.keys():
            if '\n\nAnswer:' in g['text']:
                text = g['text'].split('\n\nAnswer:')[1]
        else:
            print('empty')
        yield {'docno': f'({docno1}+{docno2})-{idx}', 'text': text.replace('\n', ' ').strip()}


for DEPTH, DOC_DEPTH, no_of_mutations_per_iteration in [(2, 10, 8), (2, 2, 12)]:  # not tuned vs tuned
    # parameters recommended in Gen2IR baseline
    print(DEPTH)
    print(DOC_DEPTH)
    print(no_of_mutations_per_iteration)
    for ds, dsid in [('dl19', 'msmarco-passage/trec-dl-2019/judged'), ('dl20', 'msmarco-passage/trec-dl-2020/judged'),
                     ('dev', 'msmarco-passage/dev/small')]:

        savedf = pd.DataFrame(
            columns=['Query', 'Result_RR', 'Top2', 'MonoT5_RR', 'Result_Gen', 'MonoT5_Gen', 'Iter_Gen', 'Judge'])

        query_count = 0
        for i, query in enumerate(tqdm(ir_datasets.load(dsid).queries)):
            query_count += 1
            print(query)
            res = pipeline.search(query.text)
            # print(res['text'].iloc[2])
            original = res['text'].iloc[0]
            #print('original: ' + original)

            top1 = original
            top2 = res['text'].iloc[1]
            print('top1')
            print(top1)
            print('top2')
            print(top2)

            # Add re-ranking retrieval results to heap
            heap = [(float('-inf'), '', '')]
            for item in res.itertuples(index=False):
                scores = scorer.score(top1 + ' ' + top2, item.text)
                # print(item.score)
                # print(scores[ROUGETYPE][2])
                # print(item.score + scores[ROUGETYPE][2])
                heap.append((item.score, item.docno, item.text))
            heap = sorted(heap)

            iter = 0
            while True:
                last_heap_depth_score = heap[-1 * DEPTH][0]
                # Mutations
                res = []
                for n in range(no_of_mutations_per_iteration):
                    iter += 1
                    case = random.random()
                    try:
                        if case <= 0.33:
                            docid = int(random.random() * 100) % DOC_DEPTH + 1
                            res.extend(rewrite(heap[-1 * docid][1], heap[-1 * docid][2]))
                        elif case <= 0.66:
                            docid = int(random.random() * 100) % DOC_DEPTH + 1
                            res.extend(query_rewrite(heap[-1 * docid][1], heap[-1 * docid][2], query.text))
                        else:
                            docid1 = int(random.random() * 100) % DOC_DEPTH + 1
                            docid2 = int(random.random() * 100) % DOC_DEPTH + 1
                            if docid1 == docid2:
                                if docid1 == DOC_DEPTH:
                                    docid2 -= 1
                                else:
                                    docid2 += 1
                            res.extend(combine(heap[-1 * docid1][1], heap[-1 * docid2][1], heap[-1 * docid1][2],
                                               heap[-1 * docid2][2], query.text))
                    except:
                        print('err_er') 
                        continue
                # Evaluate new documents
                
                res = pd.DataFrame({'qid': query.query_id, 'query': query.text, 'docno': [x['docno'] for x in res],
                                    'text': [x['text'] for x in res]})
                res = electra(res)
                # print(res)

                # Add new documents to heap
                for item in res.itertuples(index=False):
                    scores = scorer.score(top1 + ' ' + top2, item.text)
                    #print(scores[ROUGETYPE][2])
                    #print(item.score)

                    #lambda predetermined for normalization
                    heap.append((item.score+ LAMBDA*scores[ROUGETYPE][2], item.docno, item.text))
                heap = sorted(heap)

                print(' '.join([str(x) for x in heap[-1]]))

                # Termination criteria
                if heap[-1 * DEPTH][0] <= last_heap_depth_score:
                    break

            print('final')
            print(heap[-1][2])

            inp1 = pd.DataFrame([['q1', query, 'd1', original]], columns=['qid', 'query', 'docno', 'text'])
            mono_score_rr = monoT5.transform(inp1).loc[0].at["score"]
            inp2 = pd.DataFrame([['q1', query, 'd1', heap[-1][2]]], columns=['qid', 'query', 'docno', 'text'])
            mono_score_gen = monoT5.transform(inp2).loc[0].at["score"]
            j = mono_score_gen - mono_score_rr
            if j > 0:
                j = 1
            elif j < 0:
                j = -1
            savedf = pd.concat([savedf, pd.DataFrame([{'Query': query, 'Result_RR': original, 'Top2': top2, 'MonoT5_RR': mono_score_rr, 'Result_Gen': heap[-1][2],
                 'MonoT5_Gen': mono_score_gen, 'Iter_Gen': iter, 'Judge': j}])], ignore_index=True)
           
        print(savedf)
        savedf.to_csv('gpt3/' + ROUGETYPE + ds + '_' + str(DEPTH) + '_' + str(DOC_DEPTH) + '_' + str(
            no_of_mutations_per_iteration) + '.csv', index=False)