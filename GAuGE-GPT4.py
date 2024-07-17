import pandas as pd
from time import sleep
import json
from tqdm import tqdm
import ir_datasets
import os
import heapq
import random
from rouge_score import rouge_scorer

import pyterrier as pt
pt.init()
import pyterrier_dr
import pyterrier_pisa
from pyterrier_t5 import MonoT5ReRanker

from openai import OpenAI

ROUGETYPE = 'rouge2'
LAMBDA = 10 # scaling parameter

client = OpenAI(api_key='')

def magic(docid1, docid2, passage1, passage2, query, type):
    if type == 1:
        msg = [
        {"role": "system", "content": "You are an expert editor"},
        {"role": "user", "content": "Answer the query by re-writing this passage: " + passage1}
        ]
        docid = f'{docid1}-qr{type}'
    elif type == 2:
        msg = [
            {"role": "system", "content": "You are an expert editor trying to answer this query: " + query},
            {"role": "user", "content": "Answer the query by re-writing this passage: " + passage1}
        ]
        docid = f'{docid1}-qr{type}'
    elif type == 3:
        msg = [
        {"role": "system", "content": "You are an expert editor trying to answer this query: " + query},
        {"role": "user", "content": "Answer the query by combining ideas from these two passages: Passage 1:" + passage1 + 'Passage 2: ' + passage2}
        ]
        docid = f'{docid1}-{docid2}-qr{type}'
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        temperature = 0.7,
    )

    text = completion.choices[0].message.content
    #print(text)
    text = text.strip().replace('\n', ' ').strip()
    #print(text)
    return [{'docno': docid, 'text': text}]



electra = pyterrier_dr.ElectraScorer(verbose=False)
monoT5 = MonoT5ReRanker()  # loads castorini/monot5-base-msmarco by default
scorer = rouge_scorer.RougeScorer([ROUGETYPE], use_stemmer=True)

bm25 = pyterrier_pisa.PisaIndex.from_dataset('msmarco_passage').bm25(num_results=1000)
pipeline = bm25 >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text') >> electra



for DEPTH, DOC_DEPTH, no_of_mutations_per_iteration in [(2, 2, 12)]:  # tuned
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
            #print('original')
            #print(original)
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
                    case = random.randint(1, 3)

                    try:
                        if case == 1:
                            docid = random.randint(1, 2)
                            res.extend(magic(heap[-1 * docid][1], 0, heap[-1 * docid][2], '', '', case))
                        elif case == 2:
                            docid = random.randint(1, 2)
                            res.extend(magic(heap[-1 * docid][1], 0, heap[-1 * docid][2], '', query.text, case))
                        else:
                            docid1, docid2 = 1, 2
                            res.extend(magic(heap[-1][1], heap[-2][1], heap[-1][2],
                                               heap[-2][2], query.text, case))
                    except:
                        print('err_er')
                        continue
                # Evaluate new documents

                #print(res)
                res = pd.DataFrame({'qid': query.query_id, 'query': query.text, 'docno': [x['docno'] for x in res],
                                    'text': [x['text'] for x in res]})
                res = electra(res)

                # Add new documents to heap
                for item in res.itertuples(index=False):
                    scores = scorer.score(top1 + ' ' + top2, item.text)

                    # print(scores[ROUGETYPE][2])
                    # print(item.score)
                    # print()

                    heap.append((item.score+ LAMBDA*scores[ROUGETYPE][2], item.docno, item.text))
#                    heap.append((item.score, item.docno, item.text))
                heap = sorted(heap)
##
                #print(' '.join([str(x) for x in heap[-1]]))


                # Termination criteria
                if heap[-1 * DEPTH][0] <= last_heap_depth_score:
                    break

            print('final')
            print(heap[-1][2])
            print(iter)
            # break

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

            savedf.to_csv('gpt4/' + ROUGETYPE + ds + '.csv', index=False)
