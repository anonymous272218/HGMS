from rouge_score.rouge_scorer import RougeScorer


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    abstract = '\n'.join(abstract_sent_list)
    max_rouge = 0.0
    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sent_list)):
            if i in selected:
                continue
            selected_temp = selected + [i]
            sents = '\n'.join([doc_sent_list[i] for i in selected_temp])
            score = scorer.score(abstract, sents)
            rouge1 = score["rouge1"].fmeasure
            rouge2 = score["rouge2"].fmeasure

            rouge_score = rouge1 + rouge2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return selected


def get_score(doc_sent_list, abstract_sent_list):
    scorer = RougeScorer(['rouge1'], use_stemmer=True)
    abstract = '\n'.join(abstract_sent_list)
    score_list = []
    for sent in doc_sent_list:
        score = scorer.score(abstract, sent)
        rouge1 = score["rouge1"].fmeasure
        score_list.append(rouge1)

    return score_list