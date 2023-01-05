from statistics import mean
from datasets import load_metric
from tqdm.auto import tqdm


class Metric():
    def __init__(self):
        self.rouge = load_metric('rouge', keep_in_memory=True)
        self.bertscore = load_metric('bertscore', keep_in_memory=True)
        self.ppl = load_metric('perplexity', keep_in_memory=True)
        self.bleu = load_metric('bleu', keep_in_memory=True)

    def add_batch(self, preds, refs):
        self.rouge.add_batch(predictions=preds, references=refs)
        self.ppl.add_batch(input_texts=preds)
        self.bertscore.add_batch(predictions=preds, references=refs)

    def add(self, pred, ref):
        self.rouge.add(prediction=pred, reference=ref)
        self.ppl.add(input_texts=pred)
        self.bertscore.add(prediction=pred, reference=ref)

    def compute(self):
        # compute rouge
        rouge_score = self.rouge.compute(use_stemmer=True)
        rouge_score = {key: value.mid.fmeasure * 100 for key, value in rouge_score.items()}
        rouge_score = {k: round(v, 4) for k, v in rouge_score.items()}

        # compute PPL
        ppl_score = self.ppl.compute(model_id='gpt2', stride=512)
        ppl_score = {k: round(v, 4) for k, v in ppl_score.items()}

        # compute bertscore
        bert_score = self.bertscore.compute(lang='en')
        bert_score = {'precision': round(mean(bert_score['precision']) * 100, 4),
                      'recall': round(mean(bert_score['recall']) * 100, 4),
                      'f1': round(mean(bert_score['f1']) * 100, 4)}
        return {
            'rouge': rouge_score,
            'ppl': ppl_score,
            'bertscore': bert_score
        }

    def compute_rouge(self, pred, ref, record=False, use_recall=False):
        res = []
        if record:
            progress_bar = tqdm(range(len(pred)))
        for p, r in zip(pred, ref):
            if record:
                progress_bar.update(1)
            rouge_score = self.rouge._compute([p], [r], use_stemmer=True)
            if use_recall:
                rouge_score = {key: value.mid.recall for key, value in rouge_score.items()}
            else:
                rouge_score = {key: value.mid.fmeasure for key, value in rouge_score.items()}
            res.append(rouge_score)
        return res

    def compute_ppl(self, pred):

        ppl_score = self.ppl._compute([pred], model_id='gpt2', stride=512)
        return ppl_score

    def compute_bertscore(self, pred, ref, model_type='roberta-large'):
        res = []
        bert_score = self.bertscore._compute(pred, ref, model_type=model_type)
        for p, r, f1 in zip(bert_score['precision'], bert_score['recall'], bert_score['f1']):
            res.append({
                'precision': p,
                'recall': r,
                'f1': f1
            })
        return res

    def compute_bleu(self, pred, ref, k=4):
        res = []
        for p, r in zip(pred, ref):
            bleu_score = self.bleu._compute([p], [r], k)
            res.append(bleu_score)
        return res


class Metric_CLS():
    def __init__(self):
        self.accuracy = load_metric('accuracy', keep_in_memory=True)
        self.precision = load_metric('precision', keep_in_memory=True)
        self.recall = load_metric('recall', keep_in_memory=True)
        self.f1 = load_metric('f1', keep_in_memory=True)

    def add_batch(self, preds, refs):
        self.accuracy.add_batch(predictions=preds, references=refs)
        self.precision.add_batch(predictions=preds, references=refs)
        self.recall.add_batch(predictions=preds, references=refs)
        self.f1.add_batch(predictions=preds, references=refs)

    def add(self, pred, ref):
        self.accuracy.add(prediction=pred, reference=ref)
        self.precision.add(prediction=pred, reference=ref)
        self.recall.add(prediction=pred, reference=ref)
        self.f1.add(prediction=pred, reference=ref)

    def compute(self, mode="binary", labels=None):

        if mode is not None and mode != "binary":
            mode = None

        accuracy = self.accuracy.compute()
        precision = self.precision.compute(average=mode, labels=labels)
        recall = self.recall.compute(average=mode, labels=labels)
        f1 = self.f1.compute(average=mode, labels=labels)

        if mode is None:
            return {
                'accuracy': round(accuracy['accuracy'], 6),
                'precision': [round(x, 6) for x in precision['precision'].tolist()],
                'recall': [round(x, 6) for x in recall['recall'].tolist()],
                'f1': [round(x, 6) for x in f1['f1'].tolist()]
            }
        else:
            return {
                'accuracy': round(accuracy['accuracy'], 6),
                'precision': round(precision['precision'], 6),
                'recall': round(recall['recall'], 6),
                'f1': round(f1['f1'], 6)
            }
