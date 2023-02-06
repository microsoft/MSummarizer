# OLDS

[Towards Understanding Omission in Dialogue Summarization](https://arxiv.org/abs/2211.07145) by Yicheng Zou, Kaitao Song, Xu Tan, Zhongkai Fu, Tao Gui, Qi Zhang, Dongsheng Li, is a study to understand **Omission Problem** in Dialogue Summarization. We present a definition of Omission information at the sentence level, and give an automatic labelling pipeline to obtain omission information from the omission and its summarization, and our collected dataset is called **OLDS**. We introduce different solutions to address omission detection and how to utilize omission information to improve model performance.

## Environments

* Python 3.9.12
 
* pytorch 1.12.1

* transformers 4.21.1

* datasets 2.0.0

* bert_score 0.3.11

* rouge_score 0.1.2

* accelerate 0.12.0

* nltk ('punkt')

* sklearn

* RTX V100 GPU & A100 GPU

* CUDA 11.7

## Data

The Omission Detection Dataset OLDS in our work is available at [Google Drive]() or [Baidu Pan  (extract code: xxxx)](), including sub-domains SAMSum, DialogSum, EmailSum, QMSum, and TweetSumm.

* Download the OLDS dataset from the above data links and put them into the directory **data** like this:

	```
	--- data
	  |
	  |--- dialogsum
	  |
      |--- emailsum
      |
      |--- qmsum
      |
      |--- samsum
      |
      |--- tweetsumm
	```

* We could load these sub-datasets via the commands of [datasets tool (HuggingFace)](https://huggingface.co/docs/datasets/index).

	```python
        from datasets import load_from_disk
        samsum = load_from_disk("data/samsum/omission.save")
	```

* (Optional) You can generate candidate summaries from scratch via the scripts in the folder **process**.

    * Pre-process raw datasets: 
        
        **process_data.ipynb**

    * Train models and generate candidate summaries:
        ```
        sh main_DATASET_NAME.sh
        ```

    * Produce omission labels automatically:
        ```
        sh process.sh
        ```
## Usage

* Omission Detection

    * Training:
        ```
        sh src/train.sh
        ```
    * Testing:
        ```
        sh src/test.sh
        ```
    * Predict Omissions and save the prediction results:

        ```
        sh src/predict.sh
        ```
* Post-edit Refinement

    * Training
        ```
        sh src/post_edit_train.sh
        ```

    * Testing
        ```
        sh src/post_edit_test.sh
        ```

## Citation

    @article{
    
    }
