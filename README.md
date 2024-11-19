# BERTopicSum  

**BERTopicSum** is a modified version of the BERTSum model, designed specifically for abstractive text summarization in the Indonesian language. This project was developed as part of a thesis and focuses on improving summarization performance by incorporating topic representations.  

## 🛠 Key Features and Differences  
1. **Indonesian Dataset**:  
   - BERTopicSum is trained and tested on the Indonesian dataset  [**XLSum**](https://github.com/csebuetnlp/xl-sum), whereas the original BERTSum was based on English datasets. 

2. **Abstractive Summarization**:  
   - The model exclusively focuses on abstractive summarization tasks.  

3. **Integration with BERTopic**:  
   - BERTopic is utilized to extract topics from the input documents.  
   - The extracted topics are converted into topic representations, which are then added as an additional input to the decoder.  

4. **Addressing Input Token Limitations**:  
   - BERT (used as the encoder in BERTSum) has a fixed input token limit, causing long documents to be truncated and leading to poorer summaries for these cases.  
   - By incorporating topic representations, BERTopicSum compensates for truncated information, providing additional semantic context to improve summary quality.  

## 🚀 Purpose  
The primary aim of BERTopicSum is to enhance abstractive summarization for Indonesian texts by leveraging topic modeling to address input token limitations in BERT-based models. 

$$ 📈 Results
Results on XLSum-Indonesia dataset (4/9/2024):

|  Models| ROUGE-1 | ROUGE-2 |ROUGE-L		
| :---         |     :---      |         :--- |          :--- |
| BERTSum (IndoBERT)   | 24.56     | 8.49    |19.88    |
| BERTopicSum (IndoBERT)     | 25.39       | 9.16    |20.61      |

Results on XLSum-Indonesia dataset (long documents only) (4/9/2024): 		

|  Models| ROUGE-1 | ROUGE-2 |ROUGE-L		
| :---         |     :---      |         :--- |          :--- |
| BERTSum (IndoBERT)   | 21.57     | 6.72    |16.99    |
| BERTopicSum (IndoBERT)     | 22.11       | 7.37    |17.53      |

- BERTopicSum achieves 4.71% improvement of overall average ROUGE scores than the baseline model.
- The integration of BERTopic successfully improved summarization performance for long documents by addressing BERT's input token limitation.

## 📂 Repository Structure  
- **`raw_data/`**: Contains the raw XLSum dataset  
- **`models/`**: Contains fine-tuned BERTSum and BERTopicSum models
- **`topic_models/`**: Contains BERTopic models
- **`notebooks/`**: Contains Jupyter Notebooks for various tasks (training, testing, etc.) 
- **`src/`**: Source code for the BERTopicSum implementation
- **`results/`**: Results from testing the models (generated summaries and ROUGE score)
- **`README.md`**: Project documentation
  
## 🔧 Instructions
1. Follow the Experiment Instructions
Detailed instructions for training and testing the model can be found in the Experiment Instructions.pdf included in the repository.

2. Refer to the Original BERTSum Repository
Additional implementation details can be found in the original [**BERTSum**](https://github.com/nlpyang/BertSum) GitHub repository.
