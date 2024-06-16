import mlx.core as mx
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
toy data for training word vectors
"""
DATA_SET = """
The history of artificial intelligence (AI) is a rich and complex narrative that spans several decades, marked by significant milestones, breakthroughs, and periods of both optimism and disillusionment. Here is a detailed overview of the key events and developments in the history of AI:

## Early Concepts and Foundations

### Ancient and Early Modern Periods
- **Myths and Legends**: The idea of creating machines that mimic human intelligence dates back to ancient myths and legends about automatons and thinking machines[2][3].
- **Philosophical Foundations**: Philosophers like René Descartes in the 17th century contemplated the possibility of machines thinking and making decisions[6].

### 1940s-1950s: Birth of AI
- **1943**: Warren McCulloch and Walter Pitts presented a model of artificial neurons, considered the first artificial intelligence, even though the term did not yet exist[2][6].
- **1950**: Alan Turing published "Computing Machinery and Intelligence," introducing the Turing Test to determine if a machine can exhibit intelligent behavior indistinguishable from a human[2][3][6].
- **1956**: John McCarthy coined the term "artificial intelligence" at the Dartmouth Conference, marking the official birth of AI as a field of study[2][3][6].

## The Golden Age and Early Challenges

### 1950s-1970s: Early Enthusiasm and Setbacks
- **1956-1974**: Known as the "Golden Age" of AI, this period saw significant advancements, including the development of the first AI programming language, LISP, and early AI systems like the Logic Theorist and General Problem Solver[2][4][5].
- **1966**: Joseph Weizenbaum developed ELIZA, the first natural language processing computer program, simulating human conversation[2][6].
- **1974-1980**: The first "AI Winter" occurred due to high expectations not being met, leading to a substantial decrease in research funding[3][4][5].

## Revival and Modern Developments

### 1980s-1990s: Expert Systems and Renewed Interest
- **1980-1987**: AI experienced a renaissance with the development of expert systems like MYCIN and DENDRAL, which utilized knowledge bases and rules to solve complex problems[4][5].
- **1997**: IBM's Deep Blue defeated world chess champion Garry Kasparov, showcasing the potential of AI in complex strategic games[2][5][6].

### 2000s-Present: Machine Learning and Deep Learning
- **2011**: IBM's Watson won the game show Jeopardy!, demonstrating the capabilities of AI in natural language processing and question-answering[2][5][6].
- **2012**: The ImageNet competition, won by a deep learning neural network, highlighted the power of convolutional neural networks (CNNs) in image recognition tasks[5][6].
- **2016**: DeepMind's AlphaGo defeated the world champion Go player, Lee Sedol, marking a significant achievement in AI's ability to handle complex, intuitive problems[2][5][6].
- **2022**: OpenAI launched ChatGPT, an AI chatbot built on the GPT-3.5 large language model, advancing the capabilities of natural language processing and generation[14][19].

## Recent Developments and Future Directions

### 2020s: Generative AI and Large Language Models
- **2023**: The rise of large language models (LLMs) like ChatGPT has created significant changes in AI's performance and its potential to drive enterprise value[17].
- **Ongoing**: AI continues to evolve with advancements in deep learning, neural networks, and generative models, impacting various industries and aspects of daily life[18][19].

## Conclusion
The history of AI is a testament to human curiosity, perseverance, and technological advancements. From its inception as a theoretical concept to the modern breakthroughs we witness today, AI has undergone significant milestones and continues to shape the future of technology and society.

Citations:
[1] https://www.coe.int/en/web/artificial-intelligence/history-of-ai
[2] https://www.iberdrola.com/innovation/history-artificial-intelligence
[3] https://en.wikipedia.org/wiki/History_of_artificial_intelligence
[4] https://www.javatpoint.com/history-of-artificial-intelligence
[5] https://theblogsail.com/technology/historical-background-and-key-milestones-in-ai-development/
[6] https://bernardmarr.com/the-most-significant-ai-milestones-so-far/
[7] https://verloop.io/blog/the-timeline-of-artificial-intelligence-from-the-1940s/
[8] https://www.coursera.org/articles/history-of-ai
[9] https://courses.cs.washington.edu/courses/csep590/06au/projects/history-ai.pdf
[10] https://achievements.ai
[11] https://ourworldindata.org/brief-history-of-ai
[12] https://en.wikipedia.org/wiki/Timeline_of_artificial_intelligence
[13] https://redresscompliance.com/the-evolution-of-ai-tracing-its-roots-and-milestones/
[14] https://www.officetimeline.com/blog/artificial-intelligence-ai-and-chatgpt-history-and-timelines
[15] https://piktochart.com/ai-timeline-generator/
[16] https://www.techtarget.com/searchenterpriseai/tip/The-history-of-artificial-intelligence-Complete-AI-timeline
[17] https://utsouthwestern.libguides.com/artificial-intelligence/ai-timeline
[18] https://www.rigb.org/explore-science/explore/blog/10-ai-milestones-last-10-years
[19] https://www.theainavigator.com/ai-timeline
"""

def get_vocab(data: str):
    """
    get the vocabulary and index mappings for the data
    args:
        - data: the text data in string format
    returns:
        - word_to_index: a dictionary mapping words to indices
        - index_to_word: a dictionary mapping indices to words
    """
    # create a set of unique words
    words = set(data.split())
    # create a dictionary mapping words to indices
    word_to_index = {word: i for i, word in enumerate(words)}
    # create a dictionary mapping indices to words
    index_to_word = {i: word for word, i in word_to_index.items()}
    return word_to_index, index_to_word

def batch_iterate(batch_size, X, y):
    """
    iterate over the data in batches
    args:
        - batch_size: the size of the batch
        - X: the input data
        - y: the target data
    yields:
        - a batch of input and target data
    """
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]

def visualize_embedding(embeddings, word_to_index):
    """
    perform the tsne dimension reduction 
    args:
        - embeddings: the embeddings weights 
        - word_to_index: the word to index vocab
    returns:
        None
    """
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(10, 10))
    for word, idx in word_to_index.items():
        plt.scatter(*embeddings_2d[idx], marker='x', color='red')
        plt.text(*embeddings_2d[idx], word, fontsize=9)
    plt.show()