# Latent Dirichlet Allocation for Text Data

In this assignment you will

* apply standard preprocessing techniques on Wikipedia text data
* use Turi Create to fit a Latent Dirichlet allocation (LDA) model
* explore and interpret the results, including topic keywords and topic assignments for documents

Recall that a major feature distinguishing the LDA model from our previously explored methods is the notion of *mixed membership*. Throughout the course so far, our models have assumed that each data point belongs to a single cluster. k-means determines membership simply by shortest distance to the cluster center, and Gaussian mixture models suppose that each data point is drawn from one of their component mixture distributions. In many cases, though, it is more realistic to think of data as genuinely belonging to more than one cluster or category - for example, if we have a model for text data that includes both "Politics" and "World News" categories, then an article about a recent meeting of the United Nations should have membership in both categories rather than being forced into just one.

With this in mind, we will use Turi Create tools to fit an LDA model to a corpus of Wikipedia articles and examine the results to analyze the impact of a mixed membership approach. In particular, we want to identify the topics discovered by the model in terms of their most important words, and we want to use the model to predict the topic membership distribution for a given document. 

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Text Data Preprocessing
We'll start by importing our familiar Wikipedia dataset.


```python
from __future__ import print_function # to conform python 2.x print to python 3.x
import turicreate
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
# import wiki data
wiki = turicreate.SFrame('people_wiki.sframe/')
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



In the original data, each Wikipedia article is represented by a URI, a name, and a string containing the entire text of the article. Recall from the video lectures that LDA requires documents to be represented as a _bag of words_, which ignores word ordering in the document but retains information on how many times each word appears. As we have seen in our previous encounters with text data, words such as 'the', 'a', or 'and' are by far the most frequent, but they appear so commonly in the English language that they tell us almost nothing about how similar or dissimilar two documents might be. 

Therefore, before we train our LDA model, we will preprocess the Wikipedia data in two steps: first, we will create a bag of words representation for each article, and then we will remove the common words that don't help us to distinguish between documents. For both of these tasks we can use pre-implemented tools from Turi Create:


```python
wiki_docs = turicreate.text_analytics.count_words(wiki['text'])
wiki_docs = wiki_docs.dict_trim_by_keys(turicreate.text_analytics.stop_words(), exclude=True)
```

## Model fitting and interpretation
In the video lectures we saw that Gibbs sampling can be used to perform inference in the LDA model. In this assignment we will use a Turi Create method to learn the topic model for our Wikipedia data, and our main emphasis will be on interpreting the results. We'll begin by creating the topic model using create() from Turi Create's topic_model module.

Note: This may take several minutes to run.


```python
topic_model = turicreate.topic_model.create(wiki_docs, num_topics=10, num_iterations=200)
```


<pre>Learning a topic model</pre>



<pre>       Number of documents     59071</pre>



<pre>           Vocabulary size    547462</pre>



<pre>   Running collapsed Gibbs sampling</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>



<pre>| Iteration | Elapsed Time  | Tokens/Second  | Est. Perplexity |</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>



<pre>| 10        | 5.88s         | 1.56004e+07    | 0               |</pre>



<pre>| 20        | 10.76s        | 1.60757e+07    | 0               |</pre>



<pre>| 30        | 15.73s        | 1.61679e+07    | 0               |</pre>



<pre>| 40        | 20.52s        | 1.51563e+07    | 0               |</pre>



<pre>| 50        | 25.42s        | 1.62581e+07    | 0               |</pre>



<pre>| 60        | 30.67s        | 1.51213e+07    | 0               |</pre>



<pre>| 70        | 35.78s        | 1.70291e+07    | 0               |</pre>



<pre>| 80        | 40.83s        | 1.622e+07      | 0               |</pre>



<pre>| 90        | 45.78s        | 1.40307e+07    | 0               |</pre>



<pre>| 100       | 50.58s        | 1.72727e+07    | 0               |</pre>



<pre>| 110       | 55.22s        | 1.73802e+07    | 0               |</pre>



<pre>| 120       | 59.95s        | 1.69682e+07    | 0               |</pre>



<pre>| 130       | 1m 4s         | 1.74539e+07    | 0               |</pre>



<pre>| 140       | 1m 9s         | 1.56508e+07    | 0               |</pre>



<pre>| 150       | 1m 14s        | 1.56717e+07    | 0               |</pre>



<pre>| 160       | 1m 19s        | 1.68635e+07    | 0               |</pre>



<pre>| 170       | 1m 24s        | 1.7555e+07     | 0               |</pre>



<pre>| 180       | 1m 28s        | 1.74787e+07    | 0               |</pre>



<pre>| 190       | 1m 33s        | 1.66011e+07    | 0               |</pre>



<pre>| 200       | 1m 38s        | 1.71312e+07    | 0               |</pre>



<pre>+-----------+---------------+----------------+-----------------+</pre>


Turi provides a useful summary of the model we have fitted, including the hyperparameter settings for alpha, gamma (note that Turi Create calls this parameter beta), and K (the number of topics); the structure of the output data; and some useful methods for understanding the results.


```python
topic_model
```




    Class                          : TopicModel
    
    Schema
    ------
    Vocabulary Size                : 547462
    
    Settings
    --------
    Number of Topics               : 10
    alpha                          : 5.0
    beta                           : 0.1
    Iterations                     : 200
    Training time                  : 99.3369
    Verbose                        : True
    
    Accessible fields             : 
    m.topics                      : An SFrame containing the topics.
    m.vocabulary                  : An SArray containing the words in the vocabulary.
    Useful methods                : 
    m.get_topics()                : Get the most probable words per topic.
    m.predict(new_docs)           : Make predictions for new documents.



It is certainly useful to have pre-implemented methods available for LDA, but as with our previous methods for clustering and retrieval, implementing and fitting the model gets us only halfway towards our objective. We now need to analyze the fitted model to understand what it has done with our data and whether it will be useful as a document classification system. This can be a challenging task in itself, particularly when the model that we use is complex. We will begin by outlining a sequence of objectives that will help us understand our model in detail. In particular, we will

* get the top words in each topic and use these to identify topic themes
* predict topic distributions for some example documents
* compare the quality of LDA "nearest neighbors" to the NN output from the first assignment
* understand the role of model hyperparameters alpha and gamma

## Load a fitted topic model
The method used to fit the LDA model is a _randomized algorithm_, which means that it involves steps that are random; in this case, the randomness comes from Gibbs sampling, as discussed in the LDA video lectures. Because of these random steps, the algorithm will be expected to yield slighty different output for different runs on the same data - note that this is different from previously seen algorithms such as k-means or EM, which will always produce the same results given the same input and initialization.

It is important to understand that variation in the results is a fundamental feature of randomized methods. However, in the context of this assignment this variation makes it difficult to evaluate the correctness of your analysis, so we will load and analyze a pre-trained model. 

We recommend that you spend some time exploring your own fitted topic model and compare our analysis of the pre-trained model to the same analysis applied to the model you trained above.


```python
topic_model = turicreate.load_model('topic_models/lda_assignment_topic_model')
```

# Identifying topic themes by top words

We'll start by trying to identify the topics learned by our model with some major themes. As a preliminary check on the results of applying this method, it is reasonable to hope that the model has been able to learn topics that correspond to recognizable categories. In order to do this, we must first recall what exactly a 'topic' is in the context of LDA. 

In the video lectures on LDA we learned that a topic is a probability distribution over words in the vocabulary; that is, each topic assigns a particular probability to every one of the unique words that appears in our data. Different topics will assign different probabilities to the same word: for instance, a topic that ends up describing science and technology articles might place more probability on the word 'university' than a topic that describes sports or politics. Looking at the highest probability words in each topic will thus give us a sense of its major themes. Ideally we would find that each topic is identifiable with some clear theme _and_ that all the topics are relatively distinct.

We can use the Turi Create function get_topics() to view the top words (along with their associated probabilities) from each topic.

__Quiz Question:__ Identify the top 3 most probable words for the first topic. president, business, board.

**Quiz Question:** What is the sum of the probabilities assigned to the top 50 words in the 3rd topic? 0.04783977160632165


```python
topic_model.get_topics()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">topic</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">score</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">president</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008339770494721031</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">business</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008230612437460937</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">board</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.007476947242117326</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">director</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.007380128791330111</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">international</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.007123844656893367</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">university</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.04865493156332634</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">research</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.012161362488061602</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">professor</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.011922106881371426</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">school</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.009947805060239167</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">college</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.009489674880021313</td>
    </tr>
</table>
[50 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
topics = topic_model.get_topics()
topics = topics[topics['topic']==2][:50]
score_sum = 0
for i in topics:
    score_sum += i['score']
score_sum
```




    0.04783977160632165



Let's look at the top 10 words for each topic to see if we can identify any themes:


```python
[x['words'] for x in topic_model.get_topics(output_type='topic_words', num_words=10)]
```




    [['president',
      'business',
      'board',
      'director',
      'international',
      'company',
      'development',
      'member',
      'executive',
      'served'],
     ['university',
      'research',
      'professor',
      'school',
      'college',
      'science',
      'institute',
      'american',
      'received',
      'studies'],
     ['de',
      'born',
      'international',
      'music',
      'la',
      'orchestra',
      'opera',
      'french',
      'studied',
      'national'],
     ['work',
      'book',
      'art',
      'published',
      'york',
      'books',
      'magazine',
      'american',
      'arts',
      'works'],
     ['won',
      'world',
      'team',
      'club',
      'played',
      'season',
      'born',
      'cup',
      'football',
      'championship'],
     ['years',
      'born',
      'time',
      'family',
      'father',
      'people',
      'children',
      'called',
      'found',
      'son'],
     ['member',
      'served',
      'party',
      'state',
      'election',
      'law',
      'minister',
      'elected',
      'united',
      'general'],
     ['league',
      'played',
      'season',
      'career',
      'games',
      'coach',
      'major',
      'team',
      'high',
      'game'],
     ['film',
      'award',
      'television',
      'series',
      'show',
      'radio',
      'appeared',
      'role',
      'tv',
      'theatre'],
     ['music',
      'album',
      'released',
      'band',
      'records',
      'song',
      'songs',
      'recorded',
      'jazz',
      'singer']]



We propose the following themes for each topic:

- topic 0: Business
- topic 1: Science and research
- topic 2: International music
- topic 3: Art and publishing
- topic 4: Team sports
- topic 5: Family and society
- topic 6: Politics
- topic 7: International athletics
- topic 8: TV and film
- topic 9: General music

We'll save these themes for later:


```python
themes = ['business',
          'science and research',
          'international music',
          'art and publishing',
          'team sports',
          'family and society',
          'politics',
          'international athletics',
          'TV and film',
          'general music']
```

### Measuring the importance of top words

We can learn more about topics by exploring how they place probability mass (which we can think of as a weight) on each of their top words.

We'll do this with two visualizations of the weights for the top words in each topic:
 - the weights of the top 100 words, sorted by the size
 - the total weight of the top 10 words


Here's a plot for the top 100 words by weight in each topic:


```python
for i in range(10):
    plt.plot(range(100), topic_model.get_topics(topic_ids=[i], num_words=100)['score'])
plt.xlabel('Word rank')
plt.ylabel('Probability')
plt.title('Probabilities of Top 100 Words in each Topic')
```




    Text(0.5, 1.0, 'Probabilities of Top 100 Words in each Topic')




![png](output_24_1.png)


In the above plot, each line corresponds to one of our ten topics. Notice how for each topic, the weights drop off sharply as we move down the ranked list of most important words. This shows that the top 10-20 words in each topic are assigned a much greater weight than the remaining words - and remember from the summary of our topic model that our vocabulary has 547462 words in total!


Next we plot the total weight assigned by each topic to its top 10 words: 


```python
top_probs = [sum(topic_model.get_topics(topic_ids=[i], num_words=10)['score']) for i in range(10)]

ind = np.arange(10)
width = 0.5

fig, ax = plt.subplots()

ax.bar(ind-(width/2),top_probs,width)
ax.set_xticks(ind)

plt.xlabel('Topic')
plt.ylabel('Probability')
plt.title('Total Probability of Top 10 Words in each Topic')
plt.xlim(-0.5,9.5)
plt.ylim(0,0.15)
plt.show()
```


![png](output_26_0.png)


Here we see that, for our topic model, the top 10 words only account for a small fraction (in this case, between 5% and 13%) of their topic's total probability mass. So while we can use the top words to identify broad themes for each topic, we should keep in mind that in reality these topics are more complex than a simple 10-word summary.

Finally, we observe that some 'junk' words appear highly rated in some topics despite our efforts to remove unhelpful words before fitting the model; for example, the word 'born' appears as a top 10 word in three different topics, but it doesn't help us describe these topics at all.

# Topic distributions for some example documents

As we noted in the introduction to this assignment, LDA allows for mixed membership, which means that each document can partially belong to several different topics. For each document, topic membership is expressed as a vector of weights that sum to one; the magnitude of each weight indicates the degree to which the document represents that particular topic.

We'll explore this in our fitted model by looking at the topic distributions for a few example Wikipedia articles from our data set. We should find that these articles have the highest weights on the topics whose themes are most relevant to the subject of the article - for example, we'd expect an article on a politician to place relatively high weight on topics related to government, while an article about an athlete should place higher weight on topics related to sports or competition.

Topic distributions for documents can be obtained using Turi Create's predict() function. Turi Create uses a collapsed Gibbs sampler similar to the one described in the video lectures, where only the word assignments variables are sampled.  To get a document-specific topic proportion vector post-facto, predict() draws this vector from the conditional distribution given the sampled word assignments in the document.  Notice that, since these are draws from a _distribution_ over topics that the model has learned, we will get slightly different predictions each time we call this function on a document - we can see this below, where we predict the topic distribution for the article on Barack Obama:


```python
obama = turicreate.SArray([wiki_docs[int(np.where(wiki['name']=='Barack Obama')[0])]])
pred1 = topic_model.predict(obama, output_type='probability')
pred2 = topic_model.predict(obama, output_type='probability')
print(turicreate.SFrame({'topics':themes, 'predictions (first draw)':pred1[0], 'predictions (second draw)':pred2[0]}))
```

    +--------------------------+---------------------------+-------------------------+
    | predictions (first draw) | predictions (second draw) |          topics         |
    +--------------------------+---------------------------+-------------------------+
    |   0.08064516129032258    |    0.08333333333333333    |         business        |
    |   0.06720430107526881    |    0.04032258064516129    |   science and research  |
    |   0.021505376344086023   |    0.03763440860215054    |   international music   |
    |   0.026881720430107527   |    0.02956989247311828    |    art and publishing   |
    |   0.04838709677419355    |    0.021505376344086023   |       team sports       |
    |   0.051075268817204304   |    0.06989247311827956    |    family and society   |
    |    0.5698924731182796    |     0.5967741935483871    |         politics        |
    |   0.09408602150537634    |    0.08064516129032258    | international athletics |
    |   0.013440860215053764   |    0.016129032258064516   |       TV and film       |
    |   0.026881720430107527   |    0.024193548387096774   |      general music      |
    +--------------------------+---------------------------+-------------------------+
    [10 rows x 3 columns]
    


To get a more robust estimate of the topics for each document, we can average a large number of predictions for the same document:


```python
def average_predictions(model, test_document, num_trials=100):
    avg_preds = np.zeros((model.num_topics))
    for i in range(num_trials):
        avg_preds += model.predict(test_document, output_type='probability')[0]
    avg_preds = avg_preds/num_trials
    result = turicreate.SFrame({'topics':themes, 'average predictions':avg_preds})
    result = result.sort('average predictions', ascending=False)
    return result
```


```python
print(average_predictions(topic_model, obama, 100))
```

    +----------------------+-------------------------+
    | average predictions  |          topics         |
    +----------------------+-------------------------+
    |  0.5669892473118278  |         politics        |
    | 0.09983870967741934  |         business        |
    | 0.07612903225806453  |    family and society   |
    |  0.0631720430107527  |   science and research  |
    | 0.06072580645161293  | international athletics |
    |  0.0363440860215054  |       team sports       |
    | 0.029892473118279583 |   international music   |
    | 0.026344086021505387 |    art and publishing   |
    | 0.021693548387096764 |      general music      |
    | 0.01887096774193547  |       TV and film       |
    +----------------------+-------------------------+
    [10 rows x 2 columns]
    


__Quiz Question:__ What is the topic most closely associated with the article about former US President George W. Bush? Use the average results from 100 topic predictions. politics


```python
bush = turicreate.SArray([wiki_docs[int(np.where(wiki['name']=='George W. Bush')[0])]])
print(average_predictions(topic_model, bush, 100))
```

    +----------------------+-------------------------+
    | average predictions  |          topics         |
    +----------------------+-------------------------+
    | 0.46535087719298196  |         politics        |
    | 0.13564327485380112  |         business        |
    |  0.0897953216374269  |    family and society   |
    | 0.06546783625730995  | international athletics |
    | 0.06380116959064326  |   science and research  |
    | 0.052485380116959066 |    art and publishing   |
    | 0.043128654970760225 |      general music      |
    | 0.033245614035087724 |       TV and film       |
    | 0.031052631578947353 |       team sports       |
    | 0.020029239766081862 |   international music   |
    +----------------------+-------------------------+
    [10 rows x 2 columns]
    


__Quiz Question:__ What are the top 3 topics corresponding to the article about English football (soccer) player Steven Gerrard? Use the average results from 100 topic predictions. team sports, international athletics, general music


```python
gerrard = turicreate.SArray([wiki_docs[int(np.where(wiki['name']=='Steven Gerrard')[0])]])
print(average_predictions(topic_model, gerrard, 100))
```

    +----------------------+-------------------------+
    | average predictions  |          topics         |
    +----------------------+-------------------------+
    |  0.6844800000000003  |       team sports       |
    | 0.06803999999999995  | international athletics |
    | 0.037560000000000024 |      general music      |
    | 0.03592000000000002  |       TV and film       |
    | 0.03528000000000003  |   international music   |
    | 0.032080000000000025 |         business        |
    | 0.03152000000000002  |         politics        |
    | 0.02644000000000002  |    family and society   |
    | 0.024520000000000017 |    art and publishing   |
    | 0.02416000000000002  |   science and research  |
    +----------------------+-------------------------+
    [10 rows x 2 columns]
    


# Comparing LDA to nearest neighbors for document retrieval

So far we have found that our topic model has learned some coherent topics, we have explored these topics as probability distributions over a vocabulary, and we have seen how individual documents in our Wikipedia data set are assigned to these topics in a way that corresponds with our expectations. 

In this section, we will use the predicted topic distribution as a representation of each document, similar to how we have previously represented documents by word count or TF-IDF. This gives us a way of computing distances between documents, so that we can run a nearest neighbors search for a given document based on its membership in the topics that we learned from LDA. We can contrast the results with those obtained by running nearest neighbors under the usual TF-IDF representation, an approach that we explored in a previous assignment. 

We'll start by creating the LDA topic distribution representation for each document:


```python
wiki['lda'] = topic_model.predict(wiki_docs, output_type='probability')
```

Next we add the TF-IDF document representations:


```python
wiki['word_count'] = turicreate.text_analytics.count_words(wiki['text'])
wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['word_count'])
```

For each of our two different document representations, we can use Turi Create to compute a brute-force nearest neighbors model:


```python
model_tf_idf = turicreate.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                   method='brute_force', distance='cosine')
model_lda_rep = turicreate.nearest_neighbors.create(wiki, label='name', features=['lda'],
                                                    method='brute_force', distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Validating distance components.</pre>



<pre>Initializing model data.</pre>



<pre>Initializing distances.</pre>



<pre>Done.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Validating distance components.</pre>



<pre>Initializing model data.</pre>



<pre>Initializing distances.</pre>



<pre>Done.</pre>


Let's compare these nearest neighbor models by finding the nearest neighbors under each representation on an example document. For this example we'll use Paul Krugman, an American economist:


```python
model_tf_idf.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 9.919ms      |</pre>



<pre>| Done         |         | 100         | 130.628ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Elise Brezis</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.7444980172622551</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Maitreesh Ghatak</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8156498483099809</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Kai A. Konrad</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8237005644064379</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">David Colander</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8346259277590851</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Richard Blundell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8379342678735122</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Gordon Rausser</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8394153470596016</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Edward J. Nell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8421785000147988</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robin Boadway</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8423742605956992</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tim Besley</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8430881092528927</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>




```python
model_lda_rep.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 1.296ms      |</pre>



<pre>| Done         |         | 100         | 22.844ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.1102230246251565e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Isidro A. T. Savillo</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.006938472319229505</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Kent Larson (architect)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0072802002671583654</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Peter Dauvergne</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.00789359893069963</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Melissa Jawaharlal</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008164064218360023</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">K. K. N. Kurup</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008165103155030495</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stanley Hauerwas</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008901912715545857</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Xiao Qiang</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.008936769927474608</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Peter Temin</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.009022122297175916</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Paul Krugman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John B. Cobb</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.009037237587164348</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



Notice that that there is no overlap between the two sets of top 10 nearest neighbors. This doesn't necessarily mean that one representation is better or worse than the other, but rather that they are picking out different features of the documents. 

With TF-IDF, documents are distinguished by the frequency of uncommon words. Since similarity is defined based on the specific words used in the document, documents that are "close" under TF-IDF tend to be similar in terms of specific details. This is what we see in the example: the top 10 nearest neighbors are all economists from the US, UK, or Canada. 

Our LDA representation, on the other hand, defines similarity between documents in terms of their topic distributions. This means that documents can be "close" if they share similar themes, even though they may not share many of the same keywords. For the article on Paul Krugman, we expect the most important topics to be 'American college and politics' and 'science and research'. As a result, we see that the top 10 nearest neighbors are academics from a wide variety of fields, including literature, anthropology, and religious studies.

# Understanding the role of LDA model hyperparameters

Finally, we'll take a look at the effect of the LDA model hyperparameters alpha and gamma on the characteristics of our fitted model. Recall that alpha is a parameter of the prior distribution over topic weights in each document, while gamma is a parameter of the prior distribution over word weights in each topic. 

In the video lectures, we saw that alpha and gamma can be thought of as smoothing parameters when we compute how much each document "likes" a topic (in the case of alpha) or how much each topic "likes" a word (in the case of gamma). In both cases, these parameters serve to reduce the differences across topics or words in terms of these calculated preferences; alpha makes the document preferences "smoother" over topics, and gamma makes the topic preferences "smoother" over words.

Our goal in this section will be to understand how changing these parameter values affects the characteristics of the resulting topic model.

__Quiz Question:__ What was the value of alpha used to fit our original topic model? 5.0


```python
topic_model
```




    Class                          : TopicModel
    
    Schema
    ------
    Vocabulary Size                : 547462
    
    Settings
    --------
    Number of Topics               : 10
    alpha                          : 5.0
    beta                           : 0.1
    Iterations                     : 200
    Training time                  : 125.8235
    Verbose                        : True
    
    Accessible fields             : 
    m.topics                      : An SFrame containing the topics.
    m.vocabulary                  : An SArray containing the words in the vocabulary.
    Useful methods                : 
    m.get_topics()                : Get the most probable words per topic.
    m.predict(new_docs)           : Make predictions for new documents.



__Quiz Question:__ What was the value of gamma used to fit our original topic model? Remember that Turi Create uses "beta" instead of "gamma" to refer to the hyperparameter that influences topic distributions over words. 0.1


```python

```

We'll start by loading some topic models that have been trained using different settings of alpha and gamma. Specifically, we will start by comparing the following two models to our original topic model:
 - tpm_low_alpha, a model trained with alpha = 1 and default gamma
 - tpm_high_alpha, a model trained with alpha = 50 and default gamma


```python
tpm_low_alpha = turicreate.load_model('topic_models/lda_low_alpha')
tpm_high_alpha = turicreate.load_model('topic_models/lda_high_alpha')
```

### Changing the hyperparameter alpha

Since alpha is responsible for smoothing document preferences over topics, the impact of changing its value should be visible when we plot the distribution of topic weights for the same document under models fit with different alpha values. In the code below, we plot the (sorted) topic weights for the Wikipedia article on Barack Obama under models fit with high, original, and low settings of alpha.


```python
a = np.sort(tpm_low_alpha.predict(obama,output_type='probability')[0])[::-1]
b = np.sort(topic_model.predict(obama,output_type='probability')[0])[::-1]
c = np.sort(tpm_high_alpha.predict(obama,output_type='probability')[0])[::-1]
ind = np.arange(len(a))
width = 0.3

def param_bar_plot(a,b,c,ind,width,ylim,param,xlab,ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    b1 = ax.bar(ind, a, width, color='lightskyblue')
    b2 = ax.bar(ind+width, b, width, color='lightcoral')
    b3 = ax.bar(ind+(2*width), c, width, color='gold')

    ax.set_xticks(ind+width)
    ax.set_xticklabels(range(10))
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(0,ylim)
    ax.legend(handles = [b1,b2,b3],labels=['low '+param,'original model','high '+param])

    plt.tight_layout()
    
param_bar_plot(a,b,c,ind,width,ylim=1.0,param='alpha',
               xlab='Topics (sorted by weight of top 100 words)',ylab='Topic Probability for Obama Article')
```


![png](output_55_0.png)


Here we can clearly see the smoothing enforced by the alpha parameter - notice that when alpha is low most of the weight in the topic distribution for this article goes to a single topic, but when alpha is high the weight is much more evenly distributed across the topics.

__Quiz Question:__ How many topics are assigned a weight greater than 0.3 or less than 0.05 for the article on Paul Krugman in the **low alpha** model?  Use the average results from 100 topic predictions. 3


```python
paul_low = turicreate.SArray([wiki_docs[int(np.where(wiki['name']=='Paul Krugman')[0])]])
print(average_predictions(tpm_low_alpha, paul_low, 100))
```

    +----------------------+-------------------------+
    | average predictions  |          topics         |
    +----------------------+-------------------------+
    |  0.4658641975308644  |    art and publishing   |
    | 0.20956790123456784  | international athletics |
    | 0.18006172839506177  |       team sports       |
    | 0.06987654320987652  |   international music   |
    | 0.015987654320987678 |      general music      |
    | 0.015246913580246928 |   science and research  |
    | 0.012962962962962976 |    family and society   |
    | 0.011296296296296313 |         business        |
    | 0.009876543209876557 |       TV and film       |
    | 0.00925925925925927  |         politics        |
    +----------------------+-------------------------+
    [10 rows x 2 columns]
    


__Quiz Question:__ How many topics are assigned a weight greater than 0.3 or less than 0.05 for the article on Paul Krugman in the **high alpha** model? Use the average results from 100 topic predictions. 0


```python
paul_high = turicreate.SArray([wiki_docs[int(np.where(wiki['name']=='Paul Krugman')[0])]])
print(average_predictions(tpm_high_alpha, paul_high, 100))
```

    +---------------------+-------------------------+
    | average predictions |          topics         |
    +---------------------+-------------------------+
    |  0.1409509202453987 |         politics        |
    | 0.13128834355828223 |       team sports       |
    | 0.10526073619631904 |    family and society   |
    | 0.10328220858895712 |   science and research  |
    | 0.09484662576687108 |         business        |
    | 0.09018404907975462 | international athletics |
    | 0.08498466257668717 |    art and publishing   |
    | 0.08424846625766869 |      general music      |
    | 0.08309815950920241 |       TV and film       |
    |  0.0818558282208589 |   international music   |
    +---------------------+-------------------------+
    [10 rows x 2 columns]
    


### Changing the hyperparameter gamma

Just as we were able to see the effect of alpha by plotting topic weights for a document, we expect to be able to visualize the impact of changing gamma by plotting word weights for each topic. In this case, however, there are far too many words in our vocabulary to do this effectively. Instead, we'll plot the total weight of the top 100 words and bottom 1000 words for each topic. Below, we plot the (sorted) total weights of the top 100 words and bottom 1000 from each topic in the high, original, and low gamma models.

Now we will consider the following two models:
 - tpm_low_gamma, a model trained with gamma = 0.02 and default alpha
 - tpm_high_gamma, a model trained with gamma = 0.5 and default alpha


```python
del tpm_low_alpha
del tpm_high_alpha
tpm_low_gamma = turicreate.load_model('topic_models/lda_low_gamma')
tpm_high_gamma = turicreate.load_model('topic_models/lda_high_gamma')
```


```python
a_top = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
b_top = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
c_top = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]

a_bot = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
b_bot = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
c_bot = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]

ind = np.arange(len(a))
width = 0.3
    
param_bar_plot(a_top, b_top, c_top, ind, width, ylim=0.6, param='gamma',
               xlab='Topics (sorted by weight of top 100 words)', 
               ylab='Total Probability of Top 100 Words')

param_bar_plot(a_bot, b_bot, c_bot, ind, width, ylim=0.0002, param='gamma',
               xlab='Topics (sorted by weight of bottom 1000 words)',
               ylab='Total Probability of Bottom 1000 Words')
```


![png](output_63_0.png)



![png](output_63_1.png)


From these two plots we can see that the low gamma model results in higher weight placed on the top words and lower weight placed on the bottom words for each topic, while the high gamma model places relatively less weight on the top words and more weight on the bottom words. Thus increasing gamma results in topics that have a smoother distribution of weight across all the words in the vocabulary.

__Quiz Question:__ For each topic of the **low gamma model**, compute the number of words required to make a list with total probability 0.5. What is the average number of words required across all topics? (HINT: use the get\_topics() function from Turi Create with the cdf\_cutoff argument). 266.5


```python
len(tpm_low_gamma.get_topics(cdf_cutoff=.5,num_words=10000))/tpm_low_gamma.num_topics
```




    266.5



__Quiz Question:__ For each topic of the **high gamma model**, compute the number of words required to make a list with total probability 0.5. What is the average number of words required across all topics? (HINT: use the get\_topics() function from Turi Create with the cdf\_cutoff argument). 849.1


```python
len(tpm_high_gamma.get_topics(cdf_cutoff=.5,num_words=10000))/tpm_high_gamma.num_topics
```




    849.1



We have now seen how the hyperparameters alpha and gamma influence the characteristics of our LDA topic model, but we haven't said anything about what settings of alpha or gamma are best. We know that these parameters are responsible for controlling the smoothness of the topic distributions for documents and word distributions for topics, but there's no simple conversion between smoothness of these distributions and quality of the topic model. In reality, there is no universally "best" choice for these parameters. Instead, finding a good topic model requires that we be able to both explore the output (as we did by looking at the topics and checking some topic predictions for documents) and understand the impact of hyperparameter settings (as we have in this section).
