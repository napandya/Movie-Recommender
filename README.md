# Movie Recommender
The Movie Recommender project is part of the submission for Data Mining course.
It is developed in three major phases:
1) Developing the search 
2) Developing the classifier
3) Developing the recommender

# The search feature
**Dataset**:(https://www.kaggle.com/rounakbanik/the-movies-dataset/home)
- To develop the search feature for this web application,we used 'The Movies Dataset'.These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages,production companies, countries, TMDB vote counts and vote averages.This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

**Inverted Index**:

  - Two principal components of an inverted index are the dictionary and the postings lists. For each term in the text collection, there     is a postings list that contains information about the term’s occurrences in the collection. 
  - The information found in these postings lists is used by the system to process search queries.Each posting list corresponds to a         word, which stores all the IDs of documents where this word appears in ascending order.The dictionary serves as a lookup data           structure on top of the postings lists. 
  - For every query term in an incoming search query, the search engine first needs to locate the term’s postings list before it can         start processing the query. It is the job of the dictionary to provide this mapping from terms to the location of their postings         lists in the index. 
  - The life cycle of a static inverted index, built for a neverchanging text collection, consists of two distinct phases.
  - First is index construction. In this phase the text collection is processed sequentially, one token at time, and a posting list is       built for each term in the collection in an incremental fashion. 
  - The second phase is query processing. Here the information stored in the index that was built in phase one is used to process search     queries. 
  - Before building inverted indexes, we must first acquire the document collection over which these indexes are to be built. In the         case of text documents location of documents in the disk must be known.
    Reference:(https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7034561)
```
def create_inverted_index(x_data, x_cols):
    for row in x_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        for col in x_cols.keys():
            if col != "id":
                col_values = getattr(row, col)
                parameters = x_cols[col]
                if parameters is None:
                     data.append(col_values if isinstance(col_values, str) else "")
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                    for col_value in col_values:
                        for param in parameters:
                            data.append(col_value[param])
        insert(index, pre_processing(' '.join(data)))
        #break

 ```
 **TF IDF**:
- The concepts of Term Frequency (TF) and Inverse Document Frequency (IDF) are used in information retrieval systems and also content-based filtering mechanisms (such as a content based recommender). They are used to determine the relative importance of a
document / article / news item / movie etc.
- TF is simply the frequency of a word in a document. IDF is the inverse of the document frequency among the whole corpus of documents. TF-IDF is used mainly because of tworeasons: Suppose we search for “the results of latest European Socccer games” on Google. It is certain that “the” will occur more frequently than “soccer games” but therelative importance of soccer games is higher than the search query point of view. 
- In such cases, TF-IDF weighting negates the effect of high frequency words in determining
the importance of an item (document).
 ```
     for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize

def get_relevant_docs(query_list):
    relevant_docs = set()
    for query in query_list:
        if query in inverted_index:
            keys = inverted_index[query].keys()
            for key in keys:
                relevant_docs.add(key)
    if "df" in relevant_docs:
        relevant_docs.remove("df")
    return relevant_docs
```
**Vector Space Model And Cosine Similarity:**
- After calculating TF-IDF scores, how do we determine which items are closer to each other, rather closer to the user profile? This is accomplished using the Vector Space Model which computes the proximity based on the angle between the vectors. 
- In this model, each item is stored as a vector of its attributes (which are also vectors) in an ndimensional space and the angles between the vectors are calculated to determine the similarity between the vectors. 
- Next, the user profile vectors are also created based on his actions on previous attributes of items and the similarity between an item and a user is also determined in a similar way.
- Sentence 2 is more likely to be using Term 2 than using Term 1. Vice-versa for Sentence 1. The method of calculating this relative measure is calculated by taking the cosine of the angle between the sentences and the terms. The ultimate reason behind using
cosine is that the value of cosine will increase with decreasing value of the angle between which signifies more similarity. The vectors are length normalized after which they become vectors of length 1 and then the cosine calculation is simply the sum-product of vectors.
```
def cosine_similarity(relevant_docs, query_vector):
    score_map = {}
    for doc in relevant_docs:
        score = 0
        for token in query_vector:
            score += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
        score_map[doc] = score
    sorted_score_map = sorted(score_map.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map[:50]
```
# Classifier
**Naive Bayes Classifier:**
- It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. 
- Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
- The below code snippet implements the logic for multinomial Naive Bayes Classifier.
```
def get_results(query):
    global prior_probability, post_probability
    initialize()
    if os.path.isfile("classifierPickle.pkl"):
        prior_probability = pickle.load(open('classifierPicklePrior.pkl', 'rb'))
        post_probability = pickle.load(open('classifierPicklePost.pkl', 'rb'))
    else:
        (prior_probability, post_probability) = build_and_save()
    return eval_result(query)

def eval_result(query):
    processed_query = pre_processing(query)
    genre_score = {}
    for genre in prior_probability.keys():
        score = prior_probability[genre]
        # print("For genre: ", genre, ", prior score: ", score)
        for token in processed_query:
            if (genre, token) in post_probability.keys():
                score = score * post_probability[(genre, token)]
                # print("token: ", token, ", score: ", score)
        genre_score[genre] = score
    sorted_score_map = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map
 ```
# Recommender
