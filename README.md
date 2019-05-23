# Movie-Recommendation
Dataset: [Movie Lens](https://www.kaggle.com/rounakbanik/the-movies-dataset/home))
- To develop the search feature for this web application,we used 'The Movies Dataset'.These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages,production companies, countries, TMDB vote counts and vote averages.This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

## Project Video
[![Watch the video](https://img.youtube.com/vi/JWs6-rqRnGY/0.jpg)](https://www.youtube.com/watch?v=pwQSCtnXKbc&t=38s)

## [Movie Search](https://nandanpandya.netlify.com/post/blog_post/)
Blog: [https://nandanpandya.netlify.com/post/blog_post//](https://nandanpandya.netlify.com/post/blog_post//)

Search feature calculate cosine similarity between vectors space of search query and movies and top 20 movies are returned.

###Pre-Processing:Stemming
```
def pre_processing(data_string):
    for noise in noise_list:
        data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data
```
###Creation Of Inverted Index:
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
```
### Insert the token created in the index
```
def insert(index, tokens):
    for token in tokens:
        if token in inverted_index:
            value = inverted_index[token]
            if index in value.keys():
                value[index] += 1
            else:
                value[index] = 1
                value["df"] += 1
        else:
            inverted_index[token] = {index: 1, "df": 1}
```

###Calculation for TF IDF:

Calculate TF and IDF of each document:
```
def build_doc_vector():
    for token_key in inverted_index:
        token_values = inverted_index[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                log_tf = 1 + math.log10(token_values[doc_key])
                tf_idf = log_tf * idf
                if doc_key not in document_vector:
                    document_vector[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                else:
                    document_vector[doc_key][token_key] = tf_idf
                    document_vector[doc_key]["_sum_"] += math.pow(tf_idf, 2)
	
	for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize
```
###Build the Query Vector

```
def build_query_vector(processed_query):
    query_vector = {}
    sum = 0
    for token in processed_query:
        if token in inverted_index:
            tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/inverted_index[token]["df"])
            query_vector[token] = tf_idf
            sum += math.pow(tf_idf, 2)
    sum = math.sqrt(sum)
    for token in query_vector:
        query_vector[token] /= sum
    return query_vector
```
###Calculate the cosine similarity
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
## [Movie Classifier]()

**Naive Bayes Classifier:**
- It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
- Each movie can be thus classified into multiple genres.
- The below code snippet implements the logic for multinomial Naive Bayes Classifier.

Get the probability of each genre.
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
From the keywords iterate over the genres and get the highest probability for the match
```
def build_and_save():
    row_count = 0
    token_count = 0
    post_probability = {}
    token_genre_count_map = {}
    genre_count_map = {}
    for row in meta_data.itertuples():
        keywords = []
        genres = []
        for col in meta_cols.keys():
            col_values = getattr(row, col)
            parameters = meta_cols[col]
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            if parameters is None:
                keywords.append(col_values if isinstance(col_values, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            else:
                col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                for col_value in col_values:
                    for param in parameters:
                        genres.append(col_value[param])

        tokens = pre_processing(' '.join(keywords))
        for genre in genres:
            if genre in genre_count_map:
                genre_count_map[genre] += 1
            else:
                genre_count_map[genre] = 1
            for token in tokens:
                token_count += 1
                if (genre, token) in token_genre_count_map:
                    token_genre_count_map[(genre, token)] += 1
                else:
                    token_genre_count_map[(genre, token)] = 1

        row_count += 1
        # Uncomment below lines for reading specific number of rows from excel instead of the whole
        # if (row_count == 2):
        #     print(genre_count_map)
        #     break
    for (genre, token) in token_genre_count_map:
        post_probability[(genre, token)] = token_genre_count_map[(genre, token)] / token_count

    prior_probability = {x: genre_count_map[x]/row_count for x in genre_count_map}
    save(prior_probability, post_probability)
    return (prior_probability, post_probability)
```
 
### Movie Recommender
**Metadata Based Recommender**

- We will be using the information such as Credits,Keywords, Ratings and Movie details to recommend movies to a user.
- For any Recommender the more metadata it has the more it is accurate.
- To build a recommender,the following are the steps involved:
-  Decide on the metric or score to rate movies on.
-  Calculate the score for every movie.
-  Sort the movies based on the score and output the top results.
First lets clean the data
```
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```
Now, Lets Apply the clean data function to our feature extractor:
```
Features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)
```
Lets create a Metadata which combines all the features. and apply it to our count vectorizer

```
    metadata['soup'] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])
```
# Deployment On Python Anywhere
**Steps for Deploying Flask Web App**

There are two main ways to set up a Flask application on PythonAnywhere:

 - Starting from scratch using our default versions of Flask
 - Importing a pre-existing app using Manual configuration, and using a virtualenv
   The first option works well if you're just playing around and want to throw something together from scratch. Go to the Web Tab and      hit Add a new Web App, and choose Flask and the Python version you want.

**Setting up your virtualenv**

 - mkvirtualenv --python=/usr/bin/python3.6 my-virtualenv  # use whichever python version you prefer
   pip install flask
  
You'll see the prompt changes from a $ to saying (my-virtualenv)$ -- that's how you can tell your virtualenv is active. 
Whenever you want to work on your project in the console, you need to make sure the virtualenv is active. You can reactivate it at a later date with

  - $ workon my-virtualenv
	 (my-virtualenv)$

**Configuring the WSGI file**

To configure this file, you need to know which file your flask app lives in. The flask app usually looks something like this:

- app = Flask(__name__)

Make a note of the path to that file, and the name of the app variable (is it "app"? Or "application"?) 

 - In this example, let's say it's /home/yourusername/mysite/flask_app.py, and the variable is "app".

In your WSGI file, skip down to the flask section, uncomment it, and make it looks something like this:

- import sys
  path = '/home/yourusername/mysite'
  if path not in sys.path:
    sys.path.insert(0, path)
  from flask_app import app as application
  
**Do not use app.run()**
When you're using Flask on your own PC, you'll often "run" flask using a line that looks something like this:

 - app.run(host='127.0.0.1',port=8000,debug=True)
 That won't work on PythonAnywhere -- the only way your app will appear on the public internet is if it's configured via the web tab, with a wsgi file.

More importantly, 'if app.run() gets called when we import your code, it will crash your app', and you'll see a 504 error on your site, as detailed in Flask504Error

Thankfully, most Flask tutorials out there suggest you put the app.run() inside an if __name__ = '__main__': clause, which will be OK, because that won't get run when we import it.
 
## References
* [Kaggle Kernels](https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels)
* [https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf)
* [https://docs.python.org/2/library/collections.html](https://docs.python.org/2/library/collections.html)
* [https://www.numpy.org/devdocs/](https://www.numpy.org/devdocs/)
* [https://www.ics.uci.edu/~welling/teaching/CS77Bwinter12/presentations/course_Ricci/13-Item-to-Item-Matrix-CF.pdf](https://www.ics.uci.edu/~welling/teaching/CS77Bwinter12/presentations/course_Ricci/13-Item-to-Item-Matrix-CF.pdf)
* [https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels](https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels)
* [https://nlp.stanford.edu/IR-book/pdf/06vect.pdf](https://nlp.stanford.edu/IR-book/pdf/06vect.pd)
* [http://flask.pocoo.org/docs/](http://flask.pocoo.org/docs/)
* [http://pandas.pydata.org/pandas-docs/stable/](http://pandas.pydata.org/pandas-docs/stable/)
