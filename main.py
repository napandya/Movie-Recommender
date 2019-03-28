from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/search/', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    print(query)
    import recommender
    data = {'results': recommender.get_results(query)}
    print("Before jsonify: ", data)
    data = jsonify(data)
    print("After jsonify: ", data)
    return data

if __name__ == '__main__':
   app.run()