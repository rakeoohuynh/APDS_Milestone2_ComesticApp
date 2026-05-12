from flask import Flask, render_template, request
from logic.products import get_paginated_products

app = Flask(__name__)

@app.route('/')
def index():
    keyword = request.args.get('search_box', '').strip()
    page = request.args.get('page', 1, type=int)
    data = get_paginated_products(query=keyword, page=page)
    return render_template('index.html', **data, last_search=keyword)

if __name__ == '__main__':
    app.run(debug=True)
