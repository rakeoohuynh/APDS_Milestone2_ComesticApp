from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from app.logic.products import get_paginated_products, get_product_by_id, clear_products_cache
from app.logic.reviews import get_product_reviews, append_review
from app.logic.recommend import recommend_products
from app.logic.ml_model import predict, predict_with_confidence, build_models, save_new_review
import csv
from pathlib import Path
from collections import Counter

app = Flask(__name__)
app.secret_key = 'beauty-shop-secret-v2'

ADMIN_PASSWORD = 'admin123'

_BASE_DIR = Path(__file__).resolve().parent
_PROC_CSV     = _BASE_DIR / "data" / "processed.csv"
_FEEDBACK_CSV = _BASE_DIR / "data" / "test_feedback.csv"


def _get_admin_stats():
    """Compute dashboard statistics from the CSV files."""
    all_reviews = []

    for csv_path in [_PROC_CSV, _FEEDBACK_CSV]:
        if csv_path.exists():
            with open(csv_path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    text = row.get('review_text', '').strip()
                    if not text:
                        continue
                    try:
                        rating = float(row.get('review_rating', 0))
                    except (ValueError, TypeError):
                        rating = 0.0
                    is_buyer = str(row.get('is_a_buyer', '0')).strip() in ('1', 'True', 'true', 'TRUE')
                    all_reviews.append({
                        'product_id':    row.get('product_id', ''),
                        'product_title': row.get('product_title', row.get('product_id', '')),
                        'author':        row.get('author', 'Anonymous'),
                        'title':         row.get('review_title', ''),
                        'text':          text,
                        'date':          row.get('review_date', '').split(' ')[0],
                        'rating':        rating,
                        'is_buyer':      is_buyer,
                    })

    total = len(all_reviews)
    buy_count     = sum(1 for r in all_reviews if r['is_buyer'])
    not_buy_count = total - buy_count

    # Top 5 products by average rating (min 2 reviews to filter noise)
    from collections import defaultdict
    prod_ratings = defaultdict(list)
    prod_titles  = {}
    for r in all_reviews:
        pid = r['product_id']
        prod_ratings[pid].append(r['rating'])
        prod_titles[pid] = r['product_title']

    top_products = sorted(
        [
            {'id': pid, 'title': prod_titles[pid], 'avg_rating': round(sum(v) / len(v), 2), 'count': len(v)}
            for pid, v in prod_ratings.items()
            if len(v) >= 2
        ],
        key=lambda x: x['avg_rating'],
        reverse=True
    )[:5]

    # 10 most recent reviews from the feedback CSV (user-submitted ones)
    recent_reviews = []
    if _FEEDBACK_CSV.exists():
        with open(_FEEDBACK_CSV, encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if r.get('review_text', '').strip()]
        for row in reversed(rows[-10:]):
            try:
                rating = float(row.get('review_rating', 0))
            except (ValueError, TypeError):
                rating = 0.0
            recent_reviews.append({
                'product_id':    row.get('product_id', ''),
                'product_title': row.get('product_title', row.get('product_id', '')),
                'author':        row.get('author', 'Anonymous'),
                'title':         row.get('review_title', ''),
                'date':          row.get('review_date', '').split(' ')[0],
                'rating':        rating,
                'is_buyer':      str(row.get('is_a_buyer', '0')).strip() in ('1', 'True', 'true', 'TRUE'),
            })

    return {
        'total':         total,
        'buy_count':     buy_count,
        'not_buy_count': not_buy_count,
        'buy_pct':       round(buy_count / total * 100) if total else 0,
        'top_products':  top_products,
        'recent_reviews': recent_reviews,
    }

# print("--- Load AI Models ... ---")
build_models()

@app.route('/')
def index():
    keyword = request.args.get('search_box', '').strip()
    page = request.args.get('page', 1, type=int)
    data = get_paginated_products(query=keyword, page=page)
    return render_template('index.html', **data, last_search=keyword)

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = get_product_by_id(product_id)
    if not product:
        return redirect(url_for('index'))
    reviews = get_product_reviews(product_id)

    recommendations = []
    try:
        recs = recommend_products(int(product_id), top_n=5)
        if not isinstance(recs, str):  # returns "Product not found" string on failure
            for pid in recs['product_id'].astype(str):
                p = get_product_by_id(pid)
                if p:
                    recommendations.append(p)
    except Exception:
        pass

    # Label each review: 1 = good (predicted buyer), 0 = bad (predicted non-buyer)
    # Falls back to rating >= 4 as "good" if ml_model is unavailable
    try:
        from app.logic.ml_model import predict as ml_predict
        ml_available = True
    except Exception:
        ml_available = False

    for review in reviews:
        if ml_available:
            try:
                review['ml_label'] = ml_predict(review['text'], int(review['rating']))
            except Exception:
                review['ml_label'] = 1 if review['rating'] >= 4 else 0
        else:
            review['ml_label'] = 1 if review['rating'] >= 4 else 0

    is_buyer = str(product_id) in session.get('purchased', [])
    return render_template('product.html', product=product, reviews=reviews, recommendations=recommendations, is_buyer=is_buyer)

@app.route('/product/<product_id>/checkout', methods=['GET'])
def checkout(product_id):
    product = get_product_by_id(product_id)
    if not product:
        return redirect(url_for('index'))
    return render_template('checkout.html', product=product)

@app.route('/product/<product_id>/checkout', methods=['POST'])
def place_order(product_id):
    purchased = session.get('purchased', [])
    if str(product_id) not in purchased:
        purchased.append(str(product_id))
        session['purchased'] = purchased
        session.modified = True
    flash('Order placed! You can now leave a verified review.', 'success')
    return redirect(url_for('product_detail', product_id=product_id))

@app.route('/api/predict_label', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    rating = int(data.get('rating', 3))

    label_id, confidence = predict_with_confidence(text, rating)
    label_text = "Buy" if label_id == 1 else "Not Buy"
    confidence_pct = round(confidence * 100)

    return jsonify({"label": label_text, "confidence": confidence_pct})

@app.route('/product/<product_id>/review', methods=['POST'])
def add_review(product_id):
    author = request.form.get('author', '').strip()
    rating = int(request.form.get('rating', '').strip())
    title  = request.form.get('review_title', '').strip()
    text   = request.form.get('review_text', '').strip()

    final_label = int(request.form.get('final_label', 1))

    if not all([author, rating, title, text]):
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('product_detail', product_id=product_id))

    is_buyer = str(product_id) in session.get('purchased', [])
    ok = append_review(product_id, author, rating, title, text, is_buyer)

    if ok:
        clear_products_cache()
        flash('Your review has been submitted. Thank you!', 'success')
    else:
        flash('Could not save your review. Product not found.', 'error')

    return redirect(url_for('product_detail', product_id=product_id))

# ── Admin ────────────────────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('is_admin'):
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        if pwd == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Incorrect password.', 'error')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    stats = _get_admin_stats()
    return render_template('admin.html', **stats)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
