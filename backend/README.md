# 📰 Fake News Detector

A simple web app that uses FastText to detect fake news articles.

## 🔧 Backend (Flask + FastText)
- Located in `/backend`
- Trained FastText model: `best_fasttext_model.bin`
- API endpoint: `/predict`

## 🌐 Frontend (HTML + JS)
- Located in `/frontend`
- Sends news text to backend and displays prediction

## 🚀 Deployment
- **Backend**: Hosted on [Render](https://render.com)
- **Frontend**: Hosted on [Vercel](https://vercel.com)

## 🛠️ Setup
1. Clone the repo
2. Train your model or use the provided `.bin` file
3. Deploy backend via Render
4. Deploy frontend via Vercel
