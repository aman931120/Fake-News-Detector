const NEWS_API_KEY = 'ee750706fc5f48cdb68506cfeac49b0b';
const BACKEND_URL = 'http://127.0.0.1:5000/predict';

async function fetchNews() {
    const url = `https://newsapi.org/v2/everything?q=world&language=en&pageSize=40&apiKey=${NEWS_API_KEY}`;
    try {
        const res = await fetch(url);
        const data = await res.json();
        return data.articles || [];
    } catch (err) {
        console.error("Failed to fetch news:", err);
        return [];
    }
}


async function verifyNews(title, description) {
    try {
        const res = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title, description })
        });

        const data = await res.json();
        return {
            prediction: data.prediction || "Unknown",
            confidence: data.confidence || 0
        };
    } catch (err) {
        console.error("Prediction error:", err);
        return {
            prediction: "Unknown",
            confidence: 0
        };
    }
}

function createNewsCard(article) {
    const col = document.createElement('div');
    col.className = 'news-card';

    const imageDiv = document.createElement('div');
    imageDiv.className = 'news-image';

    if (article.urlToImage) {
        const img = document.createElement('img');
        img.src = article.urlToImage;
        img.alt = 'news image';
        img.style.width = '100%';
        img.style.height = '150px';
        img.style.objectFit = 'cover';
        img.style.borderRadius = '8px';
        imageDiv.appendChild(img);
    } else {
        imageDiv.textContent = "Image not available";
    }

    const title = document.createElement('h5');
    title.textContent = article.title;

    const resultPara = document.createElement('p');
    const label = document.createElement('span');
    label.className = 'label-unknown';
    label.textContent = 'Unknown';
    resultPara.innerHTML = 'News: ';
    resultPara.appendChild(label);

    const confidenceText = document.createElement('small');
    confidenceText.className = 'confidence-text';
    confidenceText.textContent = 'Confidence: 0.0%';

    col.appendChild(imageDiv);
    col.appendChild(title);
    col.appendChild(resultPara);
    col.appendChild(confidenceText);

    return { col, label, confidenceText };
}

async function loadNews() {
    const container = document.getElementById('newsContainer');
    const articles = await fetchNews();

    container.innerHTML = "";

    for (const article of articles) {
        const { col, label, confidenceText } = createNewsCard(article);
        container.appendChild(col);

        const result = await verifyNews(article.title, article.description || "");

        label.textContent = result.prediction;
        confidenceText.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

        label.className = result.prediction.includes("Fake")
            ? 'label-fake'
            : result.prediction.includes("Real")
                ? 'label-real'
                : 'label-unknown';
    }
}

document.addEventListener('DOMContentLoaded', loadNews);
