from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

statements = [
    "I did not take the money",
    "I swear I was at home all day",
    "I won the lottery yesterday",
    "Click here to get free money",
    "I was working late at the office",
    "You have won a free prize"
]

labels = [0,0,1,1,0,1]   # 0 = Truth, 1 = Lie

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(statements)

model = MultinomialNB()
model.fit(X, labels)

def predict(text):
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    return "LIE 🚨" if prob[1] > prob[0] else "TRUTH ✅"

print("🧠 AI Lie Detector \n")
msg = input("Enter a statement: ")
print("\nResult:", predict(msg))
