import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from modules.embeddings import load_embedding_model, generate_embedding
from sklearn.metrics.pairwise import cosine_similarity

model = load_embedding_model()
ref = "Inheritance allows a child class to acquire properties and behaviors from a parent class, promoting code reusability. Method overriding allows the child to provide a specific implementation."

texts = [
    "It is used to reuse code. You use it with classes. I think single is just one class.",
    "Inheritance is when you hide variables inside a class so no one can see them. Single inheritance is hiding one variable. Multilevel is hiding many variables.",
    "I like to eat pizza and watch movies. Java is a coffee. Single coffee is espresso, multilevel is a latte."
]

ref_emb = generate_embedding(ref.lower(), model)
for t in texts:
    t_emb = generate_embedding(t.lower(), model)
    sim = cosine_similarity([t_emb], [ref_emb])[0][0]
    print(f"Sim: {sim:.4f} -> {t[:30]}...")
