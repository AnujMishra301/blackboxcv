SKILLS = [
    "python", "java", "c++", "sql", "javascript",
    "machine learning", "data analysis",
    "flask", "django", "react", "node",
    "html", "css", "git"
]

def skill_score(text):
    text = text.lower()
    score = 0
    matched = []

    for skill in SKILLS:
        if skill in text:
            score += 1
            matched.append(skill)

    return score, matched
