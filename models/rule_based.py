import re


def rule_based_spam_score(message: str):
    rules = [
        lambda x: bool(re.search(r'\b(won|win|free|congratulations|urgent|prize|claim|cash)\b', x, re.I)),
        lambda x: bool(re.search(r'http[s]?://|www\.|bit\.ly', x)),
        lambda x: sum(1 for word in x.split() if word.isupper()) > 2,
        lambda x: bool(re.search(r'\b\d{4,}\b', x)),
        lambda x: x.count('!') > 3,
    ]
    return sum(rule(message) for rule in rules)

def is_rule_based_spam(message: str, threshold=2):
    te = rule_based_spam_score(message)
    return te >= threshold, te