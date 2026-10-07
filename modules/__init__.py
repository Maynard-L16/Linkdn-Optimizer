"""NLP modules for the LinkedIn Profile Optimizer & Job Matcher.

Pipeline order (see modules/pipeline.py):
    text_extraction -> preprocessing -> skill_extraction -> tfidf_analysis
    -> ner_analysis -> semantic_similarity -> scoring -> recommendations
"""
