# Automatic Labelling of Sentences in Research Papers

The problem we intend to solve involves performing automatic labelling from a set of research papers from three different fields.

The corpus used is available in UCI ML Repository: 
https://archive.ics.uci.edu/ml/machinelearningdatabases/00311/

This dataset contains sentences from the abstract and introduction of 30 annotated research papers. These articles come from three different domains:
  1. PLoS Computational Biology (PLOS)
  2. The machine learning repository on arXiv (ARXIV)
  3. The psychology journal Judgment and Decision Making (JDM)




The 5 classes (or labels) contained in each article are:

1. AIMX. The specific research goal of the paper
2. OWNX. The author’s own work, e.g. methods, results, conclusions
3. CONT. Contrast, comparison or critique of past work
4. BASE. Past work that provides the basis for the work in the article.
5. MISC. Any other sentences



Therefore the task we performed within this repository focuses on labelling a given sentence with one of these 5 classes.

A more in deep explanation of the work developed can be found in the Main.ipynb file: 
https://github.com/Riccocez/Text-Labelling/blob/master/Main.ipynb
