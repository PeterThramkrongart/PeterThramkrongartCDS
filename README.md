# PeterThramkrongartCDS
This project was made Peter Thramkrongart for the Introduction to Cultural Data Science 2020 course at Aarhus University.
The project is about performing advanced text preprocessing in R using SpaCy and openNLP.
PeterThramkrongartCDS_final_project.docx file contains a full report on the project including code.
The MetaData.docx file contains metadate for the coding aspect of this project.
The SpookyTextMining.rmd file contains all code for this project and the typeset code for the .docx document.
This is the document you need to open, if you want to run the code and explore the topic models.
The data are horror texts gathered from Project Gutenberg using the GutenbergR package. 
The /data/ folder contains LDA models saved in R's .rds format, and csv's from all major coding steps.

The csv's has the following format:

title = title of the text

titkeChunked = the title of the text with an added chunk prefix

text = the text of the document. The text can be chunked or not.
