pacman::p_load(tidyverse,
                 tm,
                 tidytext,
                 topicmodels,
                 reshape2,
                 LDAvis,
                 servr,
                 NLP,
                 gutenbergr,
                 forcats,
                 text2vec,
                 textstem,
                 openNLP, SnowballC,tictoc)
  
  
  #install.packages("openNLPmodels.en", dependencies=TRUE, repos = "http://datacube.wu.ac.at/")
library(openNLPmodels.en)

devtools::install_github("statsmaths/coreNLP")
coreNLP::downloadCoreNLP()

IDs <-
    gutenberg_metadata %>%
    filter(str_detect(gutenberg_bookshelf, "Horror") == T &
             language == "en") %>% distinct(title,.keep_all = T)
  
texts <- gutenberg_download(IDs$gutenberg_id, meta_fields = c("title", "author"))  
texts <- texts %>% group_by(title) %>% mutate(text = glue::glue_collapse(text, " ")) %>% unique()


texts %>% write_csv("texts.csv")

texts$text <- texts$text %>% str_replace_all("[^[:alnum:][:space:]'\\.,]", "")

lemma_dictionary <- texts$text %>%  make_lemma_dictionary( engine = 'lexicon')
   
words <-
  texts %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word")


word_counts <- words %>%
  count(title, word , sort = T)

total_words <- word_counts %>%
  group_by(title) %>%
  summarize(total = sum(n))


word_counts <- left_join(word_counts, total_words)

book_tf_idf <- word_counts %>%
  bind_tf_idf(word, title, n)

book_tf_idf %>%
  group_by(title) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = title)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ title, scales = "free_y") +
  labs(x = "tf-idf", y = NULL)


books_dtm <- word_counts %>%
  cast_dtm(title, word, n)

# books_lda <- LDA(books_dtm, k = 10, control = list(seed = 1234))
#
# books_topics <- tidy(books_lda, matrix = "beta")
#
#
# top_terms <- books_topics %>%
#   group_by(topic) %>%
#   top_n(5, beta) %>%
#   ungroup() %>%
#   arrange(topic, -beta)
#
# top_terms %>%
#   mutate(term = reorder_within(term, beta, topic)) %>%
#   ggplot(aes(beta, term, fill = factor(topic))) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   scale_y_reordered()


topicmodels2LDAvis <- function(x, ...) {
  post <- topicmodels::posterior(x)
  if (ncol(post[["topics"]]) < 3)
    stop("The model must contain > 2 topics")
  mat <- x@wordassignments
  LDAvis::createJSON(
    phi = post[["terms"]],
    theta = post[["topics"]],
    vocab = colnames(post[["terms"]]),
    doc.length = slam::row_sums(mat, na.rm = TRUE),
    term.frequency = slam::col_sums(mat, na.rm = TRUE)
  )
}

books_lda <- LDA(books_dtm, k = 5, control = list(seed = 123))
serVis(topicmodels2LDAvis(books_lda))



tokens = texts$text %>% word_tokenizer()
it = itoken(tokens, ids = texts$title, progressbar = T)
v = create_vocabulary(it, stopwords = stopwords::stopwords())
v = prune_vocabulary(v, term_count_min = 5, doc_proportion_max = 0.2)

vectorizer = vocab_vectorizer(v)
dtm = create_dtm(it, vectorizer, type = "dgTMatrix")

lda_model = LDA$new(n_topics = 6, doc_topic_prior = 0.1, topic_word_prior = 0.01)
doc_topic_distr = 
  lda_model$fit_transform(x = dtm, n_iter = 1000, 
                          convergence_tol = 0.001, n_check_convergence = 25, 
                          progressbar = T)

lda_model$plot()

text <- texts$text[7] %>% as.String()
sent_annot = Maxent_Sent_Token_Annotator()
word_annot = Maxent_Word_Token_Annotator()
pos_annot = Maxent_POS_Tag_Annotator()
loc_annot = Maxent_Entity_Annotator(kind = "location")
people_annot = Maxent_Entity_Annotator(kind = "person") #annotate person

annotated_string<- NLP::annotate(text, list(sent_annot,word_annot,pos_annot,loc_annot,people_annot))

k <- sapply(annotated_string$features, `[[`,"kind")
locations = text[annotated_string[k == "location"]]
people = text[annotated_string[k == "person"]]



