Preprocessing and topic-modeling horror literature from Project
Gutenberg
================
Peter Thramkrongart

``` r
#Loading packages

pacman::p_load(
  tidyverse,
  tm,
  tidytext,
  topicmodels,
  reshape2,
  LDAvis,
  servr,
  NLP,
  gutenbergr,
  text2vec,
  textstem,
  openNLP,
  tictoc
)


#install.packages("openNLPmodels.en", dependencies=TRUE, repos = "http://datacube.wu.ac.at/")
library(openNLPmodels.en)
```

``` r
#making a lists of the horror titles that we are going to use
IDs <-
  gutenberg_metadata %>%
  filter(str_detect(gutenberg_bookshelf, "Horror") == T &
           language == "en") %>% distinct(title, .keep_all = T)

IDs %>% head()
```

    ## # A tibble: 6 x 8
    ##   gutenberg_id title author gutenberg_autho~ language gutenberg_books~ rights
    ##          <int> <chr> <chr>             <int> <chr>    <chr>            <chr> 
    ## 1           42 The ~ Steve~               35 en       Horror/Movie Bo~ Publi~
    ## 2          209 The ~ James~              113 en       Opera/Best Book~ Publi~
    ## 3          345 Drac~ Stoke~              190 en       Gothic Fiction/~ Publi~
    ## 4          375 An O~ Bierc~              206 en       Horror/US Civil~ Publi~
    ## 5          389 The ~ Mache~              214 en       Horror           Publi~
    ## 6         1188 The ~ Stoke~              190 en       Horror/Gothic F~ Publi~
    ## # ... with 1 more variable: has_text <lgl>

``` r
#Downloading the data

texts <- gutenberg_download(IDs$gutenberg_id, meta_fields = c("title", "author"))
```

    ## Determining mirror for Project Gutenberg from http://www.gutenberg.org/robot/harvest

    ## Using mirror http://aleph.gutenberg.org

``` r
#collapsing data into single texts

texts <- texts %>% group_by(title) %>% mutate(text = glue::glue_collapse(text, " ")) %>% unique()


texts %>% write_csv("texts.csv")

texts %>% head()
```

    ## # A tibble: 6 x 4
    ## # Groups:   title [6]
    ##   gutenberg_id text                             title               author      
    ##          <int> <glue>                           <chr>               <chr>       
    ## 1           42 "                              ~ The Strange Case o~ Stevenson, ~
    ## 2          209 "THE TURN OF THE SCREW  by Henr~ The Turn of the Sc~ James, Henry
    ## 3          345 "                              ~ Dracula             Stoker, Bram
    ## 4          375 "AN OCCURRENCE AT OWL CREEK BRI~ An Occurrence at O~ Bierce, Amb~
    ## 5          389 "THE GREAT GOD PAN  by  ARTHUR ~ The Great God Pan   Machen, Art~
    ## 6         1188 "Transcribed form the 1911 W. F~ The Lair of the Wh~ Stoker, Bram

``` r
#removing unnecessary characters and 

texts$text <- texts$text %>% str_replace_all("[^[:alnum:][:space:]'\\.,]", "")

#This next part takes over an hour to run, so be careful...

#making dictionary for lemmatization. Here I use the "lexicon" method. It is better to use the treetagger method, but that requeires a Perl installation, and is difficult to install in itself as well.

# lemma_dictionary <- texts$text %>%  make_lemma_dictionary( engine = 'lexicon')
#
##Lemmatizing texts
#
# texts$text <- texts$text %>% lemmatize_strings(dictionary = lemma_dictionary)
#
# texts %>% write_csv("lemmatized_texts.csv")
tic()

texts <- read_csv("lemmatized_texts.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   gutenberg_id = col_double(),
    ##   text = col_character(),
    ##   title = col_character(),
    ##   author = col_character()
    ## )

``` r
texts %>% head()
```

    ## # A tibble: 6 x 4
    ##   gutenberg_id text                             title               author      
    ##          <dbl> <chr>                            <chr>               <chr>       
    ## 1           42 STRANGE CASE OF DR. JEKYLL AND ~ The Strange Case o~ Stevenson, ~
    ## 2          209 THE TURN OF THE SCREW by Henry ~ The Turn of the Sc~ James, Henry
    ## 3          345 DRACULA DRACULA by Bram Stoker ~ Dracula             Stoker, Bram
    ## 4          375 a OCCURRENCE AT OWL CREEK BRIDG~ An Occurrence at O~ Bierce, Amb~
    ## 5          389 THE GREAT GOD PAN by ARTHUR MAC~ The Great God Pan   Machen, Art~
    ## 6         1188 transcribe form the 1911 W. Fou~ The Lair of the Wh~ Stoker, Bram

``` r
#defining a generic function for creating and fitting LDA models

custom_LDA_func <- function(df, n) {
  tokens = df$text %>% word_tokenizer()
  it = itoken(tokens, ids = df$title, progressbar = T)
  v = create_vocabulary(it, stopwords = stopwords::stopwords())
  v = prune_vocabulary(v,
                       term_count_min = 5,
                       doc_proportion_max = 0.2)
  
  vectorizer = vocab_vectorizer(v)
  dtm = create_dtm(it, vectorizer, type = "dgTMatrix")
  
  lda_model = LDA$new(
    n_topics = n,
    doc_topic_prior = 0.1,
    topic_word_prior = 0.01
  )
  doc_topic_distr =
    lda_model$fit_transform(
      x = dtm,
      n_iter = 1000,
      convergence_tol = 0.001,
      n_check_convergence = 25,
      progressbar = T
    )
  return(lda_model)
}


#fitting model on lemmatized texts
lemmatized_LDA <- custom_LDA_func(texts, 6)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   1%  |                                                                              |=                                                                     |   1%  |                                                                              |=                                                                     |   2%  |                                                                              |==                                                                    |   2%  |                                                                              |==                                                                    |   3%  |                                                                              |==                                                                    |   4%  |                                                                              |===                                                                   |   4%  |                                                                              |===                                                                   |   5%  |                                                                              |======================================================================| 100%INFO  [11:35:30.799] early stopping at 50 iteration 
    ## 
    ##   |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   1%  |                                                                              |=                                                                     |   1%  |                                                                              |=                                                                     |   2%  |                                                                              |==                                                                    |   2%  |                                                                              |==                                                                    |   3%  |                                                                              |==                                                                    |   4%  |                                                                              |===                                                                   |   4%  |                                                                              |===                                                                   |   5%  |                                                                              |======================================================================| 100%INFO  [11:35:35.105] early stopping at 50 iteration

``` r
#this part is not visible in the knitted document. It starts a server that hosts the visualization.
lemmatized_LDA$plot()


#function for splitting large texts into equal parts

equal_parts <- function(x, np = 5) {
  n <- cut(seq_along(x), np)
  n <- as.integer(n)
  cumsum(c(1, diff(n) > 0))
}

#separating texts in smaller parts

texts <- separate_rows(texts, text) %>%
  group_by(title) %>%
  mutate(grp = equal_parts(text)) %>%  
  group_by(grp, add = TRUE) %>%
  mutate(title = paste(title, grp, sep = "_")) %>%
  summarise(text = paste0(text, collapse = ' '))
```

    ## Warning: The `add` argument of `group_by()` is deprecated as of dplyr 1.0.0.
    ## Please use the `.add` argument instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_warnings()` to see where this warning was generated.

    ## `summarise()` regrouping output by 'title' (override with `.groups` argument)

``` r
texts %>% head()
```

    ## # A tibble: 6 x 3
    ## # Groups:   title [6]
    ##   title                     grp text                                            
    ##   <chr>                   <dbl> <chr>                                           
    ## 1 A Thin Ghost and Other~     1 A THIN GHOST AND other by MONTAGUE RHODES JAMES~
    ## 2 A Thin Ghost and Other~     2 spend a few day at the King s Head ostensibly o~
    ## 3 A Thin Ghost and Other~     3 do. He be puzzle to think why it should have st~
    ## 4 A Thin Ghost and Other~     4 short and I can see they be trouble. My word th~
    ## 5 A Thin Ghost and Other~     5 or out. There be peopleonly a fewon either side~
    ## 6 An Occurrence at Owl C~     1 a OCCURRENCE AT OWL CREEK BRIDGE by Ambrose Bie~

``` r
#difining pipeline for POS tagging

sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
pos_tag_annotator <- Maxent_POS_Tag_Annotator()

#creating the df before the loop
documents_df <- NULL

#defining the chunk size to lessen load on the ram
chunk_size <- 250


#this loop is quite a mouthful. It takes 3088.11 seconds to run on my computer

# for (i in seq_along(texts$title)) {
#
#   para_text <- texts$text[i] #grapping text
#   
#   text_s <- as.String(para_text) # tunring into the right string format

#   #start annotating string
#   annotated_string <- annotate(text_s,
#                                list(
#                                  sent_token_annotator,
#                                  word_token_annotator,
#                                  pos_tag_annotator
#                                )) 
#   
#
#   word_pos <- subset(annotated_string, type == "word") #grapping position of words
#
#   tags_v <- sapply(word_pos$features, `[[`, "POS") #grapping pos tags
#
#   words_v <- text_s[word_pos] #grapping pos tags
#
#   word_pos_df <- data.frame(Token = words_v,
#                             POS = tags_v,
#                             stringsAsFactors = FALSE) #assembling in dataframe
#
#   filtered_lemmas_df <-
#     #we are only interested in nouns verbs and adjectives
#     filter(word_pos_df,
#            POS == "NN" | str_detect(POS, "VB") | str_detect(POS, "JJ")) %>% 
#     select(Token) %>%
#     mutate(Token = tolower(Token))
#
#   #This whole ting is about chunking into even smaller parts
#   word_v <- filtered_lemmas_df$Token
#
#   x <- seq_along(word_v)
#
#   chunks_l <- split(word_v, ceiling(x / chunk_size))
#   if (length(chunks_l[[length(chunks_l)]]) <= chunk_size / 2) {
#     chunks_l[[length(chunks_l) - 1]] <- c(chunks_l[[length(chunks_l) - 1]],
#                                           chunks_l[[length(chunks_l)]])
#     chunks_l[[length(chunks_l)]] <- NULL
#   }
#   chunk_strings_l <- lapply(chunks_l, paste, collapse = " ")
#   chunks_df <- do.call(rbind, chunk_strings_l)
#   textname_v <- gsub("\\..*", "", texts$title[i])
#   chunk_ids_v <- 1:nrow(chunks_df)
#   chunk_names_v <- paste(textname_v, chunk_ids_v, sep = "_")
#   file_df <- data.frame(id = chunk_names_v,
#                         text = chunks_df,
#                         stringsAsFactors = FALSE)
#
#   documents_df <- rbind(documents_df, file_df) adding to the dataframe
#   cat("Done with", texts$title[i], "\r")#status text
# }
toc()
```

    ## 67.2 sec elapsed

``` r
#3088.11 sec elapsed


##removing the chunk tags and saving data
# documents_df <-
#   documents_df %>% mutate(title = id %>% str_replace_all("_\\d+", ""))
# 
# documents_df %>% write_csv("cleaned_texts.csv")


#running LDA

lemmatized_pos_texts <- read_csv("cleaned_texts.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   id = col_character(),
    ##   text = col_character(),
    ##   title = col_character()
    ## )

``` r
lemmatized_pos_LDA <- lemmatized_pos_texts %>% custom_LDA_func(20)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   1%  |                                                                              |=                                                                     |   1%  |                                                                              |=                                                                     |   2%  |                                                                              |==                                                                    |   2%  |                                                                              |==                                                                    |   3%  |                                                                              |==                                                                    |   4%  |                                                                              |===                                                                   |   4%  |                                                                              |===                                                                   |   5%  |                                                                              |====                                                                  |   5%  |                                                                              |====                                                                  |   6%  |                                                                              |=====                                                                 |   6%  |                                                                              |=====                                                                 |   7%  |                                                                              |=====                                                                 |   8%  |                                                                              |======                                                                |   8%  |                                                                              |======                                                                |   9%  |                                                                              |=======                                                               |   9%  |                                                                              |=======                                                               |  10%  |                                                                              |=======                                                               |  11%  |                                                                              |========                                                              |  11%  |                                                                              |========                                                              |  12%  |                                                                              |=========                                                             |  12%  |                                                                              |=========                                                             |  13%  |                                                                              |=========                                                             |  14%  |                                                                              |==========                                                            |  14%  |                                                                              |==========                                                            |  15%  |                                                                              |===========                                                           |  15%  |                                                                              |===========                                                           |  16%  |                                                                              |============                                                          |  16%  |                                                                              |============                                                          |  17%  |                                                                              |============                                                          |  18%  |                                                                              |=============                                                         |  18%  |                                                                              |=============                                                         |  19%  |                                                                              |==============                                                        |  19%  |                                                                              |==============                                                        |  20%  |                                                                              |==============                                                        |  21%  |                                                                              |===============                                                       |  21%  |                                                                              |===============                                                       |  22%  |                                                                              |================                                                      |  22%  |                                                                              |================                                                      |  23%  |                                                                              |================                                                      |  24%  |                                                                              |=================                                                     |  24%  |                                                                              |=================                                                     |  25%  |                                                                              |======================================================================| 100%INFO  [11:37:54.853] early stopping at 250 iteration 
    ## 
    ##   |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   1%  |                                                                              |=                                                                     |   1%  |                                                                              |=                                                                     |   2%  |                                                                              |==                                                                    |   2%  |                                                                              |==                                                                    |   3%  |                                                                              |==                                                                    |   4%  |                                                                              |===                                                                   |   4%  |                                                                              |===                                                                   |   5%  |                                                                              |======================================================================| 100%INFO  [11:38:12.247] early stopping at 50 iteration

``` r
lemmatized_pos_LDA$plot()


#session info
sessioninfo::session_info()
```

    ## - Session info ---------------------------------------------------------------
    ##  setting  value                       
    ##  version  R version 4.0.0 (2020-04-24)
    ##  os       Windows 10 x64              
    ##  system   x86_64, mingw32             
    ##  ui       RTerm                       
    ##  language (EN)                        
    ##  collate  Danish_Denmark.1252         
    ##  ctype    Danish_Denmark.1252         
    ##  tz       Europe/Paris                
    ##  date     2020-11-29                  
    ## 
    ## - Packages -------------------------------------------------------------------
    ##  ! package          * version  date       lib source        
    ##    assertthat         0.2.1    2019-03-21 [1] CRAN (R 4.0.0)
    ##    backports          1.1.7    2020-05-13 [1] CRAN (R 4.0.0)
    ##    blob               1.2.1    2020-01-20 [1] CRAN (R 4.0.0)
    ##    broom              0.5.6    2020-04-20 [1] CRAN (R 4.0.0)
    ##    cellranger         1.1.0    2016-07-27 [1] CRAN (R 4.0.0)
    ##    cli                2.0.2    2020-02-28 [1] CRAN (R 4.0.0)
    ##    colorspace         1.4-1    2019-03-18 [1] CRAN (R 4.0.0)
    ##    crayon             1.3.4    2017-09-16 [1] CRAN (R 4.0.0)
    ##    curl               4.3      2019-12-02 [1] CRAN (R 4.0.0)
    ##    data.table         1.12.8   2019-12-09 [1] CRAN (R 4.0.0)
    ##    DBI                1.1.0    2019-12-15 [1] CRAN (R 4.0.0)
    ##    dbplyr             1.4.4    2020-05-27 [1] CRAN (R 4.0.0)
    ##    digest             0.6.25   2020-02-23 [1] CRAN (R 4.0.0)
    ##    dplyr            * 1.0.0    2020-05-29 [1] CRAN (R 4.0.0)
    ##    ellipsis           0.3.1    2020-05-15 [1] CRAN (R 4.0.0)
    ##    evaluate           0.14     2019-05-28 [1] CRAN (R 4.0.0)
    ##    fansi              0.4.1    2020-01-08 [1] CRAN (R 4.0.0)
    ##    float              0.2-4    2020-04-22 [1] CRAN (R 4.0.3)
    ##    forcats          * 0.5.0    2020-03-01 [1] CRAN (R 4.0.0)
    ##    fs                 1.4.1    2020-04-04 [1] CRAN (R 4.0.0)
    ##    generics           0.0.2    2018-11-29 [1] CRAN (R 4.0.0)
    ##    ggplot2          * 3.3.1    2020-05-28 [1] CRAN (R 4.0.0)
    ##    glue               1.4.1    2020-05-13 [1] CRAN (R 4.0.0)
    ##    gtable             0.3.0    2019-03-25 [1] CRAN (R 4.0.0)
    ##    gutenbergr       * 0.2.0    2020-09-22 [1] CRAN (R 4.0.3)
    ##    haven              2.2.0    2019-11-08 [1] CRAN (R 4.0.0)
    ##    hms                0.5.3    2020-01-08 [1] CRAN (R 4.0.0)
    ##    htmltools          0.4.0    2019-10-04 [1] CRAN (R 4.0.0)
    ##    httpuv             1.5.3.1  2020-05-26 [1] CRAN (R 4.0.0)
    ##    httr               1.4.1    2019-08-05 [1] CRAN (R 4.0.0)
    ##    janeaustenr        0.1.5    2017-06-10 [1] CRAN (R 4.0.3)
    ##    jsonlite           1.6.1    2020-02-02 [1] CRAN (R 4.0.0)
    ##    knitr              1.28     2020-02-06 [1] CRAN (R 4.0.0)
    ##    koRpus           * 0.13-3   2020-10-15 [1] CRAN (R 4.0.3)
    ##    koRpus.lang.en   * 0.1-4    2020-10-24 [1] CRAN (R 4.0.3)
    ##    later              1.0.0    2019-10-04 [1] CRAN (R 4.0.0)
    ##    lattice            0.20-41  2020-04-02 [2] CRAN (R 4.0.0)
    ##    LDAvis           * 0.3.2    2015-10-24 [1] CRAN (R 4.0.3)
    ##    lgr                0.4.1    2020-10-20 [1] CRAN (R 4.0.3)
    ##    lifecycle          0.2.0    2020-03-06 [1] CRAN (R 4.0.0)
    ##    lubridate          1.7.8    2020-04-06 [1] CRAN (R 4.0.0)
    ##    magrittr           1.5      2014-11-22 [1] CRAN (R 4.0.0)
    ##    Matrix             1.2-18   2019-11-27 [2] CRAN (R 4.0.0)
    ##    mlapi              0.1.0    2017-12-17 [1] CRAN (R 4.0.3)
    ##    modelr             0.1.8    2020-05-19 [1] CRAN (R 4.0.0)
    ##    modeltools         0.2-23   2020-03-05 [1] CRAN (R 4.0.3)
    ##    munsell            0.5.0    2018-06-12 [1] CRAN (R 4.0.0)
    ##    nlme               3.1-147  2020-04-13 [2] CRAN (R 4.0.0)
    ##    NLP              * 0.2-1    2020-10-14 [1] CRAN (R 4.0.3)
    ##    openNLP          * 0.2-7    2019-10-26 [1] CRAN (R 4.0.3)
    ##    openNLPdata        1.5.3-4  2017-11-12 [1] CRAN (R 4.0.3)
    ##    openNLPmodels.en * 1.5-1    2020-11-26 [1] local         
    ##    pacman             0.5.1    2019-03-11 [1] CRAN (R 4.0.0)
    ##    pillar             1.4.4    2020-05-05 [1] CRAN (R 4.0.0)
    ##    pkgconfig          2.0.3    2019-09-22 [1] CRAN (R 4.0.0)
    ##    plyr               1.8.6    2020-03-03 [1] CRAN (R 4.0.0)
    ##    promises           1.1.0    2019-10-04 [1] CRAN (R 4.0.0)
    ##    proxy              0.4-24   2020-04-25 [1] CRAN (R 4.0.3)
    ##    purrr            * 0.3.4    2020-04-17 [1] CRAN (R 4.0.0)
    ##    R6                 2.4.1    2019-11-12 [1] CRAN (R 4.0.0)
    ##    Rcpp               1.0.4.6  2020-04-09 [1] CRAN (R 4.0.0)
    ##    readr            * 1.3.1    2018-12-21 [1] CRAN (R 4.0.0)
    ##    readxl             1.3.1    2019-03-13 [1] CRAN (R 4.0.0)
    ##    reprex             0.3.0    2019-05-16 [1] CRAN (R 4.0.0)
    ##    reshape2         * 1.4.4    2020-04-09 [1] CRAN (R 4.0.0)
    ##    RhpcBLASctl        0.20-137 2020-05-17 [1] CRAN (R 4.0.3)
    ##  D rJava              0.9-12   2020-03-24 [1] CRAN (R 4.0.0)
    ##    RJSONIO            1.3-1.4  2020-01-15 [1] CRAN (R 4.0.3)
    ##    rlang              0.4.6    2020-05-02 [1] CRAN (R 4.0.0)
    ##    rmarkdown          2.4      2020-09-30 [1] CRAN (R 4.0.2)
    ##    rsparse            0.4.0    2020-04-01 [1] CRAN (R 4.0.3)
    ##    rstudioapi         0.11     2020-02-07 [1] CRAN (R 4.0.0)
    ##    rvest              0.3.5    2019-11-08 [1] CRAN (R 4.0.0)
    ##    scales             1.1.1    2020-05-11 [1] CRAN (R 4.0.0)
    ##    servr            * 0.20     2020-10-19 [1] CRAN (R 4.0.3)
    ##    sessioninfo        1.1.1    2018-11-05 [1] CRAN (R 4.0.0)
    ##    slam               0.1-47   2019-12-21 [1] CRAN (R 4.0.3)
    ##    SnowballC          0.7.0    2020-04-01 [1] CRAN (R 4.0.3)
    ##    stopwords          2.0      2020-04-14 [1] CRAN (R 4.0.3)
    ##    stringi            1.4.6    2020-02-17 [1] CRAN (R 4.0.0)
    ##    stringr          * 1.4.0    2019-02-10 [1] CRAN (R 4.0.0)
    ##    sylly            * 0.1-6    2020-09-20 [1] CRAN (R 4.0.3)
    ##    sylly.en           0.1-3    2018-03-19 [1] CRAN (R 4.0.3)
    ##    text2vec         * 0.6      2020-02-18 [1] CRAN (R 4.0.3)
    ##    textstem         * 0.1.4    2018-04-09 [1] CRAN (R 4.0.3)
    ##    tibble           * 3.0.1    2020-04-20 [1] CRAN (R 4.0.0)
    ##    tictoc           * 1.0      2014-06-17 [1] CRAN (R 4.0.3)
    ##    tidyr            * 1.1.0    2020-05-20 [1] CRAN (R 4.0.0)
    ##    tidyselect         1.1.0    2020-05-11 [1] CRAN (R 4.0.0)
    ##    tidytext         * 0.2.6    2020-09-20 [1] CRAN (R 4.0.3)
    ##    tidyverse        * 1.3.0    2019-11-21 [1] CRAN (R 4.0.0)
    ##    tm               * 0.7-7    2019-12-12 [1] CRAN (R 4.0.3)
    ##    tokenizers         0.2.1    2018-03-29 [1] CRAN (R 4.0.3)
    ##    topicmodels      * 0.2-11   2020-04-19 [1] CRAN (R 4.0.3)
    ##    triebeard          0.3.0    2016-08-04 [1] CRAN (R 4.0.3)
    ##    urltools           1.7.3    2019-04-14 [1] CRAN (R 4.0.3)
    ##    usethis            1.6.1    2020-04-29 [1] CRAN (R 4.0.0)
    ##    utf8               1.1.4    2018-05-24 [1] CRAN (R 4.0.0)
    ##    vctrs              0.3.0    2020-05-11 [1] CRAN (R 4.0.0)
    ##    withr              2.2.0    2020-04-20 [1] CRAN (R 4.0.0)
    ##    xfun               0.14     2020-05-20 [1] CRAN (R 4.0.0)
    ##    xml2               1.3.2    2020-04-23 [1] CRAN (R 4.0.0)
    ##    yaml               2.2.1    2020-02-01 [1] CRAN (R 4.0.0)
    ## 
    ## [1] D:/Users/thram_000/Documents/R/win-library/4.0
    ## [2] C:/Program Files/R/R-4.0.0/library
    ## 
    ##  D -- DLL MD5 mismatch, broken installation.
