GraphAware NLP Using OpenNLP
==========================================

Getting the Software
---------------------

### Server Mode
When using Neo4j in the standalone <a href="http://docs.neo4j.org/chunked/stable/server-installation.html" target="_blank">standalone server</a> mode, you will need the <a href="https://github.com/graphaware/neo4j-framework" target="_blank">GraphAware Neo4j Framework</a> and <a href="https://github.com/graphaware/neo4j-nlp" target="_blank">GraphAware NLP</a>.jar files (both of which you can download here) dropped into the plugins directory of your Neo4j installation. Finally, the following needs to be appended to the `neo4j.conf` file in the `config/` directory:

```
  dbms.unmanaged_extension_classes=com.graphaware.server=/graphaware
  com.graphaware.runtime.enabled=true

  com.graphaware.module.NLP.2=com.graphaware.nlp.module.NLPBootstrapper
```

### For Developers
This package is an extention of the <a href="https://github.com/graphaware/neo4j-nlp" target="_blank">GraphAware NLP</a>, which therefore needs to be packaged and installed beforehand. No other dependencies required.

```
  cd neo4j-nlp
  mvn clean install

  cd ../neo4j-nlp-opennlp
  mvn clean package
```


Introduction and How-To
-------------------------

The Apache OpenNLP library provides basic features for processing natural language text: sentence segmentation, tokenization, lemmatization, part-of-speach tagging, named entities identification, chunking, parsing and sentiment analysis. OpenNLP is implemented by extending the general <a href="https://github.com/graphaware/neo4j-nlp" target="_blank">GraphAware NLP</a> package with extra parameters:

### Tag Extraction / Annotations
```
#Annotate the news
MATCH (n:News)
CALL ga.nlp.annotate({text:n.text, id: n.uuid, textProcessor: "com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor", pipeline: "tokenizer"}) YIELD result
MERGE (n)-[:HAS_ANNOTATED_TEXT]->(result)
RETURN n, result
```

Available pipelines:
  * `tokenizer` - tokenization, lemmatization, stop-words removal, part-of-speach tagging
  * `sentiment` - tokenization, sentiment analysis
  * `tokenizerAndSentiment` - tokenization, lemmatization, stop-words removal, part-of-speach tagging, sentiment analysis
  * `phrase` (not supported yet) - tokenization, stop-words removal, relations, sentiment analysis

### Sentiment Analysis
The current implementation of a sentiment analysis is just a toy - it relies on a file with 100 labeled twitter samples which are used to build a model when Neo4j starts (general recommendation for number of training samples is 10k and more). The current model supports only three options - Positive, Neutral, Negative - which are chosen based on the highest probability (the algorithm returns an array of probabilities for each category). If the highest probability is less then 70% (can easily happen in case of more than 2 categories), the category is not regarded trustworthy and is set to Neutral instead.

The sentiment analysis can be run either as part of the annotation (see paragraph above) or as an independent procedure (see command below) which takes in AnnotatedText nodes, analyzes all attached sentences and adds to them a label corresponding to its sentiment.

```
MATCH (a:AnnotatedText {id: {id}})
CALL ga.nlp.sentiment({node:a, textProcessor: "com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor"}) YIELD result
MATCH (result)-[:CONTAINS_SENTENCE]->(s:Sentence) 
RETURN labels(s) as labels
```

