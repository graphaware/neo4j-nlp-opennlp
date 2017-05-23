GraphAware NLP Using OpenNLP
==========================================

Getting the Software
---------------------

### Server Mode
When using Neo4j in the <a href="http://docs.neo4j.org/chunked/stable/server-installation.html" target="_blank">standalone server</a> mode, you will need the <a href="https://github.com/graphaware/neo4j-framework" target="_blank">GraphAware Neo4j Framework</a> and <a href="https://github.com/graphaware/neo4j-nlp" target="_blank">GraphAware NLP</a> .jar files (both of which you can download here) dropped into the plugins directory of your Neo4j installation. Finally, the following needs to be appended to the `neo4j.conf` file in the `config/` directory:

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

### Customizing pipeline models
For the moment, only Named Entity Recognition (NER) in OpenNLP Text Processor is customizable. To use it, run the following:
```
CALL ga.nlp.train({[project: "my_XY",] alg: "NER", model: "component", file: "<path_to_your_training_file>" [, lang: "en"]})
```
  * `project` (case insensitive) allows to specify in the `annotate()` procedure that we also want to use the custom NER model(s)
  * `alg` (case insensitive) specifies which algorithm is about to be trained; currently available algs: NER, sentiment
  * `model` in combination with `alg` (and with `project` if it's specified) provides a unique identifier of the model that you want to train (will be used for e.g. saving it into .bin file)
  * `file` is path to the training data file
  * `lang` (default is "en") specifies the language
  * resulting model is save to a binary file in Neo4j's `import/` directory: `<lang>-<alg>-<model>-<project>.bin`

```
# Example of a text to analyze
CREATE (l:Lesson {lesson: "Power system distribution at Kennedy Space Center (KSC) consists primarily of high-voltage, underground cables. These cables include approximately 5000 splices.ľ Splice failures result in arc flash events that are extremely hazardous to personnel in the vicinity of the arc flash. Some construction and maintenance tasks cannot be performed effectively in the required personal protective equipment (PPE), and de-energizing the cables is not feasible due to cost, lost productivity, and safety risk to others implementing the required outages. To verify alternate and effective mitigations, arc flash testing was conducted in a controlled environment. The arc flash effects were greater than expected. Testing also demonstrated the addition of neutral grounding resistors (NGRs) would result in substantial reductions to arc flash effects. As a result, NGRs are being installed on KSC primary substation transformers. The presence of the NGRs, enable usage of less cumbersome PPE.  Laboratory testing revealed higher than anticipated safety risks from a potential arc-flash event in a manhole environment when conducted at KSCęs unreduced fault current levels.ľ The safety risks included bright flash, excessive sound, and smoke.ľľ Due to these findings and absence of other mitigations installed at the time, manhole entries require full arc-flash PPE.ľ Furthermore, manhole entries were temporarily restricted to short duration inspections until further mitigations could be implemented.ľ With installation of neutral grounding resistors (NGRs) on substation transformers, the flash, sound and flame energy was reduced.ľ The hazard reduction was so substantial that the required PPE would be less cumbersome and enable effective performance of maintenance tasks in the energized configuration."})

WITH l

# Annotate it and use newly trained NER model(s)
CALL ga.nlp.annotate({text:l.lesson, id: l.uuid, customProject: "my_XY"}) YIELD result
MERGE (l)-[:HAS_ANNOTATED_TEXT]->(result)
RETURN l, result;
```
