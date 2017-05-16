/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Properties;
import java.util.HashMap;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.lemmatizer.LemmatizerModel;      // needs OpenNLP >=1.7
import opennlp.tools.lemmatizer.LemmatizerME;         // needs OpenNLP >=1.7
import opennlp.tools.lemmatizer.DictionaryLemmatizer; // needs OpenNLP >=1.7
//import opennlp.tools.lemmatizer.SimpleLemmatizer;   // for OpenNLP < 1.7
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.util.Span;
import opennlp.tools.util.model.BaseModel;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author ale
 */
public class OpenNLPPipeline {

    public static final String PROPERTY_PATH_CHUNKER_MODEL = "chuncker";
    public static final String PROPERTY_PATH_POS_TAGGER_MODEL = "pos";
    public static final String PROPERTY_PATH_SENTENCE_MODEL = "sentence";
    public static final String PROPERTY_PATH_TOKENIZER_MODEL = "tokenizer";
    public static final String PROPERTY_PATH_LEMMATIZER_MODEL = "lemmatizer";
    public static final String PROPERTY_PATH_SENTIMENT_MODEL = "sentiment";

    public static final String PROPERTY_DEFAULT_CHUNKER_MODEL = "en-chunker.bin";
    public static final String PROPERTY_DEFAULT_POS_TAGGER_MODEL = "en-pos-maxent.bin";
    public static final String PROPERTY_DEFAULT_SENTENCE_MODEL = "en-sent.bin";
    public static final String PROPERTY_DEFAULT_TOKENIZER_MODEL = "en-token.bin";
    public static final String PROPERTY_DEFAULT_LEMMATIZER_MODEL = "en-lemmatizer.dict";
    public static final String PROPERTY_DEFAULT_SENTIMENT_MODEL = "";

    public static final String PROPERTY_DEFAULT_SENTIMENT_TRAIN = "sentiment_tweets.train";    

    // Named Entities: mapping from labels to models
    public static HashMap<String, String> PROPERTY_NE_MODELS = new HashMap<>();

    // Named Entities: mapping from labels to identifiers that are used in the graph
    //public static HashMap<String, String> PROPERTY_NE_IDS = new HashMap<String, String>();

    // Named Entities: objects
    public HashMap<String, NameFinderME> nameDetector = new HashMap<String, NameFinderME>();

    public final List<String> annotators;
    public final List<String> stopWords;

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    private TokenizerME wordBreaker;
    private POSTaggerME posme;
    private ChunkerME chunkerME;
    private SentenceDetectorME sentenceDetector;
    //private LemmatizerME lemmaDetector;
    private DictionaryLemmatizer lemmaDetector; // needs OpenNLP >=1.7
    //private SimpleLemmatizer lemmaDetector; // for OpenNLP < 1.7
    private DocumentCategorizerME sentimentDetector;

    public OpenNLPPipeline(Properties properties) {
        // Named Entities: mapping from labels to models
        PROPERTY_NE_MODELS = new HashMap<String, String>();
        PROPERTY_NE_MODELS.put("namefinder", "en-ner-person.bin");
        PROPERTY_NE_MODELS.put("datefinder", "en-ner-date.bin");
        PROPERTY_NE_MODELS.put("locationfinder", "en-ner-location.bin");
        PROPERTY_NE_MODELS.put("timefinder", "en-ner-time.bin");
        PROPERTY_NE_MODELS.put("organizationfinder", "en-ner-organization.bin");
        PROPERTY_NE_MODELS.put("moneyfinder", "en-ner-money.bin");
        PROPERTY_NE_MODELS.put("percentagefinder", "en-ner-percentage.bin");

        // Named Entities: mapping from labels to identifiers that are used in the graph
        /*PROPERTY_NE_IDS = new HashMap<String, String>();
        PROPERTY_NE_IDS.put("namefinder", "person");
        PROPERTY_NE_IDS.put("datefinder", "date");
        PROPERTY_NE_IDS.put("locationfinder", "location");
        PROPERTY_NE_IDS.put("timefinder", "time");
        PROPERTY_NE_IDS.put("organizationfinder", "organization");
        PROPERTY_NE_IDS.put("moneyfinder", "money");
        PROPERTY_NE_IDS.put("percentagefinder", "percentage");*/

        nameDetector = new HashMap<String, NameFinderME>();

        annotators = Arrays.asList(properties.getProperty("annotators", "").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());
        stopWords  = Arrays.asList(properties.getProperty("stopword", "").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());

        init(properties);
    }

    private void init(Properties properties) {
        try {
            senteceSplitter(properties);
            tokenizer(properties);
            posTagger(properties);
            chuncker(properties);
            namedEntitiesFinders(properties);
            lemmatizer(properties);
            categorizer(properties);

        } catch (IOException e) {
            LOG.error("Could not initialize OpenNLP models: " + e.getMessage());
            throw new RuntimeException("Could not initialize OpenNLP models", e);
        }
    }

    protected void chuncker(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_CHUNKER_MODEL, PROPERTY_DEFAULT_CHUNKER_MODEL);
        ChunkerModel chunkerModel = loadModel(ChunkerModel.class, is);
        closeInputStream(is, PROPERTY_PATH_CHUNKER_MODEL);
        chunkerME = new ChunkerME(chunkerModel);
    }

    private void posTagger(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_POS_TAGGER_MODEL, PROPERTY_DEFAULT_POS_TAGGER_MODEL);
        POSModel pm = loadModel(POSModel.class, is);
        closeInputStream(is, PROPERTY_PATH_POS_TAGGER_MODEL);
        posme = new POSTaggerME(pm);
    }

    private void tokenizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_TOKENIZER_MODEL, PROPERTY_DEFAULT_TOKENIZER_MODEL);
        TokenizerModel tm = loadModel(TokenizerModel.class, is);
        closeInputStream(is, PROPERTY_PATH_TOKENIZER_MODEL);
        wordBreaker = new TokenizerME(tm);
    }

    private void senteceSplitter(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_SENTENCE_MODEL, PROPERTY_DEFAULT_SENTENCE_MODEL);
        SentenceModel sentenceModel = loadModel(SentenceModel.class, is);
        closeInputStream(is, PROPERTY_PATH_SENTENCE_MODEL);
        sentenceDetector = new SentenceDetectorME(sentenceModel);
    }

    private void namedEntitiesFinders(Properties properties) throws FileNotFoundException {
        for (String key : PROPERTY_NE_MODELS.keySet()) {
          InputStream is = getInputStream(properties, key, PROPERTY_NE_MODELS.get(key));
          TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
          closeInputStream(is, key);
          nameDetector.put(key, new NameFinderME(nameModel));
        }
    }

    private void lemmatizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_LEMMATIZER_MODEL, PROPERTY_DEFAULT_LEMMATIZER_MODEL);
        //LemmatizerModel lemmaModel = loadModel(LemmatizerModel.class, is);
        lemmaDetector = new DictionaryLemmatizer(is); // needs OpenNLP >=1.7
        //lemmaDetector = new SimpleLemmatizer(is); // for OpenNLP < 1.7
        closeInputStream(is, PROPERTY_PATH_LEMMATIZER_MODEL);
        //lemmaDetector = new LemmatizerME(lemmaModel);
    }

    private void categorizer(Properties properties) throws FileNotFoundException {
        /*InputStream is = getInputStream(properties, PROPERTY_PATH_SENTIMENT_MODEL, PROPERTY_DEFAULT_SENTIMENT_MODEL);
        DoccatModel doccatModel = loadModel(DoccatModel.class, is);
        closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
        sentimentDetector = new DocumentCategorizerME(doccatModel);*/

        // First: train a model
        DoccatModel model = null;
        ImprovisedInputStreamFactory dataIn = null;
        try {
          dataIn = new ImprovisedInputStreamFactory(properties, PROPERTY_PATH_SENTIMENT_MODEL, PROPERTY_DEFAULT_SENTIMENT_TRAIN);
          ObjectStream<String> lineStream = new PlainTextByLineStream(dataIn, "UTF-8");
          ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
          TrainingParameters params = new TrainingParameters();
          params.put(TrainingParameters.CUTOFF_PARAM, "2");
          params.put(TrainingParameters.ITERATIONS_PARAM, "30");
          model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());
          //model = DocumentCategorizerME.train("en", sampleStream, TrainingParameters.defaultParams(), new DoccatFactory());
        } catch (IOException e) {
          e.printStackTrace();
        } finally {
          if (dataIn!=null)
            dataIn.closeInputStream();
        }

        if (model!=null)
          sentimentDetector = new DocumentCategorizerME(model);
    }

    public void annotate(OpenNLPAnnotation document) {

        String text = document.getText();
        try {
            Span sentences[] = sentenceDetector.sentPosDetect(text);
            document.setSentences(sentences);

            document.getSentences().stream().forEach(sentence -> {
                if (annotators.contains("tokenize")) {
                  // Tokenization
                  Span[] word_spans = wordBreaker.tokenizePos(sentence.getSentence());

                  if (!annotators.contains("stopword"))
                    sentence.setWordsAndSpans(word_spans);
                  else {
                    String[] words = Span.spansToStrings(word_spans, sentence.getSentence());
                    ArrayList<Span> fin_spans = new ArrayList<Span>();
                    ArrayList<String> fin_words = new ArrayList<String>();
                    for (int i=0; i<words.length; i++) {
                      if (stopWords.contains(words[i]))
                        continue;
                      fin_words.add(words[i]);
                      fin_spans.add(word_spans[i]);
                    }
                    sentence.setWords(fin_words.toArray(new String[fin_words.size()]));
                    sentence.setWordSpans(fin_spans.toArray(new Span[fin_spans.size()]));
                  }

                  if (annotators.contains("pos")) {
                    // Part of Speach
                    String[] posTags = posme.tag(sentence.getWords());
                    sentence.setPosTags(posTags);

                    if (annotators.contains("lemma")) {
                      // Lemmatizer: for each token/word get its lemma
                      //   Version a: for OpenNLP >= 1.7
                      String[] finLemmas = lemmaDetector.lemmatize(sentence.getWords(), posTags);
                      sentence.setLemmas(finLemmas);
                      //    Version b: for OpenNLP < 1.7
                      /*String[] words = sentence.getWords();
                      for (int i=0; i<posTags.length; i++) {
                        try {
                          sentence.setLemma(i, lemmaDetector.lemmatize(words[i], posTags[i]));
                        } catch (ArrayIndexOutOfBoundsException ex) {
                          LOG.error("Index %d not in array of words.", i);
                        }
                      }*/
                    } // lemma
                    else
                      sentence.setDefaultLemmas();

                    if (annotators.contains("relation")) {
                      // Chunking
                      Span[] chunks = chunkerME.chunkAsSpans(sentence.getWords(), posTags);
                      sentence.setChunks(chunks);
                      String[] chunkStrings = Span.spansToStrings(chunks, sentence.getWords());
                      sentence.setChunkStrings(chunkStrings);
                      List<String> chunkSentiments = new ArrayList<String>();
                      for (int i = 0; i < chunks.length; i++) {
                        //if (chunks[i].getType().equals("NP"))
                            sentence.addPhraseIndex(i);
                        // run sentiment analysis on chunks? (would be needed to ensure only chunks with >=2 words are used)
                        /*if (annotators.contains("sentiment")) {
                          double[] outcomes = sentimentDetector.categorize(chunkStrings[i]);
                          String category = sentimentDetector.getBestCategory(outcomes);
                          if (Arrays.stream(outcomes).max().getAsDouble()<0.7)
                            category = "2"; // not conclusive, setting it to "Neutral"
                          chunkSentiments.add(category);
                          LOG.info("Sentence: " + chunkStrings[i] + "; category = " + category + "; outcomes = " + Arrays.toString(outcomes));
                        }*/
                      }
                      if (chunkSentiments!=null)
                        sentence.setChunkSentiments(chunkSentiments.toArray(new String[chunkSentiments.size()]));
                    } // chunk
                    else
                      sentence.setDefaultChunks();
                  } // pos
                  else {
                    sentence.setDefaultPosTags();
                    sentence.setDefaultLemmas();
                    sentence.setDefaultChunks();
                  }

                  if (annotators.contains("ner")) {
                    // Named Entities identification
                    for (String key : PROPERTY_NE_MODELS.keySet()) {
                      Arrays.asList(nameDetector.get(key).find(sentence.getWords())).stream()
                            .forEach(span -> {
                                sentence.setNamedEntity(span.getStart(), span.getEnd(), span.getType());
                            });
                    }
                  } // ner
                  else
                    sentence.setDefaultNamedEntities();
                } // tokenize

                if (annotators.contains("sentiment")) {
                  double[] outcomes = sentimentDetector.categorize(sentence.getSentence());
                  String category = sentimentDetector.getBestCategory(outcomes);
                  if (Arrays.stream(outcomes).max().getAsDouble()<0.7)
                    category = "2"; // not conclusive, setting it to "Neutral"
                  sentence.setSentiment(category);
                  LOG.info("Sentence: " + sentence.getSentence() + "; category = " + category + "; outcomes = " + Arrays.toString(outcomes));
                }
            });
        } catch (Exception ex) {
            LOG.error("Error processing sentence for phrase: " + text, ex);
            throw new RuntimeException("Error processing sentence for phrase: " + text, ex);
        }
    }

    private <T extends BaseModel> T loadModel(Class<T> clazz, InputStream in) {
        try {
            Constructor<T> modelConstructor = clazz.getConstructor(InputStream.class);
            T model = modelConstructor.newInstance(in);
            return model;
        } catch (Exception ex) {
            LOG.error("Error while initializing model of class: " + clazz, ex);
            throw new RuntimeException("Error while initializing model of class: " + clazz, ex);
        }
    }

    private InputStream getInputStream(Properties properties, String property, String defaultValue) {
        String path = properties.getProperty(property, defaultValue);
        InputStream is;
        try {
            if (path.startsWith("file://")) {
                is = new FileInputStream(new File(new URI(path)));
            } else if (path.startsWith("/")) {
                is = new FileInputStream(new File(path));
            } else {
                is = this.getClass().getResourceAsStream(path);
            }
        } catch (FileNotFoundException | URISyntaxException ex) {
            LOG.error("Error while loading model from path: " + path, ex);
            throw new RuntimeException("Error while loading model from path: " + path, ex);
        }
        return is;
    }

    private void closeInputStream(InputStream is, String type) {
        try {
          is.close();
        } catch (IOException ex) {
          LOG.warn("Attept to close stream for " + type + " model failed.");
        }
        return;
    }


    class ImprovisedInputStreamFactory implements InputStreamFactory {
      //private final String fileName;
      private final Properties properties;
      private final String property;
      private final String defaultProperty;
      private InputStream is;

      ImprovisedInputStreamFactory(Properties properties, String property, String defaultValue) {
        //this.fileName = name;
        this.properties = properties;
        this.property = property;
        this.defaultProperty = defaultValue;

        this.is = getInputStream(this.properties, this.property, this.defaultProperty);
      }

      @Override
      public InputStream createInputStream() throws IOException {
        //return getInputStream(this.properties, this.property, this.defaultProperty);
        return this.is;
      }

      public void closeInputStream() {
        try {
          if (this.is!=null)
            this.is.close();
        } catch (IOException ex) {
          LOG.warn("Attept to close input stream failed.");
        }
      }
    }

}
