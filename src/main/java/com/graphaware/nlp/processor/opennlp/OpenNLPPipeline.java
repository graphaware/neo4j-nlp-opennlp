/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Properties;
import java.util.HashMap;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;
import java.util.Collections;
import java.util.Map;
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
import opennlp.tools.namefind.NameSample;
import opennlp.tools.lemmatizer.LemmatizerModel;      // needs OpenNLP >=1.7
import opennlp.tools.lemmatizer.LemmatizerME;         // needs OpenNLP >=1.7
import opennlp.tools.lemmatizer.DictionaryLemmatizer; // needs OpenNLP >=1.7
//import opennlp.tools.lemmatizer.SimpleLemmatizer;   // for OpenNLP < 1.7
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.util.Span;
import opennlp.tools.util.model.BaseModel;
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
    public static final String PROPERTY_DEFAULT_SENTIMENT_MODEL = "en-sentiment-tweets_toy.bin";

    public static final String PROPERTY_DEFAULT_SENTIMENT_TRAIN = "sentiment_tweets.train";
    public static final String DEFAULT_PROJECT_VALUE = "default";

    // Named Entities: mapping from labels to models
    public static HashMap<String, String> PROPERTY_NE_MODELS;

    // Custom models related: key vs file path
    private String globalProject;
    public static HashMap<String, String> CUSTOM_PROPERTY_NE_MODELS;
    public static HashMap<String, String> CUSTOM_PROPERTY_SENTIMENT_MODELS;

    // Named Entities: mapping from labels to identifiers that are used in the graph
    //public static HashMap<String, String> PROPERTY_NE_IDS = new HashMap<String, String>();

    // Named Entities: objects
    public HashMap<String, NameFinderME> nameDetectors;

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

    // Sentiment Analysis: objects
    private DocumentCategorizerME sentimentDetector;
    public HashMap<String, DocumentCategorizerME> sentimentDetectors;


    public OpenNLPPipeline(Properties properties) {
        this.globalProject = DEFAULT_PROJECT_VALUE;

        // Named Entities: mapping from labels to models
        PROPERTY_NE_MODELS = new HashMap<String, String>();
        PROPERTY_NE_MODELS.put("namefinder", "en-ner-person.bin");
        PROPERTY_NE_MODELS.put("datefinder", "en-ner-date.bin");
        PROPERTY_NE_MODELS.put("locationfinder", "en-ner-location.bin");
        PROPERTY_NE_MODELS.put("timefinder", "en-ner-time.bin");
        PROPERTY_NE_MODELS.put("organizationfinder", "en-ner-organization.bin");
        PROPERTY_NE_MODELS.put("moneyfinder", "en-ner-money.bin");
        PROPERTY_NE_MODELS.put("percentagefinder", "en-ner-percentage.bin");

        CUSTOM_PROPERTY_NE_MODELS = new HashMap<String, String>();
        CUSTOM_PROPERTY_SENTIMENT_MODELS = new HashMap<String, String>();
        findModelFiles("import/");

        nameDetectors = new HashMap<String, NameFinderME>();
        sentimentDetectors = new HashMap<String, DocumentCategorizerME>();

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
        // Default NE models
        for (String key : PROPERTY_NE_MODELS.keySet()) {
          InputStream is = getInputStream(properties, key, PROPERTY_NE_MODELS.get(key));
          if (is==null) continue;
          TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
          closeInputStream(is, key);
          nameDetectors.put(key, new NameFinderME(nameModel));
        }

        // Custom NE models (in the `import/` dir of the Neo4j installation)
        for (String key : CUSTOM_PROPERTY_NE_MODELS.keySet()) {
          InputStream is = new FileInputStream(new File(CUSTOM_PROPERTY_NE_MODELS.get(key)));
          if (is==null) continue;
          TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
          closeInputStream(is, key);
          nameDetectors.put(key, new NameFinderME(nameModel));
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
        // first a default model
        InputStream is = getInputStream(properties, PROPERTY_PATH_SENTIMENT_MODEL, PROPERTY_DEFAULT_SENTIMENT_MODEL);
        if (is!=null) {
          DoccatModel doccatModel = loadModel(DoccatModel.class, is);
          closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
          sentimentDetectors.put(DEFAULT_PROJECT_VALUE, new DocumentCategorizerME(doccatModel));
        } else
          sentimentDetectors.put(DEFAULT_PROJECT_VALUE, null);
        sentimentDetector = sentimentDetectors.get(DEFAULT_PROJECT_VALUE);

        // next custom models (in the `import/` dir of the Neo4j installation)
        for (String key : CUSTOM_PROPERTY_SENTIMENT_MODELS.keySet()) {
          try {
            is = new FileInputStream(new File(CUSTOM_PROPERTY_SENTIMENT_MODELS.get(key)));
          } catch (IOException ex) {
            LOG.error("Error while opening file " + CUSTOM_PROPERTY_SENTIMENT_MODELS.get(key));
            ex.printStackTrace();
          }
          if (is==null) continue;

          DoccatModel doccatModel = loadModel(DoccatModel.class, is);
          closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
          sentimentDetectors.put(key, new DocumentCategorizerME(doccatModel));
        }
    }

    public void annotate(OpenNLPAnnotation document) {
        String text = document.getText();
        try {
            Span sentences[] = sentenceDetector.sentPosDetect(text);
            document.setSentences(sentences);

            document.getSentences().stream().forEach(sentence -> {
                if (annotators.contains("tokenize") && wordBreaker!=null) {
                  // Tokenization
                  Span[] word_spans = wordBreaker.tokenizePos(sentence.getSentence());

                  if (!annotators.contains("stopword"))
                    sentence.setWordsAndSpans(word_spans);
                  else {
                    String[] words = Span.spansToStrings(word_spans, sentence.getSentence());
                    ArrayList<Span> fin_spans = new ArrayList<Span>();
                    ArrayList<String> fin_words = new ArrayList<String>();
                    for (int i=0; i<words.length; i++) {
                      if (stopWords.contains(words[i].toLowerCase()))
                        continue;
                      fin_words.add(words[i]);
                      fin_spans.add(word_spans[i]);
                    }
                    sentence.setWords(fin_words.toArray(new String[fin_words.size()]));
                    sentence.setWordSpans(fin_spans.toArray(new Span[fin_spans.size()]));
                  }
                  LOG.debug("Final words: " + Arrays.toString(sentence.getWords()));

                  if (annotators.contains("pos") && posme!=null) {
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
                    //else
                    //  sentence.setDefaultLemmas();

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
                    //else
                    //  sentence.setDefaultChunks();
                  } // pos
                  /*else {
                    sentence.setDefaultPosTags();
                    sentence.setDefaultLemmas();
                    sentence.setDefaultChunks();
                  }*/

                  if (annotators.contains("ner")) {
                    // Named Entities identification; needs to be performed after lemmas and POS (see implementation of Sentence.setNamedEntities())
                    for (String key : PROPERTY_NE_MODELS.keySet()) {
                      if (!nameDetectors.containsKey(key)) {
                        LOG.warn("NER model with key " + key + " not available.");
                        continue;
                      }
                      List ners = Arrays.asList(nameDetectors.get(key).find(sentence.getWords()));
                      sentence.setNamedEntities(ners);
                      /*
                      Arrays.asList(nameDetectors.get(key).find(sentence.getWords())).stream()
                            .forEach(span -> {
                                sentence.setNamedEntity(span.getStart(), span.getEnd(), span.getType());
                                LOG.debug("NER type: " + span.getType());
                            });*/
                    }
                    if (!this.globalProject.equals(DEFAULT_PROJECT_VALUE)) {
                      for (String key : CUSTOM_PROPERTY_NE_MODELS.keySet()) {
                        if (!nameDetectors.containsKey(key)) {
                          LOG.warn("Custom NER model with key " + key + " not available.");
                          continue;
                        }
                        if (key.split("-").length==0) continue;
                        if (!key.split("-")[0].equals(this.globalProject)) continue;
                        List ners = Arrays.asList(nameDetectors.get(key).find(sentence.getWords()));
                        sentence.setNamedEntities(ners);
                        /*Arrays.asList(nameDetectors.get(key).find(sentence.getWords())).stream()
                              .forEach(span -> {
                                  sentence.setNamedEntity(span.getStart(), span.getEnd(), span.getType());
                                  LOG.info("Custom NER type: " + span.getType());
                              });*/
                      }
                    }
                    sentence.finalizeNamedEntities();
                  } // ner
                  //else
                  //  sentence.setDefaultNamedEntities();
                } // tokenize

                if (annotators.contains("sentiment") && sentimentDetector!=null) {
                  double[] outcomes = sentimentDetector.categorize(sentence.getSentence());
                  String category = sentimentDetector.getBestCategory(outcomes);
                  if (Arrays.stream(outcomes).max().getAsDouble()<0.7)
                    category = "2"; // not conclusive, setting it to "Neutral"
                  sentence.setSentiment(category);
                  LOG.info("Sentiment results: sentence = " + sentence.getSentence() + "; category = " + category + "; outcomes = " + Arrays.toString(outcomes));
                }
            });
        } catch (Exception ex) {
            LOG.error("Error processing sentence for phrase: " + text, ex);
            throw new RuntimeException("Error processing sentence for phrase: " + text, ex);
        }
    }

    public String train(String project, String alg, String model_str, String fileTrain, String lang, Map<String, String> params) {
      String proj = project.toLowerCase();
      String fileOut = createModelFileName(lang, alg, model_str, proj);
      String newKey = proj + "-" + model_str;
      //LOG.info(proj + " - " + alg + "(" + alg.toLowerCase() + ") - " + model_str + " - " + fileTrain);
      String result = "";

      if (alg.toLowerCase().equals("ner")) {
        NERModelTool nerModel = new NERModelTool(fileTrain, model_str, lang, params);
        nerModel.train();
        result = nerModel.validate();
        nerModel.saveModel(fileOut);

        // incorporate this model to the OpenNLPPipeline
        if (nerModel.getModel()!=null) {
          CUSTOM_PROPERTY_NE_MODELS.put(newKey, fileOut);
          if (!nameDetectors.containsKey(newKey)) {
            nameDetectors.put(newKey, new NameFinderME((TokenNameFinderModel)nerModel.getModel()));
          }
        }

        nerModel.close();
      }

      else if (alg.toLowerCase().equals("sentiment")) {
        SentimentModelTool sentModel = new SentimentModelTool(fileTrain, model_str, lang, params);
        sentModel.train();
        result = sentModel.validate();
        sentModel.saveModel(fileOut);

        // incorporate this model to the OpenNLPPipeline
        if (sentModel.getModel()!=null) {
          CUSTOM_PROPERTY_SENTIMENT_MODELS.put(proj, fileOut);
          sentimentDetectors.put(proj, new DocumentCategorizerME((DoccatModel)sentModel.getModel()));
        }

        sentModel.close();
      }

      else {
        throw new UnsupportedOperationException("Undefined training procedure for algorithm " + alg);
      }

      return result;
    }

    public void reset() {
      updateProjectValue(DEFAULT_PROJECT_VALUE);
    }

    public void useTheseCustomModels(String project) {
      if (project==null) {
        //updateProjectValue(DEFAULT_PROJECT_VALUE);
        return;
      } else if (project.length()==0) {
        //updateProjectValue(DEFAULT_PROJECT_VALUE);
        return;
      } else {
        updateProjectValue(project);
      }
    }

    private void updateProjectValue(String project) {
      this.globalProject = project.toLowerCase();

      if (sentimentDetectors.containsKey(this.globalProject)) {
        LOG.info("Switching to a sentiment model: " + this.globalProject);
      } else {
        LOG.warn("Required sentiment model (" + this.globalProject + ") doesn't exist, setting it to the default.");
        this.globalProject = DEFAULT_PROJECT_VALUE;
      }
      sentimentDetector = sentimentDetectors.get(this.globalProject);
    }

    private void findModelFiles(String path) {
      if (path==null) return;
      if (path.length()==0) {
        LOG.warn("Scanning for model files: wrong path specified.");
        return;
      }
      File folder = new File(path);
      File[] listOfFiles = folder.listFiles();
      if (listOfFiles==null)
        return;

      String p = path;
      if (p.charAt(p.length()-1) != "/".charAt(0)) path += "/";
      LOG.debug("path = " + path);

      for (int i=0; i<listOfFiles.length; i++) {
        if (!listOfFiles[i].isFile())
          continue;
        String name = listOfFiles[i].getName();
        String[] sp = name.split("-");
        if (sp.length<3) continue;
        if (!name.substring(name.length()-4).equals(".bin")) continue;
        LOG.info("Custom models: Found file " + name);

        String key = sp[sp.length-1].toLowerCase();
        key = key.substring(0, key.length()-4);
        for (int j=2; j<sp.length-1; j++)
          key += "-" + sp[2];

        LOG.debug("Scanning for model files: registering model name for algorithm " + sp[1] + " under the key " + key);
        if (sp[1].toLowerCase().equals("ner"))
          CUSTOM_PROPERTY_NE_MODELS.put(key, path + name);
        else if (sp[1].toLowerCase().equals("sentiment"))
          CUSTOM_PROPERTY_SENTIMENT_MODELS.put(/*key*/sp[sp.length-1].toLowerCase().substring(0, sp[sp.length-1].length()-4), path + name);
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

    private void saveModel(BaseModel model, String file) {
      if (model==null) {
        LOG.error("Can't save training results to a " + file + ": model is null");
        return;
      }
      BufferedOutputStream modelOut = null;
      try {
        modelOut = new BufferedOutputStream(new FileOutputStream(file));
        model.serialize(modelOut);
        modelOut.close();
      } catch (IOException ex) {
        LOG.error("Error saving model to file " + file, ex);
        throw new RuntimeException("Error saving model to file " + file, ex);
      }
      return;
    }

    private InputStream getInputStream(Properties properties, String property, String defaultValue) {
        String path = defaultValue;
        if (properties!=null) path = properties.getProperty(property, defaultValue);
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

    private String createModelFileName(String lang, String alg, String model, String project) {
      String delim = "-";
      String name = "import/" + lang.toLowerCase() + delim + alg.toLowerCase();
      if (model!=null) {
        if (model.length()>0) name += delim + model.toLowerCase();
      }
      name += delim + project.toLowerCase() + ".bin"; 
      return name;
    }


    /*class ImprovisedInputStreamFactory implements InputStreamFactory {
      private File inputSourceFile;
      private String inputSourceStr;

      ImprovisedInputStreamFactory(Properties properties, String property, String defaultValue) {
        this.inputSourceFile = null;
        this.inputSourceStr = defaultValue;
        if (properties!=null) this.inputSourceStr = properties.getProperty(property, defaultValue);

        try {
          if (this.inputSourceStr.startsWith("file://"))
            this.inputSourceFile = new File(new URI(this.inputSourceStr));
          else if (this.inputSourceStr.startsWith("/"))
            this.inputSourceFile = new File(this.inputSourceStr);
        } catch (Exception ex) {
          LOG.error("Error while loading model from " + this.inputSourceStr);
          throw new RuntimeException("Error while loading model from " + this.inputSourceStr);
        }
      }

      @Override
      public InputStream createInputStream() throws IOException {
        LOG.debug("Creating input stream from " + this.inputSourceFile.getPath());
        //return getClass().getClassLoader().getResourceAsStream(this.inputSourceFile.getPath());
        return new FileInputStream(this.inputSourceFile.getPath());
      }
    }*/

}
