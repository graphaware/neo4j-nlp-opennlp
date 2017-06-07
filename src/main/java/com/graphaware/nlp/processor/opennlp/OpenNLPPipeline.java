/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import com.graphaware.nlp.processor.opennlp.model.NERModelTool;
import com.graphaware.nlp.processor.opennlp.model.SentimentModelTool;
import static com.graphaware.nlp.processor.opennlp.OpenNLPAnnotation.DEFAULT_LEMMA_OPEN_NLP;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Properties;
import java.util.HashMap;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
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
import opennlp.tools.lemmatizer.DictionaryLemmatizer; // needs OpenNLP >=1.7
//import opennlp.tools.lemmatizer.SimpleLemmatizer;   // for OpenNLP < 1.7
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.util.Span;
import opennlp.tools.util.model.BaseModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenNLPPipeline {

    protected static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    public static final String DEFAULT_BACKGROUND_SYMBOL = "O";

    protected static final String IMPORT_DIRECTORY = "import/";

    protected static final String PROPERTY_PATH_CHUNKER_MODEL = "chuncker";
    protected static final String PROPERTY_PATH_POS_TAGGER_MODEL = "pos";
    protected static final String PROPERTY_PATH_SENTENCE_MODEL = "sentence";
    protected static final String PROPERTY_PATH_TOKENIZER_MODEL = "tokenizer";
    protected static final String PROPERTY_PATH_LEMMATIZER_MODEL = "lemmatizer";
    protected static final String PROPERTY_PATH_SENTIMENT_MODEL = "sentiment";

    protected static final String PROPERTY_DEFAULT_CHUNKER_MODEL = "en-chunker.bin";
    protected static final String PROPERTY_DEFAULT_POS_TAGGER_MODEL = "en-pos-maxent.bin";
    protected static final String PROPERTY_DEFAULT_SENTENCE_MODEL = "en-sent.bin";
    protected static final String PROPERTY_DEFAULT_TOKENIZER_MODEL = "en-token.bin";
    protected static final String PROPERTY_DEFAULT_LEMMATIZER_MODEL = "en-lemmatizer.dict";
    protected static final String PROPERTY_DEFAULT_SENTIMENT_MODEL = "en-sentiment-tweets_toy.bin";

    protected static final String PROPERTY_DEFAULT_SENTIMENT_TRAIN = "sentiment_tweets.train";
    protected static final String DEFAULT_PROJECT_VALUE = "default";

    protected Map<String, String> customNeModels = new HashMap<>();
    protected Map<String, String> customSentimentModels = new HashMap<>();

    protected Map<String, NameFinderME> nameDetectors = new HashMap<>();

    protected final List<String> annotators;
    protected final List<String> stopWords;

    protected TokenizerME wordBreaker;
    protected POSTaggerME posme;
    protected ChunkerME chunkerME;
    protected SentenceDetectorME sentenceDetector;
    protected DictionaryLemmatizer lemmaDetector; // needs OpenNLP >=1.7

    protected Map<String, DocumentCategorizerME> sentimentDetectors = new HashMap<>();

    protected static Map<String, String> BASIC_NE_MODEL;

    {
        BASIC_NE_MODEL = new HashMap<>();
        BASIC_NE_MODEL.put("namefinder", "en-ner-person.bin");
        BASIC_NE_MODEL.put("datefinder", "en-ner-date.bin");
        BASIC_NE_MODEL.put("locationfinder", "en-ner-location.bin");
        BASIC_NE_MODEL.put("timefinder", "en-ner-time.bin");
        BASIC_NE_MODEL.put("organizationfinder", "en-ner-organization.bin");
        BASIC_NE_MODEL.put("moneyfinder", "en-ner-money.bin");
        BASIC_NE_MODEL.put("percentagefinder", "en-ner-percentage.bin");
    }

    public OpenNLPPipeline(Properties properties) {
        this.annotators = Arrays.asList(properties.getProperty("annotators", "").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());
        this.stopWords = Arrays.asList(properties.getProperty("stopword", "").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());
        LOG.info("Annotators: " + annotators);
        LOG.info("Stop words: " + stopWords);
        init(properties);
    }

    private void init(Properties properties) {
        try {
            findAndLoadModelFiles(IMPORT_DIRECTORY);
            setSenteceSplitter(properties);
            setTokenizer(properties);
            setPosTagger(properties);
            setChuncker(properties);
            loadNamedEntitiesFinders(properties);
            setLemmatizer(properties);
            setCategorizer(properties);

        } catch (IOException e) {
            LOG.error("Could not initialize OpenNLP models: " + e.getMessage());
            throw new RuntimeException("Could not initialize OpenNLP models", e);
        }
    }

    private void setChuncker(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_CHUNKER_MODEL, PROPERTY_DEFAULT_CHUNKER_MODEL);
        ChunkerModel chunkerModel = loadModel(ChunkerModel.class, is);
        closeInputStream(is, PROPERTY_PATH_CHUNKER_MODEL);
        chunkerME = new ChunkerME(chunkerModel);
    }

    private void setPosTagger(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_POS_TAGGER_MODEL, PROPERTY_DEFAULT_POS_TAGGER_MODEL);
        POSModel pm = loadModel(POSModel.class, is);
        closeInputStream(is, PROPERTY_PATH_POS_TAGGER_MODEL);
        posme = new POSTaggerME(pm);
    }

    private void setTokenizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_TOKENIZER_MODEL, PROPERTY_DEFAULT_TOKENIZER_MODEL);
        TokenizerModel tm = loadModel(TokenizerModel.class, is);
        closeInputStream(is, PROPERTY_PATH_TOKENIZER_MODEL);
        wordBreaker = new TokenizerME(tm);
    }

    private void setSenteceSplitter(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_SENTENCE_MODEL, PROPERTY_DEFAULT_SENTENCE_MODEL);
        SentenceModel sentenceModel = loadModel(SentenceModel.class, is);
        closeInputStream(is, PROPERTY_PATH_SENTENCE_MODEL);
        sentenceDetector = new SentenceDetectorME(sentenceModel);
    }

    private void loadNamedEntitiesFinders(Properties properties) throws FileNotFoundException {
        // Default NE models
        BASIC_NE_MODEL.entrySet().stream().forEach((item) -> {
            InputStream is = getInputStream(properties, item.getKey(), item.getValue());
            if (!(is == null)) {
                TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
                closeInputStream(is, item.getKey());
                nameDetectors.put(item.getKey(), new NameFinderME(nameModel));
            }
        });

        // Custom NE models (in the `import/` dir of the Neo4j installation)
        for (String key : customNeModels.keySet()) {
            InputStream is = new FileInputStream(new File(customNeModels.get(key)));
            TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
            closeInputStream(is, key);
            nameDetectors.put(key, new NameFinderME(nameModel));
        }
    }

    private void setLemmatizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_LEMMATIZER_MODEL, PROPERTY_DEFAULT_LEMMATIZER_MODEL);
        lemmaDetector = new DictionaryLemmatizer(is); 
        closeInputStream(is, PROPERTY_PATH_LEMMATIZER_MODEL);
    }

    private void setCategorizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_SENTIMENT_MODEL, PROPERTY_DEFAULT_SENTIMENT_MODEL);
        if (is != null) {
            DoccatModel doccatModel = loadModel(DoccatModel.class, is);
            closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
            sentimentDetectors.put(DEFAULT_PROJECT_VALUE, new DocumentCategorizerME(doccatModel));
        } else {
            sentimentDetectors.put(DEFAULT_PROJECT_VALUE, null);
        }

        for (String key : customSentimentModels.keySet()) {
            try {
                is = new FileInputStream(new File(customSentimentModels.get(key)));
            } catch (IOException ex) {
                LOG.error("Error while opening file " + customSentimentModels.get(key), ex);
            }
            if (is == null) {
                continue;
            }

            DoccatModel doccatModel = loadModel(DoccatModel.class, is);
            closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
            sentimentDetectors.put(key, new DocumentCategorizerME(doccatModel));
        }
    }

    public void annotate(OpenNLPAnnotation document) {
        String text = document.getText();
        String customProject = document.getProject();
        LOG.info("Annotating text: " + text + " using stopwords: " + stopWords);
        try {
            Span sentences[] = sentenceDetector.sentPosDetect(text);
            document.setSentences(sentences);
            document.getSentences().stream().forEach((OpenNLPAnnotation.Sentence sentence) -> {
                if (annotators.contains("tokenize") && wordBreaker != null) {
                    Span[] wordSpans = wordBreaker.tokenizePos(sentence.getSentence());

                    sentence.setWordsAndSpans(wordSpans);
                    LOG.info("Final words: " + Arrays.toString(sentence.getWords()));
                    if (annotators.contains("pos") && posme != null) {
                        String[] posTags = posme.tag(sentence.getWords());
                        sentence.setPosTags(posTags);
                        if (annotators.contains("lemma")) {
                            String[] finLemmas = lemmaDetector.lemmatize(sentence.getWords(), posTags);
                            sentence.setLemmas(finLemmas);
                        }

                        //FIXME: this is wrong
//                        if (annotators.contains("relation")) {
//                            Span[] chunks = chunkerME.chunkAsSpans(sentence.getWords(), posTags);
//                            sentence.setChunks(chunks);
//                            LOG.info("Found " + chunks.length + " phrases.");
//                            String[] chunkStrings = Span.spansToStrings(chunks, sentence.getWords());
//                            sentence.setChunkStrings(chunkStrings);
//                            List<String> chunkSentiments = new ArrayList<>();
//                            for (int i = 0; i < chunks.length; i++) {
//                                sentence.addPhraseIndex(i);
//                            }
//                            if (!chunkSentiments.isEmpty()) {
//                                sentence.setChunkSentiments(chunkSentiments.toArray(new String[chunkSentiments.size()]));
//                            }
//                        } 
                    }
                    
                    Map<Integer, List<Span>> nerOccurrences = new HashMap<>();
                    if (annotators.contains("ner")) {

                        // Named Entities identification; needs to be performed after lemmas and POS (see implementation of Sentence.addNamedEntities())
                        BASIC_NE_MODEL.keySet().stream().forEach((modelKey) -> {
                            if (!nameDetectors.containsKey(modelKey)) {
                                LOG.warn("NER model with key " + modelKey + " not available.");
                            } else {
                                List<Span> ners = Arrays.asList(nameDetectors.get(modelKey).find(sentence.getWords()));
                                addNer(ners, nerOccurrences);
                            }
                        });

                        if (customProject != null) {
                            for (String key : customNeModels.keySet()) {
                                if (!nameDetectors.containsKey(key)) {
                                    LOG.warn("Custom NER model with key " + key + " not available.");
                                    continue;
                                }
                                if (key.split("-").length == 0) {
                                    continue;
                                }
                                if (!key.split("-")[0].equals(customProject)) {
                                    continue;
                                }
                                LOG.info("Running custom NER for project " + customProject + ": " + key);
                                List ners = Arrays.asList(nameDetectors.get(key).find(sentence.getWords()));
                                addNer(ners, nerOccurrences);
                            }
                        }
                    }
                    processTokens(sentence, nerOccurrences);
                }
                String sentimentDetectorName = customProject != null ? customProject : DEFAULT_PROJECT_VALUE;
                DocumentCategorizerME sentimentDetector = sentimentDetectors.get(sentimentDetectorName);
                if (annotators.contains("sentiment") && sentimentDetector != null) {
                    double[] outcomes = sentimentDetector.categorize(sentence.getSentence());
                    String category = sentimentDetector.getBestCategory(outcomes);
                    if (Arrays.stream(outcomes).max().getAsDouble() < document.getSentimentProb()) {
                        category = "2";
                    }
                    sentence.setSentiment(category);
                    LOG.info("Sentiment results: sentence = " + sentence.getSentence() + "; category = " + category + "; outcomes = " + Arrays.toString(outcomes));
                }
            });

            if (annotators.contains("ner")) {
                for (String key : BASIC_NE_MODEL.keySet()) {
                    if (nameDetectors.containsKey(key)) {
                        nameDetectors.get(key).clearAdaptiveData();
                    }
                }
                if (customProject != null) {
                    for (String key : customNeModels.keySet()) {
                        if (nameDetectors.containsKey(key)) {
                            nameDetectors.get(key).clearAdaptiveData();
                        }
                    }
                }
            }
        } catch (Exception ex) {
            LOG.error("Error processing sentence for text: " + text, ex);
            throw new RuntimeException("Error processing sentence for text: " + text, ex);
        }
    }

    protected void addNer(List<Span> ners, Map<Integer, List<Span>> nerOccurrences) {
        if (ners != null && !ners.isEmpty()) {
            ners.stream().forEach((ner) -> {
                List<Span> currentNer = nerOccurrences.get(ner.getStart());
                if (currentNer == null) {
                    currentNer = new ArrayList<>();
                    nerOccurrences.put(ner.getStart(), currentNer);
                }
                currentNer.add(ner);
            });
        }
    }

    public String train(String project, String alg, String model_str, String fileTrain, String lang, Map<String, String> params) {
        String proj = project.toLowerCase();
        String fileOut = createModelFileName(lang, alg, model_str, proj);
        String newKey = proj + "-" + model_str;
        String result = "";

        if (alg.toLowerCase().equals("ner")) {
            NERModelTool nerModel = new NERModelTool(fileTrain, model_str, lang, params);
            nerModel.train();
            result = nerModel.validate();
            nerModel.saveModel(fileOut);
            // incorporate this model to the OpenNLPPipeline
            if (nerModel.getModel() != null) {
                customNeModels.put(newKey, fileOut);
                if (!nameDetectors.containsKey(newKey)) {
                    nameDetectors.put(newKey, new NameFinderME((TokenNameFinderModel) nerModel.getModel()));
                }
            }
        } else if (alg.toLowerCase().equals("sentiment")) {
            SentimentModelTool sentModel = new SentimentModelTool(fileTrain, model_str, lang, params);
            sentModel.train();
            result = sentModel.validate();
            sentModel.saveModel(fileOut);
            // incorporate this model to the OpenNLPPipeline
            if (sentModel.getModel() != null) {
                customSentimentModels.put(proj, fileOut);
                sentimentDetectors.put(proj, new DocumentCategorizerME((DoccatModel) sentModel.getModel()));
            }
        } else {
            throw new UnsupportedOperationException("Undefined training procedure for algorithm " + alg);
        }
        return result;
    }

    private void processTokens(OpenNLPAnnotation.Sentence sentence, Map<Integer, List<Span>> nerOccurrences) {
        if (sentence.getWords() == null) {
            return;
        }
        String[] words = sentence.getWords();
        String[] lemmas = sentence.getLemmas();
        String[] posTags = sentence.getPosTags();
        Span[] wordSpans = sentence.getWordSpans();

        for (int i = 0; i < words.length; i++) {
            if (nerOccurrences != null && nerOccurrences.containsKey(i)) {
                List<Span> ners = nerOccurrences.get(i);
                final int startSpan = wordSpans[i].getStart();
                AtomicInteger index = new AtomicInteger(i);
                ners.forEach(ne -> {
                    String value = "";
                    String lemma = "";
                    String type = ne.getType();
                    Set<String> posSet = new HashSet<>();
                    int endSpan = startSpan;
                    for (int j = ne.getStart(); j < ne.getEnd(); j++) {
                        value += " " + words[j];
                        lemma += " " + (lemmas[j].equals(DEFAULT_LEMMA_OPEN_NLP) ? words[j].toLowerCase() : lemmas[j]);
                        posSet.add(posTags[j]);
                        endSpan = wordSpans[j].getEnd();
                        if (index.get() < j) {
                            index.set(j);
                        }
                    }

                    value = value.trim();
                    lemma = lemma.trim();
                    //check stopwords
                    if (isNotStopWord(lemma)) {
                        OpenNLPAnnotation.Token token = sentence.getToken(value, lemma);
                        token.addTokenNE(type);
                        token.addTokenPOS(posSet);
                        token.addTokenSpans(new Span(startSpan, endSpan));
                    }
                });
                i = index.get();
            } else {
                String value = words[i];
                String lemma = lemmas[i].equals(DEFAULT_LEMMA_OPEN_NLP) ? words[i].toLowerCase() : lemmas[i];
                String ne = DEFAULT_BACKGROUND_SYMBOL;
                String pos = posTags[i];
                Set<String> posSet = new HashSet<>();
                if (isNotStopWord(lemma)) {
                    OpenNLPAnnotation.Token token = sentence.getToken(value, lemma);
                    token.addTokenNE(ne);
                    posSet.add(pos);
                    token.addTokenPOS(posSet);
                    token.addTokenSpans(wordSpans[i]);
                }
            }
        }
    }

    private boolean isNotStopWord(String value) {
        return !annotators.contains("stopword") || !stopWords.contains(value.toLowerCase());
    }

    private void findAndLoadModelFiles(String path) {
        if (path == null || path.length() == 0) {
            LOG.error("Scanning for model files: wrong path specified.");
            return;
        }
        
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();
        if (listOfFiles == null) {
            return;
        }

        String p = path;
        if (p.charAt(p.length() - 1) != "/".charAt(0)) {
            path += "/";
        }
        LOG.info("path = " + path);

        for (int i = 0; i < listOfFiles.length; i++) {
            if (!listOfFiles[i].isFile()) {
                continue;
            }
            String name = listOfFiles[i].getName();
            String[] sp = name.split("-");
            if (sp.length < 3) {
                continue;
            }
            if (!name.substring(name.length() - 4).equals(".bin")) {
                continue;
            }
            LOG.info("Custom models: Found file " + name);

            String key = sp[sp.length - 1].toLowerCase();
            key = key.substring(0, key.length() - 4);
            for (int j = 2; j < sp.length - 1; j++) {
                key += "-" + sp[2];
            }

            LOG.debug("Scanning for model files: registering model name for algorithm " + sp[1] + " under the key " + key);
            if (sp[1].toLowerCase().equals("ner")) {
                customNeModels.put(key, path + name);
            } else if (sp[1].toLowerCase().equals("sentiment")) {
                customSentimentModels.put(sp[sp.length - 1].toLowerCase().substring(0, sp[sp.length - 1].length() - 4), path + name);
            }
        }
    }

    private <T extends BaseModel> T loadModel(Class<T> clazz, InputStream in) {
        try {
            Constructor<T> modelConstructor = clazz.getConstructor(InputStream.class);
            T model = modelConstructor.newInstance(in);
            return model;
        } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
            LOG.error("Error while initializing model of class: " + clazz, ex);
            throw new RuntimeException("Error while initializing model of class: " + clazz, ex);
        }
    }

    private void saveModel(BaseModel model, String file) {
        if (model == null) {
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
        if (properties != null) {
            path = properties.getProperty(property, defaultValue);
        }
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

    private void closeInputStream(InputStream is, String name) {
        try {
            if (is != null) {
                is.close();
            }
        } catch (IOException ex) {
            LOG.warn("Attept to close stream for " + name + " model failed.");
        }
        return;
    }

    private String createModelFileName(String lang, String alg, String model, String project) {
        String delim = "-";
        String name = "import/" + lang.toLowerCase() + delim + alg.toLowerCase();
        if (model != null) {
            if (model.length() > 0) {
                name += delim + model.toLowerCase();
            }
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
