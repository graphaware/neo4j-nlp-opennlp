/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import com.graphaware.nlp.processor.opennlp.model.NERModelTool;
import com.graphaware.nlp.processor.opennlp.model.SentimentModelTool;
import com.graphaware.nlp.processor.AbstractTextProcessor;
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

    protected static final String DEFAULT_PROJECT_VALUE = "default";

    protected final List<String> annotators;
    protected final List<String> stopWords;

    protected TokenizerME wordBreaker;
    protected POSTaggerME posme;
    protected ChunkerME chunkerME;
    protected SentenceDetectorME sentenceDetector;
    protected DictionaryLemmatizer lemmaDetector; // needs OpenNLP >=1.7

    protected Map<String, String> customNeModels = new HashMap<>();
    protected Map<String, String> customSentimentModels = new HashMap<>();

    protected Map<String, NameFinderME> nameDetectors = new HashMap<>();
    //protected Map<String, DocumentCategorizerME> sentimentDetectors = new HashMap<>();
    protected DocumentCategorizerME sentimentDetector;

    protected static Map<String, String> BASIC_NE_MODEL;

    {
        BASIC_NE_MODEL = new HashMap<>();
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-person", "en-ner-person.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-date", "en-ner-date.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-location", "en-ner-location.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-time", "en-ner-time.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-organization", "en-ner-organization.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-money", "en-ner-money.bin");
        BASIC_NE_MODEL.put(DEFAULT_PROJECT_VALUE + "-percentage", "en-ner-percentage.bin");
    }

    public OpenNLPPipeline(Properties properties) {
        findModelFiles(IMPORT_DIRECTORY);
        this.annotators = Arrays.asList(properties.getProperty("annotators", "").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());
        this.stopWords = Arrays.asList(properties.getProperty("stopword", "").split(",")).stream().map(str -> str.trim().toLowerCase()).collect(Collectors.toList());
        init(properties);
    }

    private void init(Properties properties) {
        try {
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
        if (properties.containsKey("customNEs")) {
            List<String> requiredModels = Arrays.asList(properties.getProperty("customNEs").split(",")).stream().map(str -> str.trim()).collect(Collectors.toList());
            for (String key: requiredModels) {
                if (!customNeModels.containsKey(key)) {
                    LOG.error("Custom NE model " + key + " not found!");
                    throw new RuntimeException("Custom NE model " + key + " not found!");
                }
                LOG.info("Extracting custom NER model: " + key);
                InputStream is = new FileInputStream(new File(customNeModels.get(key)));
                TokenNameFinderModel nameModel = loadModel(TokenNameFinderModel.class, is);
                closeInputStream(is, key);
                nameDetectors.put(key, new NameFinderME(nameModel));
                LOG.info("Custom NER model " + key + " loaded for this pipeline.");
            }
        }
    }

    private void setLemmatizer(Properties properties) throws FileNotFoundException, IOException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_LEMMATIZER_MODEL, PROPERTY_DEFAULT_LEMMATIZER_MODEL);
        lemmaDetector = new DictionaryLemmatizer(is);
        closeInputStream(is, PROPERTY_PATH_LEMMATIZER_MODEL);
    }

    private void setCategorizer(Properties properties) throws FileNotFoundException {
        // Default sentiment model
        if (!properties.containsKey("customSentiment")) {
            InputStream is = getInputStream(properties, PROPERTY_PATH_SENTIMENT_MODEL, PROPERTY_DEFAULT_SENTIMENT_MODEL);
            if (is != null) {
                DoccatModel doccatModel = loadModel(DoccatModel.class, is);
                closeInputStream(is, PROPERTY_PATH_SENTIMENT_MODEL);
                //sentimentDetectors.put(DEFAULT_PROJECT_VALUE, new DocumentCategorizerME(doccatModel));
                sentimentDetector = new DocumentCategorizerME(doccatModel);
            } else {
                LOG.warn("No default sentiment detector available (input stream is null).");
                //sentimentDetectors.put(DEFAULT_PROJECT_VALUE, null);
                sentenceDetector = null;
            }
        }
        // Custom sentiment model (currently only one is possible)
        else {
            String customModel = properties.getProperty("customSentiment");
            LOG.info("Extracting custom sentiment model: " + customModel);
            if (!customSentimentModels.containsKey(customModel)) {
                LOG.error("Custom sentiment model " + customModel + " not found!");
                throw new RuntimeException("Custom sentiment model " + customModel + " not found!");
            }
            try {
                InputStream is = new FileInputStream(new File(customSentimentModels.get(customModel)));
                if (is == null) {
                    LOG.error("Custom sentiment model: input stream is null");
                    return;
                }
                DoccatModel doccatModel = loadModel(DoccatModel.class, is);
                closeInputStream(is, customSentimentModels.get(customModel));
                //sentimentDetectors.put(customModel, new DocumentCategorizerME(doccatModel));
                sentimentDetector = new DocumentCategorizerME(doccatModel);
                LOG.info("Custom sentiment model " + customModel + " loaded for this pipeline.");
            } catch (IOException ex) {
                LOG.error("Error while opening file " + customSentimentModels.get(customModel), ex);
            }
        }
    }

    public void annotate(OpenNLPAnnotation document) {
        String text = document.getText();
        try {
            Span sentences[] = sentenceDetector.sentPosDetect(text);
            document.setSentences(sentences);
            document.getSentences().stream()
                    .forEach((OpenNLPAnnotation.Sentence sentence) -> {
                        if (annotators.contains("tokenize") && wordBreaker != null) {
                            Span[] wordSpans = wordBreaker.tokenizePos(sentence.getSentence());
                            if (wordSpans != null && wordSpans.length > 0) {
                                sentence.setWordsAndSpans(wordSpans);

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
                                if (annotators.contains("ner") && sentence.getWords() != null) {

                                    // Named Entities identification; needs to be performed after lemmas and POS (see implementation of Sentence.addNamedEntities())
                                    BASIC_NE_MODEL.keySet().stream().forEach((modelKey) -> {
                                        if (!nameDetectors.containsKey(modelKey)) {
                                            LOG.warn("NER model with key " + modelKey + " not available.");
                                        } else {
                                            List<Span> ners = Arrays.asList(nameDetectors.get(modelKey).find(sentence.getWords()));
                                            addNer(ners, nerOccurrences);
                                        }
                                    });

                                    if (!customNeModels.isEmpty()) {
                                        for (String key : customNeModels.keySet()) {
                                            if (!nameDetectors.containsKey(key)) {
                                                LOG.warn("Custom NER model with key " + key + " not available.");
                                                continue;
                                            }
                                            if (key.split("-").length == 0) {
                                                continue;
                                            }
                                            LOG.info("Running custom NER: " + key);
                                            List ners = Arrays.asList(nameDetectors.get(key).find(sentence.getWords()));
                                            addNer(ners, nerOccurrences);
                                        }
                                    }
                                }
                                processTokens(sentence, nerOccurrences);
                            }
                        }
                        if (sentence.getWords() != null && sentence.getWords().length > 0) {
                            if (annotators.contains("sentiment") && sentimentDetector != null) {
                                double[] outcomes = sentimentDetector.categorize(sentence.getWords());
                                String category = sentimentDetector.getBestCategory(outcomes);
                                if (Arrays.stream(outcomes).max().getAsDouble() < document.getSentimentProb()) {
                                    category = "2";
                                }
                                sentence.setSentiment(category);
                                LOG.info("Sentiment results: sentence = " + sentence.getSentence() + "; category = " + category + "; outcomes = " + Arrays.toString(outcomes));
                            }
                        }
                    });

//            if (annotators.contains("ner")) {
//                for (String key : BASIC_NE_MODEL.keySet()) {
//                    if (nameDetectors.containsKey(key)) {
//                        nameDetectors.get(key).clearAdaptiveData();
//                    }
//                }
//                if (customProject != null) {
//                    for (String key : customNeModels.keySet()) {
//                        if (nameDetectors.containsKey(key)) {
//                            nameDetectors.get(key).clearAdaptiveData();
//                        }
//                    }
//                }
//            }
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

    public String train(String alg, String modelId, String fileTrain, String lang, Map<String, Object> params) {
        String fileOut = createModelFileName(lang, alg, modelId);
        String newKey = /*lang.toLowerCase() + "-" +*/ modelId.toLowerCase();
        String result = "";

        if (alg.toLowerCase().equals("ner")) {
            NERModelTool nerModel = new NERModelTool(fileTrain, modelId, lang, params);
            nerModel.train();
            result = nerModel.validate();
            nerModel.saveModel(fileOut);
            // incorporate this model to the OpenNLPPipeline
            if (nerModel.getModel() != null) {
                customNeModels.put(newKey, fileOut);
                /*if (!nameDetectors.containsKey(newKey)) {
                    nameDetectors.put(newKey, new NameFinderME((TokenNameFinderModel) nerModel.getModel()));
                }*/
            }
        }
        else if (alg.toLowerCase().equals("sentiment")) {
            SentimentModelTool sentModel = new SentimentModelTool(fileTrain, modelId, lang, params);
            sentModel.train();
            result = sentModel.validate();
            sentModel.saveModel(fileOut);
            // incorporate this model to the OpenNLPPipeline
            if (sentModel.getModel() != null) {
                customSentimentModels.put(newKey, fileOut);
                //sentimentDetectors.put(newKey, new DocumentCategorizerME((DoccatModel) sentModel.getModel()));
            }
        } else {
            throw new UnsupportedOperationException("Undefined training procedure for algorithm " + alg);
        }

        return result;
    }

    public String test(String alg, String modelId, String file, String lang) {
        String modelID = /*lang.toLowerCase() + "-" +*/ modelId.toLowerCase();
        String result = "failure";

        if (alg.toLowerCase().equals("ner")) {
          if (customNeModels.containsKey(modelID)) {
            LOG.info("Testing NER model: " + modelID);

            TokenNameFinderModel nameModel;
            try {
                // Load model
                InputStream is = new FileInputStream(new File(customNeModels.get(modelID)));
                nameModel = loadModel(TokenNameFinderModel.class, is);
                closeInputStream(is, modelID);
            } catch (Exception e) {
                throw new RuntimeException("Loading custom sentiment model " + modelID + " failed: ", e);
            }

            NERModelTool nerModel = new NERModelTool();
            result = nerModel.test(file, new NameFinderME(nameModel));
          } else
            LOG.error("Required NER model doesn't exist: " + modelID);
        }
        else if (alg.toLowerCase().equals("sentiment")) {
          if (customSentimentModels.containsKey(modelID)) {
            LOG.info("Testing sentiment model: " + modelID);

            DoccatModel doccatModel;
            try {
                // Load model
                InputStream is = new FileInputStream(new File(customSentimentModels.get(modelID)));
                doccatModel = loadModel(DoccatModel.class, is);
                closeInputStream(is, customSentimentModels.get(modelID));
            } catch (Exception e) {
                throw new RuntimeException("Loading custom sentiment model " + modelID + " failed: ", e);
            }

            SentimentModelTool sentModel = new SentimentModelTool();
            result = sentModel.test(file, new DocumentCategorizerME(doccatModel));
          } else
            LOG.error("Required sentiment model doesn't exist: " + modelID);
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
                    String type = ne.getType().toUpperCase();
                    Set<String> posSet = new HashSet<>();
                    int endSpan = startSpan;
                    for (int j = ne.getStart(); j < ne.getEnd(); j++) {
                        value += " " + words[j].trim();
                        lemma += " " + (lemmas[j].equals(DEFAULT_LEMMA_OPEN_NLP) ? words[j].toLowerCase().trim() : lemmas[j].trim());
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
                String value = words[i].trim();
                String lemma = lemmas[i].equals(DEFAULT_LEMMA_OPEN_NLP) ? words[i].toLowerCase() : lemmas[i].trim();
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

    private void findModelFiles(String path) {
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

        for (int i = 0; i < listOfFiles.length; i++) {
            if (!listOfFiles[i].isFile()) {
                continue;
            }
            String name = listOfFiles[i].getName();
            String[] sp = name.split("-");
            if (sp.length < 2) {
                continue;
            }
            if (!name.substring(name.length() - 4).equals(".bin")) {
                continue;
            }
            LOG.info("Custom models: Found file " + name);

            String alg  = sp[0].toLowerCase();

            String modelId = sp[1];
            // this is useful in case user-defined model ID contained symbol "-"
            for (int j = 2; j < sp.length; j++)
                modelId += "-" + sp[j];
            modelId = modelId.substring(0, modelId.length() - 4).toLowerCase(); // remove ".bin"
            //modelId = lang + "-" + modelId;

            LOG.info("Registering model name for algorithm " + alg + " under the key " + modelId);
            if (alg.equals("ner")) {
                customNeModels.put(modelId, path + name);
            } else if (alg.equals("sentiment")) {
                customSentimentModels.put(modelId, path + name);
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

    private String createModelFileName(String lang, String alg, String model) {
        String delim = "-";
        //String name = "import/" + lang.toLowerCase() + delim + alg.toLowerCase();
        String name = "import/" + alg.toLowerCase();
        if (model != null) {
            if (model.length() > 0) {
                name += delim + model.toLowerCase();
            }
        }
        name += ".bin";
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

    public Properties getProperties() {
        return new Properties();//to be implemented
    }
}
