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
import java.lang.reflect.Constructor;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Properties;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
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

    public static final String PROPERTY_DEFAULT_CHUNKER_MODEL = "en-chunker.bin";
    public static final String PROPERTY_DEFAULT_POS_TAGGER_MODEL = "en-pos-maxent.bin";
    public static final String PROPERTY_DEFAULT_SENTENCE_MODEL = "en-sent.bin";
    public static final String PROPERTY_DEFAULT_TOKENIZER_MODEL = "en-token.bin";

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    private TokenizerME wordBreaker;
    private POSTaggerME posme;
    private ChunkerME chunkerME;
    private SentenceDetectorME sentenceDetector;

    public OpenNLPPipeline(Properties properties) {
        init(properties);
    }

    private void init(Properties properties) {

        try {
            senteceSplitter(properties);
            tokenizer(properties);
            posTagger(properties);
            chuncker(properties);

        } catch (IOException e) {
            LOG.error("Could not initialize OpenNLP models: " + e.getMessage());
            throw new RuntimeException("Could not initialize OpenNLP models", e);
        }
    }

    protected void chuncker(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_CHUNKER_MODEL, PROPERTY_DEFAULT_CHUNKER_MODEL);
        ChunkerModel chunkerModel = loadModel(ChunkerModel.class, is);
        chunkerME = new ChunkerME(chunkerModel);
    }

    private void posTagger(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_POS_TAGGER_MODEL, PROPERTY_DEFAULT_POS_TAGGER_MODEL);
        POSModel pm = loadModel(POSModel.class, is);
        posme = new POSTaggerME(pm);
    }

    private void tokenizer(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_TOKENIZER_MODEL, PROPERTY_DEFAULT_TOKENIZER_MODEL);
        TokenizerModel tm = loadModel(TokenizerModel.class, is);
        wordBreaker = new TokenizerME(tm);
    }

    private void senteceSplitter(Properties properties) throws FileNotFoundException {
        InputStream is = getInputStream(properties, PROPERTY_PATH_SENTENCE_MODEL, PROPERTY_DEFAULT_SENTENCE_MODEL);
        SentenceModel sentenceModel = loadModel(SentenceModel.class, is);
        sentenceDetector = new SentenceDetectorME(sentenceModel);
    }

    public void annotate(OpenNLPAnnotation document) {

        String text = document.getText();
        try {
            Span sentences[] = sentenceDetector.sentPosDetect(text);
            document.setSentences(sentences);

            document.getSentences().stream().forEach((sentence) -> {
                String[] words = wordBreaker.tokenize(sentence.getSentence());
                sentence.setWords(words);
                String[] posTags = posme.tag(words);
                sentence.setPosTags(posTags);
                Span[] chunks = chunkerME.chunkAsSpans(words, posTags);
                sentence.setChunks(chunks);
                String[] chunkStrings = Span.spansToStrings(chunks, words);
                sentence.setChunkStrings(chunkStrings);
                for (int i = 0; i < chunks.length; i++) {
                    //if (chunks[i].getType().equals("NP")) {
                        sentence.addPhraseIndex(i);
                    //}
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

}
