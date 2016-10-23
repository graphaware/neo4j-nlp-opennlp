//package com.graphaware.nlp.processor.opennlp;
//
//import com.graphaware.nlp.processor.opennlp.OpenNLPPhraseProcessor;
//import org.junit.Test;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.util.List;
//
///**
// * Created by michael kilgore on 10/2/16.
// * Begining test class for OpenNLP Phrases
// */
//public class TestOpenNLP
//{
//    private static final Logger LOG = LoggerFactory.getLogger(TestOpenNLP.class);
//
//    public TestOpenNLP()
//    {
//    }
//
//    @Test
//    public void testPhrase()
//    {
//        LOG.info("starting test");
//        String workingDir = System.getProperty("user.dir");
//        OpenNLPPhraseProcessor openNLP = new OpenNLPPhraseProcessor();
//        openNLP.init(workingDir+"/");
//        List<String> phrases = openNLP.processForPhrases("Barack Hussein Obama II  is the 44th and current President of the United States, and the first African American to hold the office.");
//        for (String phrase : phrases)
//            LOG.info(phrase);
//    }
//}
