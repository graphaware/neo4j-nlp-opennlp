package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.NLPIntegrationTest;
import org.junit.Test;

import static org.junit.Assert.*;

public class CustomSentimentModelIntegrationTest extends NLPIntegrationTest {

    @Test
    public void testTrainCustomModelWithProcedure() {
        String p = getClass().getClassLoader().getResource("import/sentiment_tweets.train").getPath();
        String q = "CALL ga.nlp.processor.train({textProcessor: \"com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor\", modelIdentifier: \"component-en\", alg: \"sentiment\", inputFile: \""+p+"\" , lang: \"en\"})";
        executeInTransaction(q, (result -> {
            assertTrue(result.hasNext());
        }));
    }

}
