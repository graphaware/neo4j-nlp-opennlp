package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.processor.opennlp.OpenNLPIntegrationTest;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class CustomNERIntegrationTest extends OpenNLPIntegrationTest {

    @Test
    public void testTrainNER() {
        String p = getClass().getClassLoader().getResource("import/ner.train").getPath();
        String q = "CALL ga.nlp.processor.train({textProcessor: \"com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor\", modelIdentifier: \"test-ner\", alg: \"ner\", inputFile: \"" + p + "\", trainingParameters: {iter: 10}})";
        executeInTransaction(q, (result -> {
            assertTrue(result.hasNext());
        }));

        String addPipelineQuery = "CALL ga.nlp.processor.addPipeline({textProcessor: 'com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor', name: 'customNER', processingSteps: {tokenize: true, ner: true, sentiment: true, dependency: false, customNER: \"test-ner\"}})";
        executeInTransaction(addPipelineQuery, emptyConsumer());


        String textNew = "Mrs Kus said she intended to form a government.";
        executeInTransaction("CALL ga.nlp.annotate({text: {text}, id:'test-ner', pipeline:'customNER', textProcessor: 'com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor'}) YIELD result RETURN result", Collections.singletonMap("text", textNew), result -> {
            assertTrue(result.hasNext());
        });

        executeInTransaction("MATCH (n:NER_Person) RETURN n.value AS v", (result -> {
            assertTrue(result.hasNext());
        }));
    }

    @Test
    public void testTrainOnNonExistingFileThrowsMeaningfulError() {
        String p = "non-exist";
        String q = "CALL ga.nlp.processor.train({textProcessor: \"com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor\", modelIdentifier: \"test-ner\", alg: \"ner\", inputFile: \"" + p + "\", trainingParameters: {iter: 10}})";
        Exception ex = null;
        try {
            executeInTransaction(q, (result -> {
                assertTrue(result.hasNext());
            }));
        } catch (Exception e) {
            ex = e;
        }
        assertNotNull(ex);
        assertTrue(ex.getMessage().contains("Unable to open"));
    }

    @Test
    public void testTrainingOnSmallSetThrowsInsufficientException() {
        String p = getClass().getClassLoader().getResource("import/ner-small.train").getPath();
        String q = "CALL ga.nlp.processor.train({textProcessor: \"com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor\", modelIdentifier: \"test-ner\", alg: \"ner\", inputFile: \"" + p + "\", trainingParameters: {iter: 10}})";
        Exception ex = null;
        try {
            executeInTransaction(q, (result -> {
                assertTrue(result.hasNext());
            }));
        } catch (Exception e) {
            ex = e;
        }
        assertNotNull(ex);
        assertTrue(ex.getMessage().contains("Insufficient"));
    }
}
