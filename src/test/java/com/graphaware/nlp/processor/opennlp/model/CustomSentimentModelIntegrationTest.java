package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.processor.opennlp.OpenNLPIntegrationTest;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

public class CustomSentimentModelIntegrationTest extends OpenNLPIntegrationTest {

    //@Test
    public void testTrainCustomModelWithProcedure() {
        String p = getClass().getClassLoader().getResource("import/sentiment_tweets.train").getPath();
        String q = "CALL ga.nlp.processor.train({textProcessor: \"com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor\", modelIdentifier: \"component-en\", alg: \"sentiment\", inputFile: \""+p+"\" , lang: \"en\"})";
        executeInTransaction(q, (result -> {
            assertTrue(result.hasNext());
        }));

        String addPipelineQuery = "CALL ga.nlp.processor.addPipeline({textProcessor: 'com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor', name: 'customSentiment', processingSteps: {tokenize: true, ner: true, sentiment: true, dependency: false, customSentiment: \"component-en\"}})";
        executeInTransaction(addPipelineQuery, emptyConsumer());

        String insertQ = "CREATE (tweet:Tweet) SET tweet.text = \"African American unemployment is the lowest ever recorded in our country. The Hispanic unemployment rate dropped a full point in the last year and is close to the lowest in recorded history. Dems did nothing for you but get your vote!\"\n" +
                "WITH tweet\n" +
                "CALL ga.nlp.annotate({text:tweet.text, id:id(tweet), pipeline:\"customSentiment\", checkLanguage:false})\n" +
                "YIELD result\n" +
                "MERGE (tweet)-[:HAS_ANNOTATED_TEXT]->(result)";
        executeInTransaction(insertQ, emptyConsumer());
        executeInTransaction("MATCH (n:Sentence) RETURN ANY(x IN labels(n) WHERE x IN ['Positive','Very Positive','Neutral']) AS hasSentiment", (result -> {
            assertTrue(result.hasNext());
            while (result.hasNext()) {
                Map<String, Object> record = result.next();
                assertTrue((boolean) record.get("hasSentiment"));
            }
        }));
    }

}
