/*
 * Copyright (c) 2013-2016 GraphAware
 *
 * This file is part of the GraphAware Framework.
 *
 * GraphAware Framework is free software: you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details. You should have received a copy of
 * the GNU General Public License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
package com.graphaware.nlp.processor.opennlp;

import com.graphaware.nlp.domain.AnnotatedText;
import com.graphaware.nlp.domain.Sentence;
import com.graphaware.nlp.domain.Tag;
import com.graphaware.nlp.dsl.request.PipelineSpecification;
import com.graphaware.nlp.processor.AbstractTextProcessor;
import com.graphaware.nlp.processor.TextProcessor;
import com.graphaware.nlp.util.ServiceLoader;
import com.graphaware.nlp.util.TestAnnotatedText;
import com.graphaware.test.integration.EmbeddedDatabaseIntegrationTest;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.BeforeClass;
import org.junit.Test;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.QueryExecutionException;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;

import static com.graphaware.nlp.util.TagUtils.newTag;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;

public class TextProcessorTest extends OpenNLPIntegrationTest {

    private static TextProcessor textProcessor;
    private static final String TEXT_PROCESSOR = "com.graphaware.nlp.processor.opennlp.OpenNLPTextProcessor";
    private static PipelineSpecification PIPELINE_DEFAULT;

    @BeforeClass
    public static void init() {
        textProcessor = ServiceLoader.loadTextProcessor(TEXT_PROCESSOR);
        textProcessor.init();
        Map<String, Object> processingSteps = new HashMap<>();
        processingSteps.put(AbstractTextProcessor.STEP_TOKENIZE, true);
        processingSteps.put(AbstractTextProcessor.STEP_NER, true);
        PipelineSpecification pipelineSpecification = new PipelineSpecification("default", OpenNLPTextProcessor.class.getName(), processingSteps, null, 1L, Collections.emptyList(), Collections.emptyList());
        PIPELINE_DEFAULT = pipelineSpecification;
        textProcessor.createPipeline(PIPELINE_DEFAULT);
    }

    @Test
    public void testAnnotatedText() {
        AnnotatedText annotatedText = textProcessor.annotateText("On 8 May 2013, "
                + "one week before the Pakistani election, the third author, "
                + "in his keynote address at the Sentiment Analysis Symposium, "
                + "forecast the winner of the Pakistani election. The chart "
                + "in Figure 1 shows varying sentiment on the candidates for "
                + "prime minister of Pakistan in that election. The next day, "
                + "the BBC’s Owen Bennett Jones, reporting from Islamabad, wrote "
                + "an article titled “Pakistan Elections: Five Reasons Why the "
                + "Vote is Unpredictable,”1 in which he claimed that the election "
                + "was too close to call. It was not, and despite his being in Pakistan, "
                + "the outcome of the election was exactly as we predicted.", "en", PIPELINE_DEFAULT);

        TestAnnotatedText test = new TestAnnotatedText(annotatedText);
        test.assertSentencesCount(4);
        test.assertTagsCountInSentence(15, 0);
        test.assertTagsCountInSentence(11, 1);
        test.assertTagsCountInSentence(22, 2);//(24, 2); // it's 22 because `"Pakistan` & `"1` are not lemmatized by OpenNLP and checkLemmaIsValid() removes non-lemmatized version because of symbols `"`
        test.assertTagsCountInSentence(8, 3);//(9, 3); // it's 8 because OpenNLP has "be" among stopwords

        test.assertTag(newTag("pakistan", Collections.singletonList("LOCATION"), Collections.emptyList()));
        test.assertTag(newTag("show", Collections.emptyList(), Collections.singletonList("VBZ")));

    }

    @Test
    public void testLemmaLowerCasing() {
        AnnotatedText annotateText = textProcessor.annotateText(
                "Collibra’s Data Governance Innovation: Enabling Data as a Strategic Asset",
                "en", PIPELINE_DEFAULT);

        assertEquals(1, annotateText.getSentences().size());
        assertEquals("governance", annotateText.getSentences().get(0).getTagOccurrence(16).getLemma());
    }

    private void checkLocation(String location) throws QueryExecutionException {
        try (Transaction tx = getDatabase().beginTx()) {
            ResourceIterator<Object> rowIterator = getTagsIterator(location);
            Node pakistanNode = (Node) rowIterator.next();
            assertFalse(rowIterator.hasNext());
            String[] neList = (String[]) pakistanNode.getProperty("ne");
            assertEquals(neList[0], "location");
            tx.success();
        }
    }

    private void checkVerb(String verb) throws QueryExecutionException {
        try (Transaction tx = getDatabase().beginTx()) {
            ResourceIterator<Object> rowIterator = getTagsIterator(verb);
            Node pakistanNode = (Node) rowIterator.next();
            assertFalse(rowIterator.hasNext());
            String[] posL = (String[]) pakistanNode.getProperty("pos");
            assertEquals(posL[0], "VBZ");
            tx.success();
        }
    }

    private ResourceIterator<Object> getTagsIterator(String value) throws QueryExecutionException {
        Map<String, Object> params = new HashMap<>();
        params.put("value", value);
        Result pakistan = getDatabase().execute("MATCH (n:Tag {value: {value}}) return n", params);
        ResourceIterator<Object> rowIterator = pakistan.columnAs("n");
        return rowIterator;
    }

    @Test
    public void testAnnotatedTag() {
        Tag annotateTag = textProcessor.annotateTag("winners", "en", PIPELINE_DEFAULT);
        assertEquals(annotateTag.getLemma(), "winner");
    }

//    @Test
//    public void testAnnotationAndConcept() {
//        // ConceptNet5Importer.Builder() - arguments need fixing
//        /*TextProcessor textProcessor = ServiceLoader.loadTextProcessor("com.graphaware.nlp.processor.stanford.StanfordTextProcessor");
//        ConceptNet5Importer conceptnet5Importer = new ConceptNet5Importer.Builder("http://conceptnet5.media.mit.edu/data/5.4", textProcessor)
//                .build();
//        String text = "Say hi to Christophe";
//        AnnotatedText annotateText = textProcessor.annotateText(text, 1, 0, "en", false);
//        List<Node> nodes = new ArrayList<>();
//        try (Transaction beginTx = getDatabase().beginTx()) {
//            Node annotatedNode = annotateText.storeOnGraph(getDatabase(), false);
//            Map<String, Object> params = new HashMap<>();
//            params.put("id", annotatedNode.getId());
//            Result queryRes = getDatabase().execute("MATCH (n:AnnotatedText)-[*..2]->(t:Tag) where id(n) = {id} return t", params);
//            ResourceIterator<Node> tags = queryRes.columnAs("t");
//            while (tags.hasNext()) {
//                Node tag = tags.next();
//                nodes.add(tag);
//                List<Tag> conceptTags = conceptnet5Importer.importHierarchy(Tag.createTag(tag), "en");
//                conceptTags.stream().forEach((newTag) -> {
//                    nodes.add(newTag.storeOnGraph(getDatabase(), false));
//                });
//            }
//            beginTx.success();
//        }*/
//    }

    //@Test
    public void testSentiment() {
        AnnotatedText annotateText = textProcessor.annotateText(
                "I really hate to study at Stanford, "
                        + "it was a waste of time, I'll never be there again", "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
        assertEquals(0, annotateText.getSentences().get(0).getSentiment());

        annotateText = textProcessor.annotateText(
                "It was really horrible to study at Stanford", "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
        assertEquals(1, annotateText.getSentences().get(0).getSentiment());

        annotateText = textProcessor.annotateText("I studied at Stanford", "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
        assertEquals(2, annotateText.getSentences().get(0).getSentiment());

        annotateText = textProcessor.annotateText("I liked to study at Stanford", "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
        assertEquals(3, annotateText.getSentences().get(0).getSentiment());

        annotateText = textProcessor.annotateText(
                "I liked so much to study at Stanford, I enjoyed my time there, I would recommend every body",
                "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
        assertEquals(4, annotateText.getSentences().get(0).getSentiment());
    }

    @Test
    public void testAnnotatedTextWithPosition() {
        AnnotatedText annotateText = textProcessor.annotateText("On 8 May 2013, "
                + "one week before the Pakistani election, the third author, "
                + "in his keynote address at the Sentiment Analysis Symposium, "
                + "forecast the winner of the Pakistani election. The chart "
                + "in Figure 1 shows varying sentiment on the candidates for "
                + "prime minister of Pakistan in that election. The next day, "
                + "the BBC’s Owen Bennett Jones, reporting from Islamabad, wrote "
                + "an article titled “Pakistan Elections: Five Reasons Why the "
                + "Vote is Unpredictable,”1 in which he claimed that the election "
                + "was too close to call. It was not, and despite his being in Pakistan, "
                + "the outcome of the election was exactly as we predicted.", "en", PIPELINE_DEFAULT);

        assertEquals(4, annotateText.getSentences().size());
        Sentence sentence1 = annotateText.getSentences().get(0);
        assertEquals(15, sentence1.getTags().size());

        assertNull(sentence1.getTagOccurrence(0));
        assertEquals("8", sentence1.getTagOccurrence(3).getLemma());
        assertEquals("may 2013", sentence1.getTagOccurrence(5).getLemma());
        assertEquals("May 2013", sentence1.getTagOccurrences().get(5).get(0).getValue());
        assertEquals("one", sentence1.getTagOccurrence(15).getLemma());
        assertEquals("before", sentence1.getTagOccurrence(24).getLemma());
        assertEquals("third", sentence1.getTagOccurrence(59).getLemma());
        //assertEquals("sentiment analysis symposium", sentence1.getTagOccurrence(103).getLemma());
        assertEquals("forecast", sentence1.getTagOccurrence(133).getLemma());
        assertNull(sentence1.getTagOccurrence(184));

        Sentence sentence2 = annotateText.getSentences().get(1);
        assertEquals("show", sentence2.getTagOccurrence(22).getLemma());
        assertEquals("shows", sentence2.getTagOccurrences().get(22).get(0).getValue());
//        assertTrue(sentence1.getPhraseOccurrence(99).contains(new Phrase("the Sentiment Analysis Symposium")));
//        assertTrue(sentence1.getPhraseOccurrence(103).contains(new Phrase("Sentiment")));
//        assertTrue(sentence1.getPhraseOccurrence(113).contains(new Phrase("Analysis")));
//
//        //his(76)-> the third author(54)
//        assertTrue(sentence1.getPhraseOccurrence(55).get(1).getContent().equalsIgnoreCase("the third author"));
//        Sentence sentence2 = annotateText.getSentences().get(1);
//        assertEquals("chart", sentence2.getTagOccurrence(184).getLemma());
//        assertEquals("Figure", sentence2.getTagOccurrence(193).getLemma());
    }

    @Test
    public void testAnnotatedShortText() {
        AnnotatedText annotateText = textProcessor.annotateText(
                "Fixing Batch Endpoint Logging Problem", "en", PIPELINE_DEFAULT);

        assertEquals(1, annotateText.getSentences().size());
//
//        GraphPersistence peristence = new LocalGraphDatabase(getDatabase());
//        peristence.persistOnGraph(annotateText, false);

    }

    @Test
    public void testAnnotatedShortText2() {
        AnnotatedText annotateText = textProcessor.annotateText(
                "Importing CSV data does nothing", "en", PIPELINE_DEFAULT);
        assertEquals(1, annotateText.getSentences().size());
//        GraphPersistence peristence = new LocalGraphDatabase(getDatabase());
//        peristence.persistOnGraph(annotateText, false);
    }
}
