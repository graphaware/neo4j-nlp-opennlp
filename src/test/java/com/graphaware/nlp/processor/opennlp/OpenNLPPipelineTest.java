/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author ale
 */
public class OpenNLPPipelineTest {

    public OpenNLPPipelineTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of annotate method, of class OpenNLPPipeline.
     */
    @Test
    public void testAnnotate() {
        String text = "Hello Dralyn. Barack Hussein Obama II  is the 44th and current President of the United States, and the first African American to hold the office.";
        OpenNLPAnnotation document = new OpenNLPAnnotation(text);
        OpenNLPPipeline instance = new PipelineBuilder()
                .tokenize()
                /*.extractPos()
                .extractRelations()*/
                .build();
        instance.annotate(document);

        document.getSentences().forEach((sentence) -> {
            System.out.println(">>>" + sentence.getSentence());
            if (sentence.getPhrasesIndex() != null) {
                sentence.getPhrasesIndex().forEach((phrase) -> {
                    System.out.println(">>>" + sentence.getChunkStrings()[phrase]);
                });
            }
        });
    }

    @Test
    public void testAnnotateNER() {
        String sentence = "Barack Hussein Obama II  is the 44th and current President of the United States, and the first African American to hold the office.";
        
        OpenNLPAnnotation document = new OpenNLPAnnotation(sentence);
        OpenNLPPipeline instance = new PipelineBuilder()
                .tokenize()
                /*.extractPos()
                .extractRelations()*/
                .build();
        instance.annotate(document);

        document.getSentences().forEach((item) -> {
            item.getTokens().stream().forEach((token) -> {
                System.out.println("" + token.getTokenPOS()+ " " + token.getToken() + " - " + token.getToken() + " " + token.getTokenNEs());
            });
        });

        InputStream modelInToken = null;
        InputStream modelIn = null;

        try {

            //1. convert sentence into tokens
            modelInToken = this.getClass().getResourceAsStream("en-token.bin");
            TokenizerModel modelToken = new TokenizerModel(modelInToken);
            Tokenizer tokenizer = new TokenizerME(modelToken);
            String tokens[] = tokenizer.tokenize(sentence);

            //2. find names
            modelIn = this.getClass().getResourceAsStream("en-ner-person.bin");
            TokenNameFinderModel model = new TokenNameFinderModel(modelIn);
            NameFinderME nameFinder = new NameFinderME(model);

            Span nameSpans[] = nameFinder.find(tokens);

            //find probabilities for names
            double[] spanProbs = nameFinder.probs(nameSpans);

            //3. print names
            for (int i = 0; i < nameSpans.length; i++) {
                System.out.println("Span: " + nameSpans[i].toString());
                System.out.println("Covered text is: " + tokens[nameSpans[i].getStart()] + " " + tokens[nameSpans[i].getStart() + 1]);
                System.out.println("Probability is: " + spanProbs[i]);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            try {
                if (modelInToken != null) {
                    modelInToken.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            };
            try {
                if (modelIn != null) {
                    modelIn.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            };
        }
    }

    @Test
    public void testStopWordsAnnotate() {
        String text = "Hello Dralyn. Barack Hussein Obama II  is the 44th and current President of the United States, and the first African American to hold the office.";
        OpenNLPAnnotation document = new OpenNLPAnnotation(text);
        OpenNLPPipeline instance = new PipelineBuilder()
                .tokenize()
                .customStopWordAnnotator("hello,is,and,of,the,to")
                /*.extractPos()
                .extractRelations()*/
                .build();
        instance.annotate(document);

        document.getSentences().forEach((sentence) -> {
            System.out.println(">>>" + sentence.getSentence());
            if (sentence.getTokens() != null) {
                sentence.getTokens().forEach((token) -> {
                    System.out.print(" " + token);
                });
                System.out.print("\n ");
            }
        });
    }

}
