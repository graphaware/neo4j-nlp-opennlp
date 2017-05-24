/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import opennlp.tools.util.Span;
import opennlp.tools.util.model.BaseModel;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

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
            sentence.getPhrasesIndex().forEach((phrase) -> {
                System.out.println(">>>" + sentence.getChunkStrings()[phrase]);
            });
        });
    }

   
    
}
