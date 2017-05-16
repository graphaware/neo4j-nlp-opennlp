/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.util.Properties;

class PipelineBuilder {

    private static final String CUSTOM_STOP_WORD_LIST = "start,starts,period,periods,a,an,and,are,as,at,be,but,by,for,if,in,into,is,it,no,not,of,o,on,or,such,that,the,their,then,there,these,they,this,to,was,will,with";

    private final Properties properties = new Properties();
    private final StringBuilder annotators = new StringBuilder(); //basics annotators
    private int threadsNumber = 4;

    private void checkForExistingAnnotators() {
        if (annotators.toString().length() > 0) {
            annotators.append(", ");
        }
    }

    public PipelineBuilder tokenize() {
        checkForExistingAnnotators();
        annotators.append("tokenize, pos, lemma, ner");
        return this;
    }

    public PipelineBuilder extractSentiment() {
        checkForExistingAnnotators();
        annotators.append("sentiment");
        return this;
    }

    public PipelineBuilder extractRelations() {
        checkForExistingAnnotators();
        annotators.append("relation");
        return this;
    }

    public PipelineBuilder extractCoref() {
        return this;
    }

    public PipelineBuilder defaultStopWordAnnotator() {
        checkForExistingAnnotators();
        annotators.append("stopword");
        properties.setProperty("stopword", CUSTOM_STOP_WORD_LIST);
        return this;
    }

    public PipelineBuilder customStopWordAnnotator(String customStopWordList) {
        checkForExistingAnnotators();
        String stopWordList;
        if (annotators.indexOf("stopword") >= 0) {
            String alreadyexistingStopWordList = properties.getProperty("stopword");
            stopWordList = alreadyexistingStopWordList + "," + customStopWordList;
        } else {
            annotators.append("stopword");
            stopWordList = customStopWordList;
        }
        properties.setProperty("stopword", stopWordList);
        return this;
    }

    public PipelineBuilder stopWordAnnotator(Properties properties) {
        return this;
    }

    public PipelineBuilder threadNumber(int threads) {
        this.threadsNumber = threads;
        return this;
    }

    public OpenNLPPipeline build() {
        properties.setProperty("annotators", annotators.toString());
        properties.setProperty("threads", String.valueOf(threadsNumber));
        OpenNLPPipeline pipeline = new OpenNLPPipeline(properties);
        return pipeline;
    }
}
