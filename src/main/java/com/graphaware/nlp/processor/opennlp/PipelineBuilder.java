/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
        annotators.append("tokenize, pos, lemma");
        return this;
    }

    public PipelineBuilder extractNEs() {
        checkForExistingAnnotators();
        annotators.append("ner");
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

    public PipelineBuilder extractCustomNEs(String ners) {
        properties.setProperty("customNEs", ners);
        return this;
    }

    public PipelineBuilder extractCustomSentiment(String sent) {
        System.out.println("Adding custom sentiment model from " + sent);
        properties.setProperty("customSentiment", sent);
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
            stopWordList = CUSTOM_STOP_WORD_LIST + "," + customStopWordList;
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
    
    public static List<String> getDefaultStopwords() {
        List<String> stopwords = new ArrayList<>();
        Arrays.stream(CUSTOM_STOP_WORD_LIST.split(",")).forEach(s -> {
            stopwords.add(s.trim());
        });

        return stopwords;
    }

    public static List<String> getCustomStopwordsList(String customStopWordList) {
        String stopWordList;
        if (customStopWordList.startsWith("+")) {
            stopWordList = CUSTOM_STOP_WORD_LIST + "," + customStopWordList.replace("+,", "").replace("+", "");
        } else {
            stopWordList = customStopWordList;
        }

        List<String> list = new ArrayList<>();
        Arrays.stream(stopWordList.split(",")).forEach(s -> {
            list.add(s.trim());
        });

        return list;
    }
}
