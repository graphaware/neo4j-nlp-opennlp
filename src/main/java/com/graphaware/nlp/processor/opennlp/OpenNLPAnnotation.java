/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.util.ArrayList;
import java.util.List;
import opennlp.tools.util.Span;

/**
 *
 * @author ale
 */
public class OpenNLPAnnotation {

    private final String text;
    private List<Sentence> sentences;

    public OpenNLPAnnotation(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }
    
    public void setSentences(Span[] sentencesArray) {
        sentences = new ArrayList<>();
        for (Span sentence: sentencesArray) {
            sentences.add(new Sentence(sentence, getText()));
        }
    }

    public List<Sentence> getSentences() {
        return sentences;
    }
    
    

    class Sentence {

        private final Span sentence;
        private final String sentenceText;
        private List<Integer> nounphrases;
        String[] words;
        String[] posTags;
        Span[] chunks;
        String[] chunkStrings;

        public Sentence(Span sentence, String text) {
            this.sentence = sentence;
            this.sentenceText = String.valueOf(sentence.getCoveredText(text));
        }
        
        public void addPhraseIndex(int phraseINdex) {
            if (nounphrases == null) {
                nounphrases = new ArrayList<>();
            }
            nounphrases.add(phraseINdex);
        }

        public Span getSentenceSpan() {
            return sentence;
        }
        
        public String getSentence() {
            return sentenceText;
        }
        
        public String[] getWords() {
            return words;
        }

        public void setWords(String[] words) {
            this.words = words;
        }

        public String[] getPosTags() {
            return posTags;
        }

        public void setPosTags(String[] posTags) {
            this.posTags = posTags;
        }

        public Span[] getChunks() {
            return chunks;
        }

        public void setChunks(Span[] chunks) {
            this.chunks = chunks;
        }

        public String[] getChunkStrings() {
            return chunkStrings;
        }

        public void setChunkStrings(String[] chunkStrings) {
            this.chunkStrings = chunkStrings;
        }

        public List<Integer> getPhrasesIndex() {
            return nounphrases;
        }
    }
}
