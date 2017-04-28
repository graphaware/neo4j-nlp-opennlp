/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
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
        private String[] words;
        private Span[] wordSpans;
        private String[] posTags;
        private String[] lemmas;
        private Span[] chunks;
        private String[] chunkStrings;
        private String[] namedEntities;
        private String defaultStringValue = "-";

        public Sentence(Span sentence, String text) {
            this.sentence = sentence;
            this.sentenceText = String.valueOf(sentence.getCoveredText(text));
        }
        
        public void addPhraseIndex(int phraseINdex) {
            if (this.nounphrases == null) {
                this.nounphrases = new ArrayList<>();
            }
            this.nounphrases.add(phraseINdex);
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

        public Span[] getWordSpans() {
            return this.wordSpans;
        }

        public void setWordSpans(Span[] spans) {
          this.wordSpans = spans;
        }

        public void setWordsAndSpans(Span[] spans) {
          if (spans==null) {
            this.wordSpans = null;
            this.words = null;
            return;
          }
          this.wordSpans = spans;
          this.words = new String[this.wordSpans.length];
          this.words = Arrays.asList(spans).stream()
                        .map(span -> new String(this.sentenceText.substring(span.getStart(), span.getEnd())))
                        .collect(Collectors.toList()).toArray(this.words);
        }

        public int getWordStart(int idx) {
          int i = -1;
          if (this.wordSpans.length>idx)
            i = this.wordSpans[idx].getStart();
          return i;
        }

        public int getWordEnd(int idx) {
          int i = -1;
          if (this.wordSpans.length>idx)
            i = this.wordSpans[idx].getEnd();
          return i;
        }

        public String[] getPosTags() {
            return posTags;
        }

        public void setPosTags(String[] posTags) {
            this.posTags = posTags;
        }

        public void setDefaultPosTags() {
          this.posTags = new String[this.words.length];
          Arrays.fill(this.posTags, defaultStringValue);
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

        public void setDefaultChunks() {
          this.chunks = new Span[this.words.length];
          Arrays.fill(this.chunks, new Span(0, 0));
          this.chunkStrings = new String[this.words.length];
          Arrays.fill(this.chunkStrings, defaultStringValue);
          this.nounphrases = new ArrayList<>();
        }

        public List<Integer> getPhrasesIndex() {
            //if (nounphrases==null)
              //return new ArrayList<Integer>();
            return nounphrases;
        }

        public String[] getNamedEntities() {
          return this.namedEntities;
        }

        public void setNamedEntity(int idxStart, int idxEnd, String type) {
          if (this.words==null)  // words/tokens must be extracted before Named Entities can be saved
            return;
          if (this.namedEntities==null)
            this.namedEntities = new String[this.words.length];
          for (int i=idxStart; i<idxEnd && i<this.words.length; i++)
            if (this.namedEntities[i]==null) // we don't want to override a possible positive result from previous NamedEntity identification attempts
              this.namedEntities[i] = type;
        }

        public void setDefaultNamedEntities() {
          this.namedEntities = new String[this.words.length];
          Arrays.fill(this.namedEntities, defaultStringValue);
        }

        public String[] getLemmas() {
          return this.lemmas;
        }

        public void setLemmas(String[] lemmas) {
          if (this.words==null || lemmas==null)
            return;
          if (this.words.length!=lemmas.length) // ... something is wrong
            return;
          this.lemmas = lemmas;
        }

        public void setDefaultLemmas() {
          this.lemmas = new String[this.words.length];
          Arrays.fill(this.lemmas, "O");
        }

        public void setLemma(int idx, String lemma) {
          if (this.words==null)
            return;
          if (this.lemmas==null)
            this.lemmas = new String[this.words.length];
          if (idx<this.lemmas.length)
            lemmas[idx] = lemma;
        }
    }
}
