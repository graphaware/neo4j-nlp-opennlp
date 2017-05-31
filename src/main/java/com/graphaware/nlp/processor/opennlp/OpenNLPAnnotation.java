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
    public static final String defaultLemmaOpenNLP = "O"; // this value is hardcoded inside DictionaryLemmatizer class

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
        private String sentenceSentiment;
        private List<Integer> nounphrases;
        private String[] words;
        private Span[] wordSpans;
        private String[] posTags;
        private String[] lemmas;
        private String[] namedEntities;
        private List<String> tokens;
        private List<Span> tokenSpans;
        private List<List<String>> tokenPOS;
        private List<String> tokenLemmas;
        private List<List<String>> tokenNEs;
        private Span[] chunks;
        private String[] chunkStrings;
        private String[] chunkSentiments;
        private final String defaultStringValue = "-"; // @Deprecated

        public Sentence(Span sentence, String text) {
            this.sentence = sentence;
            this.sentenceText = String.valueOf(sentence.getCoveredText(text));
            //this.sentenceSentiment = defaultStringValue;
        }
        
        public void addPhraseIndex(int phraseINdex) {
            if (this.nounphrases == null) {
                this.nounphrases = new ArrayList<>();
            }
            this.nounphrases.add(phraseINdex);
        }

        public Span getSentenceSpan() {
            return this.sentence;
        }
        
        public String getSentence() {
            return this.sentenceText;
        }

        public String getSentiment() {
          return this.sentenceSentiment;
        }

        public void setSentiment(String sent) {
          this.sentenceSentiment = sent;
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
                        //.map(span -> new String(this.sentenceText.substring(span.getStart(), span.getEnd())))
                        .map(span -> String.valueOf(span.getCoveredText(this.sentenceText)))
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

        public int getTokenStart(int idx) {
          int i = -1;
          if (this.tokenSpans.size()>idx)
            i = this.tokenSpans.get(idx).getStart();
          return i;
        }

        public int getTokenEnd(int idx) {
          int i = -1;
          if (this.tokenSpans.size()>idx)
            i = this.tokenSpans.get(idx).getEnd();
          return i;
        }

        public List<List<String>> getTokenPosTags() {
          return this.tokenPOS;
        }

        public String[] getPosTags() {
            return this.posTags;
        }

        public void setPosTags(String[] posTags) {
            this.posTags = posTags;
        }

        @Deprecated
        public void setDefaultPosTags() {
          this.posTags = new String[this.words.length];
          Arrays.fill(this.posTags, defaultStringValue);
        }

        public Span[] getChunks() {
            return this.chunks;
        }

        public void setChunks(Span[] chunks) {
            this.chunks = chunks;
        }

        public String[] getChunkStrings() {
            return this.chunkStrings;
        }

        public void setChunkStrings(String[] chunkStrings) {
            this.chunkStrings = chunkStrings;
        }

        public String[] getChunkSentiments() {
          return this.chunkSentiments;
        }

        public void setChunkSentiments(String[] sents) {
          if (sents==null) return;
          if (sents.length!=this.chunks.length) return;
          this.chunkSentiments = sents;
        }

        @Deprecated
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

        public List<List<String>> getTokenNEs() {
          return this.tokenNEs;
        }

        public String[] getNamedEntities() {
          return this.namedEntities;
        }

        @Deprecated
        // this method does not merge tokens (when they belong to the same NE)
        public void setNamedEntity(int idxStart, int idxEnd, String type) {
          if (this.words==null)  // words/tokens must be extracted before Named Entities can be saved
            return;
          if (this.namedEntities==null)
            this.namedEntities = new String[this.words.length];
          for (int i=idxStart; i<idxEnd && i<this.words.length; i++)
            if (this.namedEntities[i]==null) // we don't want to override a possible positive result from previous NamedEntity identification attempts
              this.namedEntities[i] = type;
        }

        public void setNamedEntities(List<Span> ners) {
          if (this.words==null)  // words/tokens must be extracted before Named Entities can be saved
            return;
          if (ners==null) return;
          if (this.tokens==null) {
            this.tokens = new ArrayList<String>();
            this.tokenSpans = new ArrayList<Span>();
            this.tokenLemmas = new ArrayList<String>();
            this.tokenPOS = new ArrayList<>();
            this.tokenNEs = new ArrayList<>();
          }
          int n = this.words.length;
          if (this.namedEntities==null)
            this.namedEntities = new String[this.words.length];
          ners.forEach(ne -> {
            String value = "";
            String lemma = "";
            List<String> posL = new ArrayList<String>();
            for (int i=ne.getStart(); i<ne.getEnd() && i<n; i++) {
              value += " " + this.words[i];
              // now make sure that if no lemma is assigned, only one default value is store in this.tokenLemmas for given token
              if (lemma.equals(defaultLemmaOpenNLP) || this.lemmas[i].equals(defaultLemmaOpenNLP))
                lemma = defaultLemmaOpenNLP;
              else
                lemma += " " + this.lemmas[i];
              // in case not all words associated with the same Named Entity have the same POS tag, store them all
              if (this.posTags[i]!=null) {
                if (!posL.contains(this.posTags[i]))
                  posL.add(this.posTags[i]);
              }
              this.namedEntities[i] = ne.getType(); // needed for finalizeNamedEntities()
            }
            int presentIdx = -1;
            for (int i=0; i<this.tokenSpans.size(); i++) {
              if (this.tokenSpans.get(i).getStart()==ne.getStart() && this.tokenSpans.get(i).getEnd()==ne.getEnd()) {
                presentIdx = i;
                break;
              }
            }
            if (presentIdx!=-1) {
              this.tokenNEs.get(presentIdx).add(ne.getType());
              //this.tokenNEs.set(presentIdx, this.tokenNEs.get(presentIdx).add(ne.getType()));
            } else {
              this.tokens.add(value);
              this.tokenSpans.add(ne);
              this.tokenLemmas.add(lemma);
              this.tokenPOS.add(posL);
              this.tokenNEs.add(Arrays.asList(ne.getType()));
            }
          });
        }

        public void finalizeNamedEntities() {
          for (int i=0; i<this.namedEntities.length; i++) {
            if (this.namedEntities[i]==null) {
              this.tokens.add(this.words[i]);
              this.tokenSpans.add(this.wordSpans[i]);
              this.tokenLemmas.add(this.lemmas[i]);
              this.tokenPOS.add(Arrays.asList(this.posTags[i]));
              this.tokenNEs.add(null);
            }
          }
        }

        @Deprecated
        public void setDefaultNamedEntities() {
          this.namedEntities = new String[this.words.length];
          Arrays.fill(this.namedEntities, defaultStringValue);
        }

        public List<String> getTokens() {
          return this.tokens;
        }

        public List<String> getTokenLemmas() {
          return this.tokenLemmas;
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

        @Deprecated
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
            this.lemmas[idx] = lemma;
        }
    }
}
