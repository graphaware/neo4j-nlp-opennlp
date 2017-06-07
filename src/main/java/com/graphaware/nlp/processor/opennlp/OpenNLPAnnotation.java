/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.graphaware.nlp.processor.opennlp;

import com.graphaware.nlp.util.OptionalNLPParameters;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import opennlp.tools.util.Span;

public class OpenNLPAnnotation {

    private static final double DEFAULT_SENTIMENT_PROBTHR = 0.7;

    private final String text;
    private List<Sentence> sentences;
    public static final String DEFAULT_LEMMA_OPEN_NLP = "O";
    public Map<String, String> otherParams;

    public OpenNLPAnnotation(String text, Map<String, String> otherParams) {
        this.text = text;
        this.otherParams = otherParams;
    }

    public OpenNLPAnnotation(String text) {
        this(text, null);
    }

    public String getText() {
        return text;
    }

    public void setSentences(Span[] sentencesArray) {
        sentences = new ArrayList<>();
        for (Span sentence : sentencesArray) {
            sentences.add(new Sentence(sentence, getText()));
        }
    }

    public List<Sentence> getSentences() {
        return sentences;
    }

    public double getSentimentProb() {
        if (otherParams != null && otherParams.containsKey(OptionalNLPParameters.SENTIMENT_PROB_THR)) {
            return Double.parseDouble(otherParams.get(OptionalNLPParameters.SENTIMENT_PROB_THR));
        }
        return DEFAULT_SENTIMENT_PROBTHR;
    }

    public String getProject() {
        if (otherParams != null) {
            return otherParams.getOrDefault(OptionalNLPParameters.CUSTOM_PROJECT, null);
        }
        return null;
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
        private final Map<String, Token> tokens;
        private Span[] chunks;
        private String[] chunkStrings;
        private String[] chunkSentiments;
        private final String defaultStringValue = "-"; // @Deprecated

        public Sentence(Span sentence, String text) {
            this.sentence = sentence;
            this.sentenceText = String.valueOf(sentence.getCoveredText(text));
            this.tokens = new HashMap<>();
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
            if (spans == null) {
                this.wordSpans = null;
                this.words = null;
                return;
            }
            this.wordSpans = spans;
            this.words = Arrays.asList(spans).stream()
                    //.map(span -> new String(this.sentenceText.substring(span.getStart(), span.getEnd())))
                    .map(span -> String.valueOf(span.getCoveredText(this.sentenceText)))
                    .collect(Collectors.toList()).toArray(new String[wordSpans.length]);
        }

        public int getWordStart(int idx) {
            if (this.wordSpans.length > idx) {
                return this.wordSpans[idx].getStart();
            }
            return -1;
        }

        public int getWordEnd(int idx) {
            if (this.wordSpans.length > idx) {
                return this.wordSpans[idx].getEnd();
            }
            return -1;
        }

        public String[] getPosTags() {
            return this.posTags;
        }

        public void setPosTags(String[] posTags) {
            this.posTags = posTags;
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
            if (sents == null) {
                return;
            }
            if (sents.length != this.chunks.length) {
                return;
            }
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

        public Collection<Token> getTokens() {
            return this.tokens.values();
        }

        public String[] getLemmas() {
            return this.lemmas;
        }

        public void setLemmas(String[] lemmas) {
            if (this.words == null || lemmas == null) {
                return;
            }
            if (this.words.length != lemmas.length) // ... something is wrong
            {
                return;
            }
            this.lemmas = lemmas;
        }

        protected Token getToken(String value, String lemma) {
            Token token;
            if (tokens.containsKey(value)) {
                token = tokens.get(value);
            } else {
                token = new Token(value, lemma);
                tokens.put(value, token);
            }
            return token;
        }
    }

    class Token {

        private final String token;
        private final Set<String> tokenPOS;
        private final String tokenLemmas;
        private final Set<String> tokenNEs;
        private final List<Span> tokenSpans;

        public Token(String token, String lemma) {
            this.token = token;
            this.tokenLemmas = lemma;
            this.tokenNEs = new HashSet<>();
            this.tokenPOS = new HashSet<>();
            this.tokenSpans = new ArrayList<>();
        }

        public List<Span> getTokenSpans() {
            return tokenSpans;
        }

        public String getToken() {
            return token;
        }

        public void addTokenSpans(Span tokenSpans) {
            this.tokenSpans.add(tokenSpans);
        }

        public Collection<String> getTokenPOS() {
            return tokenPOS;
        }

        public void addTokenPOS(Collection<String> tokenPOSes) {
            this.tokenPOS.addAll(tokenPOSes);
        }

        public String getTokenLemmas() {
            return tokenLemmas;
        }

        public Collection<String> getTokenNEs() {
            return tokenNEs;
        }

        public void addTokenNE(String ne) {
            this.tokenNEs.add(ne);
        }

    }
}
