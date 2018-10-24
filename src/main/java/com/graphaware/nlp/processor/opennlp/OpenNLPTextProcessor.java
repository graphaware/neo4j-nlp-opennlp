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

import com.graphaware.nlp.annotation.NLPTextProcessor;
import com.graphaware.nlp.domain.*;
import com.graphaware.nlp.dsl.request.PipelineSpecification;
import com.graphaware.nlp.processor.AbstractTextProcessor;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import com.graphaware.nlp.util.Timer;
import opennlp.tools.util.Span;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@NLPTextProcessor(name = "OpenNLPTextProcessor")
public class OpenNLPTextProcessor extends AbstractTextProcessor {

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPTextProcessor.class);

    private static final String CORE_PIPELINE_NAME = "OpenNLP.CORE";
    public static final String TOKENIZER = "tokenizer";
    public static final String SENTIMENT = "sentiment";

    private final Map<String, OpenNLPPipeline> pipelines = new HashMap<>();


    @Override
    public void init() {
    }

    @Override
    public String getAlias() {
        return "opennlp";
    }

    @Override
    public String override() {
        return null;
    }

    public OpenNLPPipeline getPipeline(String name) {
        if (name == null || name.isEmpty()) {
            name = TOKENIZER;
            LOG.debug("Using default pipeline: " + name);
        }
        OpenNLPPipeline pipeline = getOpenNLPPipeline(name);
        return pipeline;
    }

    private void checkPipelineExistOrCreate(PipelineSpecification pipelineSpecification) {
        if (!pipelines.containsKey(pipelineSpecification.getName())) {
            createPipeline(pipelineSpecification);
        }
    }

/*    private void createFullPipeline() {
        OpenNLPPipeline pipeline = new PipelineBuilder()
                .tokenize()
                .extractNEs()
                .defaultStopWordAnnotator()
                .extractRelations()
                .extractSentiment()
                .threadNumber(6)
                .build();
        pipelines.put(CORE_PIPELINE_NAME, pipeline);
    }

    private void createTokenizerPipeline() {
        OpenNLPPipeline pipeline = pipelines.get(CORE_PIPELINE_NAME);
        pipelines.put(TOKENIZER, pipeline);
    }

    private void createSentimentPipeline() {
        OpenNLPPipeline pipeline = pipelines.get(CORE_PIPELINE_NAME);
        pipelines.put(SENTIMENT, pipeline);
    }

    private void createTokenizerAndSentimentPipeline() {
        OpenNLPPipeline pipeline = pipelines.get(CORE_PIPELINE_NAME);
        pipelines.put(TOKENIZER_AND_SENTIMENT, pipeline);
    }

    private void createPhrasePipeline() {
        OpenNLPPipeline pipeline = pipelines.get(CORE_PIPELINE_NAME);
        pipelines.put(PHRASE, pipeline);
    }*/

    @Override
    public AnnotatedText annotateText(String text, String lang, PipelineSpecification pipelineSpecification) {
        Timer timer = Timer.start();
        checkPipelineExistOrCreate(pipelineSpecification);
        timer.lap("pipeline check");
        OpenNLPPipeline pipeline = pipelines.get(pipelineSpecification.getName());
        OpenNLPAnnotation document = new OpenNLPAnnotation(text, Collections.EMPTY_MAP);
        pipeline.annotate(document);

        AnnotatedText result = new AnnotatedText();
        List<OpenNLPAnnotation.Sentence> sentences = document.getSentences();
        final AtomicInteger sentenceSequence = new AtomicInteger(0);
        sentences.stream().forEach((sentence) -> {
            int sentenceNumber = sentenceSequence.getAndIncrement();
            final Sentence newSentence = new Sentence(sentence.getSentence(), sentenceNumber);
            extractTokens(lang, sentence, newSentence);
            if (pipelineSpecification.hasProcessingStep(STEP_SENTIMENT)) {
                extractSentiment(sentence, newSentence);
            }
            if (pipelineSpecification.hasProcessingStep(STEP_PHRASE)) {
                extractPhrases(sentence, newSentence);
            }
            result.addSentence(newSentence);
        });

        return result;
    }

    protected Map<String, Object> getPipelineProperties(OpenNLPPipeline pipeline) {
        Map<String, Object> options = new HashMap<>();
        for (Object o : pipeline.getProperties().keySet()) {
            if (o instanceof String) {
                options.put(o.toString(), pipeline.getProperties().getProperty(o.toString()));
            }
        }

        return options;
    }

    protected Map<String, Object> buildSpecifications(List<String> actives) {
        List<String> all = Arrays.asList("tokenize", "ner", "cleanxml", "truecase", "dependency", "relations", "checkLemmaIsStopWord", "coref", "sentiment", "phrase", "customSentiment", "customNER");
        Map<String, Object> specs = new HashMap<>();
        all.forEach(s -> {
            specs.put(s, actives.contains(s));
        });

        return specs;
    }


/*    @Override
    public AnnotatedText annotateText(String text, String name, String lang, Map<String, String> otherParams) {
        if (name.length() == 0) {
            name = TOKENIZER;
            LOG.info("Using default pipeline: " + name);
        }
        OpenNLPPipeline pipeline = pipelines.get(name);
        if (pipeline == null) {
            throw new RuntimeException("Pipeline: " + name + " doesn't exist");
        }
        OpenNLPAnnotation document = new OpenNLPAnnotation(text, otherParams);
        pipeline.annotate(document);
//        LOG.info("Annotation for id " + id + " finished.");

        AnnotatedText result = new AnnotatedText();
        List<OpenNLPAnnotation.Sentence> sentences = document.getSentences();
        final AtomicInteger sentenceSequence = new AtomicInteger(0);
        sentences.stream().forEach((sentence) -> {
            int sentenceNumber = sentenceSequence.getAndIncrement();
//            String sentenceId = id + "_" + sentenceNumber;
            final Sentence newSentence = new Sentence(sentence.getSentence(), sentenceNumber);
            extractTokens(lang, sentence, newSentence);
            extractSentiment(sentence, newSentence);
            extractPhrases(sentence, newSentence);
            result.addSentence(newSentence);
        });
        //extractRelationship(result, sentences, document);
        return result;
    }
*/
    private void extractPhrases(OpenNLPAnnotation.Sentence sentence, Sentence newSentence) {
        if (sentence.getPhrasesIndex() == null) {
            LOG.warn("extractPhrases(): phrases index empty, aborting extraction");
            return;
        }
        sentence.getPhrasesIndex().forEach(index -> {
            Span chunk = sentence.getChunks()[index];
            String chunkString = sentence.getChunkStrings()[index];
            newSentence.addPhraseOccurrence(chunk.getStart(), chunk.getEnd(), new Phrase(chunkString, chunk.getType()));
        });
    }

    private void extractSentiment(OpenNLPAnnotation.Sentence sentence, Sentence newSentence) {
        int score = -1;
        if (sentence.getSentiment() != null) { // && !sentence.getSentiment().equals("-")) {
            try {
                score = Integer.valueOf(sentence.getSentiment());
            } catch (NumberFormatException ex) {
                LOG.error("NumberFormatException: error extracting sentiment " + sentence.getSentiment() + " as a number.", ex);
            }
        }
        newSentence.setSentiment(score);
    }

    private void extractTokens(String lang, OpenNLPAnnotation.Sentence sentence, final Sentence newSentence) {
        Collection<OpenNLPAnnotation.Token> tokens = sentence.getTokens();
        tokens.stream().filter((token) -> token != null /*&& checkLemmaIsValid(token.getToken())*/).forEach((token) -> {
            Tag newTag = getTag(token, lang);
            if (newTag != null) {
                Tag tagInSentence = newSentence.addTag(newTag);
                token.getTokenSpans().stream().forEach((span) -> {
                    newSentence.addTagOccurrence(span.getStart(), span.getEnd(), token.getToken(), tagInSentence);
                });
            }
        });
    }

    //    private void extractRelationship(AnnotatedText annotatedText, List<CoreMap> sentences, Annotation document) {
//        Map<Integer, CorefChain> corefChains = document.get(CorefCoreAnnotations.CorefChainAnnotation.class);
//        if (corefChains != null) {
//            for (CorefChain chain : corefChains.values()) {
//                CorefChain.CorefMention representative = chain.getRepresentativeMention();
//                int representativeSenteceNumber = representative.sentNum - 1;
//                List<CoreLabel> representativeTokens = sentences.get(representativeSenteceNumber).get(CoreAnnotations.TokensAnnotation.class);
//                int beginPosition = representativeTokens.get(representative.startIndex - 1).beginPosition();
//                int endPosition = representativeTokens.get(representative.endIndex - 2).endPosition();
//                Phrase representativePhraseOccurrence = annotatedText.getSentences().get(representativeSenteceNumber).getPhraseOccurrence(beginPosition, endPosition);
//                if (representativePhraseOccurrence == null) {
//                    LOG.warn("Representative Phrase not found: " + representative.mentionSpan);
//                }
//                for (CorefChain.CorefMention mention : chain.getMentionsInTextualOrder()) {
//                    if (mention == representative) {
//                        continue;
//                    }
//                    int mentionSentenceNumber = mention.sentNum - 1;
//
//                    List<CoreLabel> mentionTokens = sentences.get(mentionSentenceNumber).get(CoreAnnotations.TokensAnnotation.class);
//                    int beginPositionMention = mentionTokens.get(mention.startIndex - 1).beginPosition();
//                    int endPositionMention = mentionTokens.get(mention.endIndex - 2).endPosition();
//                    Phrase mentionPhraseOccurrence = annotatedText.getSentences().get(mentionSentenceNumber).getPhraseOccurrence(beginPositionMention, endPositionMention);
//                    if (mentionPhraseOccurrence == null) {
//                        LOG.warn("Mention Phrase not found: " + mention.mentionSpan);
//                    }
//                    if (representativePhraseOccurrence != null
//                            && mentionPhraseOccurrence != null) {
//                        mentionPhraseOccurrence.setReference(representativePhraseOccurrence);
//                    }
//                }
//            }
//        }
//    }
    @Override
    public Tag annotateSentence(String text, String lang, PipelineSpecification pipelineSpecification) {
//        Annotation document = new Annotation(text);
//        pipelines.get(SENTIMENT).annotate(document);
//        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
//        Optional<CoreMap> sentence = sentences.stream().findFirst();
//        if (sentence.isPresent()) {
//            Optional<Tag> oTag = sentence.get().get(CoreAnnotations.TokensAnnotation.class).stream()
//                    .map((token) -> getTag(token))
//                    .filter((tag) -> (tag != null) && checkPunctuation(tag.getLemma()))
//                    .findFirst();
//            if (oTag.isPresent()) {
//                return oTag.get();
//            }
//        }
        return null;
    }

    @Override
    public Tag annotateTag(String text, String lang, PipelineSpecification pipelineSpecification) {
        OpenNLPAnnotation document = new OpenNLPAnnotation(text);
        final OpenNLPPipeline openNLPPipeline = getOpenNLPPipeline(pipelineSpecification.getName());
        openNLPPipeline.annotate(document);
        List<OpenNLPAnnotation.Sentence> sentences = document.getSentences();
        if (sentences != null && !sentences.isEmpty()) {
            if (sentences.size() > 1) {
                throw new RuntimeException("More than one sentence");
            }
            Collection<OpenNLPAnnotation.Token> tokens = sentences.get(0).getTokens();
            if (tokens != null && tokens.size() == 1) {
                OpenNLPAnnotation.Token token = tokens.iterator().next();
                Tag newTag = getTag(token, lang);
                return newTag;
            } else if (tokens != null && tokens.size() > 1) {
                OpenNLPAnnotation.Token token = document.getToken(text, text);
                Tag newTag = getTag(token, lang);
                return newTag;
            }
        }
        return null;
    }

    @NotNull
    private OpenNLPPipeline getOpenNLPPipeline(String name) {
        final OpenNLPPipeline openNLPPipeline = pipelines.get(name);
        if (openNLPPipeline == null) {
            throw new RuntimeException("Pipeline " + name + " doesn't exist");
        }
        return openNLPPipeline;
    }

    private Tag getTag(OpenNLPAnnotation.Token token, String lang) {
        List<String> pos = new ArrayList<>();
        List<String> ne = new ArrayList<>();
        String lemma = token.getTokenLemmas();
        pos.addAll(token.getTokenPOS());
        ne.addAll(token.getTokenNEs());

        // apply lemma validity check (to all words in case of NamedEntities)
        lemma = Arrays.asList(lemma.split(" ")).stream().filter(str -> checkLemmaIsValid(str)).collect(Collectors.joining(" "));
        if (lemma == null || lemma.length() == 0)
            return null;

        Tag tag = new Tag(lemma, lang);
        tag.setPos(pos);
        tag.setNe(ne);
        LOG.info("POS: " + pos + " ne: " + ne + " lemma: " + lemma);
        return tag;
    }

    private List<Tag> annotateTagsAux(String text, String lang, OpenNLPPipeline pipeline) {
        List<Tag> result = new ArrayList<>();
        OpenNLPAnnotation document = new OpenNLPAnnotation(text);
        pipeline.annotate(document);
        List<OpenNLPAnnotation.Sentence> sentences = document.getSentences();
        if (sentences != null && !sentences.isEmpty()) {
            if (sentences.size() > 1) {
                throw new RuntimeException("More than one sentence");
            }
            Collection<OpenNLPAnnotation.Token> tokens = sentences.get(0).getTokens();
            if (tokens != null && tokens.size() > 0) {
                tokens.stream().forEach((token) -> {
                    Tag newTag = getTag(token, lang);
                    if (newTag != null)
                        result.add(newTag);
                });
                return result;
            }
        }
        return null;
    }

    @Override
    public List<Tag> annotateTags(String text, String lang, PipelineSpecification pipelineSpecification) {
        return annotateTagsAux(text, lang, getOpenNLPPipeline(pipelineSpecification.getName()));
    }

    public List<Tag> annotateTags(String text, String lang) {
        return annotateTagsAux(text, lang, getOpenNLPPipeline(TOKENIZER));
    }

    @Override
    public AnnotatedText sentiment(AnnotatedText annotated) {
        OpenNLPPipeline pipeline = getOpenNLPPipeline(SENTIMENT);
        annotated.getSentences().stream().forEach(item -> { // don't use parallelStream(), it crashes with the current content of the body
            OpenNLPAnnotation document = new OpenNLPAnnotation(item.getSentence());
            pipeline.annotate(document);

            List<OpenNLPAnnotation.Sentence> sentences = document.getSentences();
            Optional<OpenNLPAnnotation.Sentence> sentence = sentences.stream().findFirst();
            if (sentence != null && sentence.isPresent()) {
                extractSentiment(sentence.get(), item);
            }
        });

        return annotated;
    }

    @Override
    public String train(String alg, String modelId, String file, String lang, Map<String, Object> params) {
        // training could be done directly here, but it's better to have everything related to model implementation in one class, therefore ...
        OpenNLPPipeline pipeline = getOpenNLPPipeline(TOKENIZER);
        return pipeline.train(alg, modelId, file, lang, params);
    }

    @Override
    public String test(String alg, String modelId, String file, String lang) {
        OpenNLPPipeline pipeline = getOpenNLPPipeline(TOKENIZER);
        return pipeline.test(alg, modelId, file, lang);

    }

    class TokenHolder {

        private String ne;
        private StringBuilder sb;
        private int beginPosition;
        private int endPosition;

        public TokenHolder() {
            reset();
        }

        public String getNe() {
            return ne;
        }

        public String getToken() {
            if (sb == null) {
                return " - ";
            }
            return sb.toString();
        }

        public int getBeginPosition() {
            return beginPosition;
        }

        public int getEndPosition() {
            return endPosition;
        }

        public void setNe(String ne) {
            this.ne = ne;
        }

        public void updateToken(String tknStr) {
            this.sb.append(tknStr);
        }

        public void setBeginPosition(int beginPosition) {
            if (this.beginPosition < 0) {
                this.beginPosition = beginPosition;
            }
        }

        public void setEndPosition(int endPosition) {
            this.endPosition = endPosition;
        }

        public final void reset() {
            sb = new StringBuilder();
            beginPosition = -1;
            endPosition = -1;
        }
    }

    class PhraseHolder implements Comparable<PhraseHolder> {

        private StringBuilder sb;
        private int beginPosition;
        private int endPosition;

        public PhraseHolder() {
            reset();
        }

        public String getPhrase() {
            if (sb == null) {
                return " - ";
            }
            return sb.toString();
        }

        public int getBeginPosition() {
            return beginPosition;
        }

        public int getEndPosition() {
            return endPosition;
        }

        public void updatePhrase(String tknStr) {
            this.sb.append(tknStr);
        }

        public void setBeginPosition(int beginPosition) {
            if (this.beginPosition < 0) {
                this.beginPosition = beginPosition;
            }
        }

        public void setEndPosition(int endPosition) {
            this.endPosition = endPosition;
        }

        public final void reset() {
            sb = new StringBuilder();
            beginPosition = -1;
            endPosition = -1;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof PhraseHolder)) {
                return false;
            }
            PhraseHolder otherObject = (PhraseHolder) o;
            if (this.sb != null
                    && otherObject.sb != null
                    && this.sb.toString().equals(otherObject.sb.toString())
                    && this.beginPosition == otherObject.beginPosition
                    && this.endPosition == otherObject.endPosition) {
                return true;
            }
            return false;
        }

        @Override
        public int compareTo(PhraseHolder o) {
            if (o == null) {
                return 1;
            }
            if (this.equals(o)) {
                return 0;
            } else if (this.beginPosition > o.beginPosition) {
                return 1;
            } else if (this.beginPosition == o.beginPosition) {
                if (this.endPosition > o.endPosition) {
                    return 1;
                }
            }
            return -1;
        }
    }

    @Override
    public List<String> getPipelines() {
        return new ArrayList<>(pipelines.keySet());
    }

    @Override
    public boolean checkPipeline(String name) {
        return pipelines.containsKey(name);
    }

    @Override
    public void createPipeline(PipelineSpecification pipelineSpecification) {
        //TODO add validation
        String name = pipelineSpecification.getName();
        PipelineBuilder pipelineBuilder = new PipelineBuilder();
        List<String> specActive = new ArrayList<>();
        List<String> stopwordsList;

        if (pipelineSpecification.hasProcessingStep("tokenize", true)) {
            pipelineBuilder.tokenize();
            specActive.add("tokenize");
        }

        if (pipelineSpecification.hasProcessingStep("ner", true)) {
            pipelineBuilder.extractNEs();
            specActive.add("ner");
        }

        String stopWords = pipelineSpecification.getStopWords() != null ? pipelineSpecification.getStopWords() : "default";
        boolean checkLemma = pipelineSpecification.hasProcessingStep("checkLemmaIsStopWord");
        if (checkLemma) {
            specActive.add("checkLemmaIsStopWord");
        }

        if (stopWords.equalsIgnoreCase("default")) {
            pipelineBuilder.defaultStopWordAnnotator();
            stopwordsList = PipelineBuilder.getDefaultStopwords();
        } else {
            pipelineBuilder.customStopWordAnnotator(stopWords);
            stopwordsList = PipelineBuilder.getCustomStopwordsList(stopWords);
        }

        if (pipelineSpecification.hasProcessingStep("sentiment")) {
            pipelineBuilder.extractSentiment();
            specActive.add("sentiment");
        }
        if (pipelineSpecification.hasProcessingStep("coref")) {
            pipelineBuilder.extractCoref();
            specActive.add("coref");
        }
        if (pipelineSpecification.hasProcessingStep("relations")) {
            pipelineBuilder.extractRelations();
            specActive.add("relations");
        }
        if (pipelineSpecification.hasProcessingStep("customNER")) {
            if (!specActive.contains("ner")) {
                pipelineBuilder.extractNEs();
                specActive.add("ner");
            }
            specActive.add("customNER");
            pipelineBuilder.extractCustomNEs(pipelineSpecification.getProcessingStepAsString("customNER"));
        }
        if (pipelineSpecification.hasProcessingStep("customSentiment")) {
            if (!specActive.contains("sentiment")) {
                pipelineBuilder.extractSentiment();
                specActive.add("sentiment");
            }
            specActive.add("customSentiment");
            pipelineBuilder.extractCustomSentiment(pipelineSpecification.getProcessingStepAsString("customSentiment"));
        }
        Long threadNumber = pipelineSpecification.getThreadNumber() != 0 ? pipelineSpecification.getThreadNumber() : 4L;
        pipelineBuilder.threadNumber(threadNumber.intValue());

        OpenNLPPipeline pipeline = pipelineBuilder.build();
        pipelines.put(name, pipeline);
    }


    @Override
    public void removePipeline(String name) {
        if (!pipelines.containsKey(name)) {
            throw new RuntimeException("No pipeline found with name: " + name);
        }
        pipelines.remove(name);
    }
}
