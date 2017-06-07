/*
 *
 *
 */
package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.processor.opennlp.OpenNLPPipeline;
import java.io.IOException;
import java.util.Map;

import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderCrossValidator;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderModel;

import opennlp.tools.util.ObjectStream;

import com.graphaware.nlp.util.GenericModelParameters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NERModelTool extends OpenNLPGenericModelTool {

    private String entityType;
    private final String lang;
    private static final String MODEL_NAME = "NER";

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    public NERModelTool(String fileIn, String modelDescr, String lang, Map<String, String> params) {
        super(fileIn, modelDescr, params);
        this.lang = lang;
        this.entityType = null; // train only specific named entity; null = train all entities present in the training set
        if (params != null) {
            if (params.containsKey(GenericModelParameters.TRAIN_ENTITYTYPE)) {
                this.entityType = params.get(GenericModelParameters.TRAIN_ENTITYTYPE);
            }
        }
    }

    public NERModelTool(String fileIn, String modelDescr, String lang) {
        this(fileIn, modelDescr, lang, null);
    }

    public void train() {
        LOG.info("Starting training of " + MODEL_NAME + " ...");
        try (ObjectStream<String> lineStream = openFile(fileIn); NameSampleDataStream sampleStream = new NameSampleDataStream(lineStream)) {
            this.model = NameFinderME.train(lang, entityType, sampleStream, trainParams, new TokenNameFinderFactory());
        } catch (IOException ex) {
            LOG.error("Error while opening training: " + fileIn, ex);
            throw new RuntimeException("Error while training " + MODEL_NAME + " model " + this.modelDescr, ex);
        } catch (Exception ex) {
            LOG.error("Error while training " + MODEL_NAME + " model " + modelDescr);
            throw new RuntimeException("Error while training " + MODEL_NAME + " model " + this.modelDescr, ex);
        }
    }

    public String validate() {
        LOG.info("Starting validation of " + MODEL_NAME + " ...");
        String result = "";
        if (this.fileValidate == null) {
            //List<EvaluationMonitor<NameSample>> listeners = new LinkedList<EvaluationMonitor<NameSample>>();
            try (ObjectStream<String> lineStream = openFile(fileIn); NameSampleDataStream sampleStream = new NameSampleDataStream(lineStream)) {
                // Using CrossValidator
                TokenNameFinderCrossValidator evaluator = new TokenNameFinderCrossValidator(lang, entityType, trainParams, null);
                // the second argument of 'evaluate()' gives number of folds (n), i.e. number of times the training-testing will be run (with data splitting train:test = (n-1):1)
                evaluator.evaluate(sampleStream, nFolds);
                result = "F = " + decFormat.format(evaluator.getFMeasure().getFMeasure())
                        + " (Precision = " + decFormat.format(evaluator.getFMeasure().getPrecisionScore())
                        + ", Recall = " + decFormat.format(evaluator.getFMeasure().getRecallScore()) + ")";
                LOG.info("Validation: " + result);
            } catch (IOException ex) {
                LOG.error("Error while opening training: " + fileIn, ex);
                throw new RuntimeException("Error while training " + MODEL_NAME + " model " + modelDescr, ex);
            } catch (Exception ex) {
                LOG.error("Error while validating " + MODEL_NAME + " model.", ex);
                throw new RuntimeException("Error while training " + MODEL_NAME + " model " + modelDescr, ex);
            }
        } else {
            try (ObjectStream<String> lineStreamValidate = openFile(fileValidate); NameSampleDataStream sampleStreamValidate = new NameSampleDataStream(lineStreamValidate)) {
                TokenNameFinderEvaluator evaluator = new TokenNameFinderEvaluator(new NameFinderME((TokenNameFinderModel) model));
                evaluator.evaluate(sampleStreamValidate);
                result = "F = " + decFormat.format(evaluator.getFMeasure().getFMeasure())
                        + " (Precision = " + decFormat.format(evaluator.getFMeasure().getPrecisionScore())
                        + ", Recall = " + decFormat.format(evaluator.getFMeasure().getRecallScore()) + ")";
                LOG.info("Validation: " + result);
            } catch (IOException ex) {
                LOG.error("Error while opening training: " + fileValidate, ex);
                throw new RuntimeException("Error while training " + MODEL_NAME + " model " + modelDescr, ex);
            } catch (Exception ex) {
                LOG.error("Error while validating " + this.MODEL_NAME + " model.", ex);
            }
        }

        return result;
    }
}
