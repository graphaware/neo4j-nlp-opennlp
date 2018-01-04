/*
 *
 *
 */
package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.processor.opennlp.OpenNLPPipeline;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Properties;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;
import java.util.Iterator;
import java.net.URI;

import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderCrossValidator;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DoccatCrossValidator;
import opennlp.tools.doccat.DocumentCategorizerEvaluator;
import opennlp.tools.doccat.DoccatModel;

import opennlp.tools.namefind.NameSample;

import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.eval.CrossValidationPartitioner;
import opennlp.tools.util.eval.FMeasure;
import opennlp.tools.util.FilterObjectStream;
//import opennlp.tools.util.eval.EvaluationMonitor;

import com.graphaware.nlp.util.GenericModelParameters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author vla
 */
public class SentimentModelTool extends OpenNLPGenericModelTool {

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    private static final String MODEL_NAME = "sentiment";
    private static final String DEFAULT_ITER = "30";
    private static final String DEFAULT_CUTOFF = "2";

    public SentimentModelTool(String fileIn, String modelDescr, String lang, Map<String, Object> params) {
        super(fileIn, modelDescr, lang, params);
    }

    public SentimentModelTool(String fileIn, String modelDescr, String lang) {
        this(fileIn, modelDescr, lang, null);
    }

    public SentimentModelTool() {
        super();
    }

    // here you can specify default parameters specific to this class
    @Override
    protected void setDefParams() {
        this.trainParams = TrainingParameters.defaultParams();
        this.trainParams.put(TrainingParameters.ITERATIONS_PARAM, DEFAULT_ITER);
        this.trainParams.put(TrainingParameters.CUTOFF_PARAM, DEFAULT_CUTOFF);
    }

    public void train() {
        LOG.info("Starting training of " + MODEL_NAME + " ...");
        try (ObjectStream<String> lineStream = openFile(fileIn); ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream)) {
            this.model = DocumentCategorizerME.train("en", sampleStream, trainParams, new DoccatFactory());
        } catch (IOException e) {
            LOG.error("IOError while training a custom " + MODEL_NAME + " model " + modelDescr, e);
            throw new RuntimeException("IOError while training a custom " + MODEL_NAME + " model " + this.modelDescr, e);
        }
    }

    public String validate() {
        LOG.info("Starting validation of " + MODEL_NAME + " ...");
        String result = "";
        if (this.fileValidate == null) {
            try (ObjectStream<String> lineStream = openFile(fileIn); ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream)) {
                DoccatCrossValidator evaluator = new DoccatCrossValidator(this.lang, this.trainParams, new DoccatFactory());
                // the second argument of 'evaluate()' gives number of folds (n): number of times the training-testing will be run (with data splitting train:test = (n-1):1)
                evaluator.evaluate(sampleStream, this.nFolds);
                result = "Accuracy = " + this.decFormat.format(evaluator.getDocumentAccuracy());
                LOG.info("Validation: " + result);
            } catch (IOException e) {
                LOG.error("Error while opening training file: " + fileIn, e);
                throw new RuntimeException("IOError while evaluating a " + MODEL_NAME + " model " + this.modelDescr, e);
            } catch (Exception ex) {
                LOG.error("Error while evaluating " + MODEL_NAME + " model.", ex);
            }
        } else {
            // Using a separate .test file provided by user
            result = test(this.fileValidate, new DocumentCategorizerME((DoccatModel) this.model));
        }

        return result;
    }

    public String test(String file, DocumentCategorizerME modelME) {
        LOG.info("Starting testing of " + MODEL_NAME + " ...");
        String result = "";
        try (ObjectStream<String> lineStream = openFile(file); ObjectStream<DocumentSample> sampleStreamValidate = new DocumentSampleStream(lineStream)) {
            //DocumentCategorizerEvaluator evaluator = new DocumentCategorizerEvaluator(new DocumentCategorizerME((DoccatModel) this.model));
            DocumentCategorizerEvaluator evaluator = new DocumentCategorizerEvaluator(modelME);
            evaluator.evaluate(sampleStreamValidate);
            result = "Accuracy = " + this.decFormat.format(evaluator.getAccuracy());
            LOG.info("Validation: " + result);
        } catch (IOException e) {
            LOG.error("Error while opening a test file: " + file, e);
            throw new RuntimeException("IOError while testing a " + MODEL_NAME + " model " + this.modelDescr, e);
        } catch (Exception ex) {
            LOG.error("Error while testing " + MODEL_NAME + " model.", ex);
        }
        return result;
    }
}
