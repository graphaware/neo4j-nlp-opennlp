/*
 *
 *
 */
package com.graphaware.nlp.processor.opennlp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Map;
import java.util.HashMap;
import java.net.URI;

import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderCrossValidator;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSampleDataStream;

import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.eval.FMeasure;
//import opennlp.tools.util.eval.EvaluationMonitor;

import com.graphaware.nlp.util.GenericModelParameters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author vla
 */
public class NERModelTool extends OpenNLPGenericModelTool {

  private ObjectStream<NameSample> sampleStream;
  private String entityType;
  private final String lang;
  private static final String myName = "NER";

  private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);


  public NERModelTool(String fileIn, String modelDescr, String lang, Map<String, String> params) {
    super(fileIn, modelDescr, params);
    this.lang = lang;
    this.entityType = null; // train only specific named entity; null = train all entities present in the training set
    if (params!=null) {
      if (params.containsKey(GenericModelParameters.TRAIN_ENTITYTYPE))
        this.entityType = params.get(GenericModelParameters.TRAIN_ENTITYTYPE);
    }
  }

  public NERModelTool(String fileIn, String modelDescr, String lang) {
    this(fileIn, modelDescr, lang, null);
  }

  public void train() {
    LOG.info("Starting training of " + this.myName + " ...");
    this.sampleStream = new NameSampleDataStream(this.lineStream);
    try {
      this.model = NameFinderME.train(this.lang, this.entityType, this.sampleStream, this.trainParams, new TokenNameFinderFactory());
    } catch (Exception ex) {
      LOG.error("Error while training " + this.myName + " model " + this.modelDescr);
      throw new RuntimeException("Error while training " + this.myName + " model " + this.modelDescr, ex);
    }
  }

  public String validate() {
    LOG.info("Starting validation of " + this.myName + " ...");
    String result = "";
    //List<EvaluationMonitor<NameSample>> listeners = new LinkedList<EvaluationMonitor<NameSample>>();
    try {
      if (this.sampleStream==null)
        this.sampleStream = new NameSampleDataStream(this.lineStream);
      // #1 Using CrossValidator
      TokenNameFinderCrossValidator evaluator = new TokenNameFinderCrossValidator(this.lang, this.entityType, this.trainParams, null, null, null);
      // the second argument of 'evaluate()' gives number of folds (n), i.e. number of times the training-testing will be run (with data splitting train:test = (n-1):1)
      evaluator.evaluate(this.sampleStream, this.nFolds);
      result = "F = " + this.decFormat.format(evaluator.getFMeasure().getFMeasure()) 
                    + " (Precision = " + this.decFormat.format(evaluator.getFMeasure().getPrecisionScore())
                    + ", Recall = " + this.decFormat.format(evaluator.getFMeasure().getRecallScore()) + ")";
      LOG.info("Validation: " + result);
    } catch (Exception ex) {
      LOG.error("Error while validating " + this.myName + " model.");
      ex.printStackTrace();
    }

    // #2 Splitting training file into train and test manually


    // #3 Using a separate .test file provided by user
    /*try {
      ImprovisedInputStreamFactory testdataIn = new ImprovisedInputStreamFactory(null, "", fileTest);
      ObjectStream<String> testlineStream = new PlainTextByLineStream(testdataIn, "UTF-8");
      ObjectStream<NameSample> testsampleStream = new NameSampleDataStream(testlineStream);
      TokenNameFinderEvaluator evaluator = new TokenNameFinderEvaluator(new NameFinderME(model));
      evaluator.evaluate(testsampleStream);
      LOG.info("Validation: F = " + evaluator.getFMeasure().getFMeasure() + " (Precision = " + evaluator.getFMeasure().getPrecisionScore() + ", Recall = " + evaluator.getFMeasure().getRecallScore() + ")");
      } catch (Exception ex) {
      LOG.error("Error while validating " + alg, ex);
      ex.printStackTrace();
    }*/
    return result;
  }

  public void close() {
    try {
      if (sampleStream!=null)
        sampleStream.close();
    } catch (IOException ex) {
      LOG.warn("Attempt to close sample-stream from source file " + this.fileIn + " failed.");
    }
    this.closeInputFile();
  }

}


