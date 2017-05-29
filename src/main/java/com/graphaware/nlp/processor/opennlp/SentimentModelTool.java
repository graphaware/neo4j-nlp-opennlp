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

  private ObjectStream<DocumentSample> sampleStream;
  private final String lang;
  private static final String myName = "sentiment";

  private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);


  public SentimentModelTool(String fileIn, String modelDescr, String lang, Map<String, String> params) {
    super(fileIn, modelDescr, params);

    this.lang = lang;
    this.sampleStream = null;
  }

  public SentimentModelTool(String fileIn, String modelDescr, String lang) {
    this(fileIn, modelDescr, lang, null);
  }

  // here you can specify default parameters specific to this class
  @Override
  protected void setDefParams() {
    this.trainParams = TrainingParameters.defaultParams();
    this.trainParams.put(TrainingParameters.ITERATIONS_PARAM, "30");
    this.trainParams.put(TrainingParameters.CUTOFF_PARAM, "2");
  }

  public void train() {
    LOG.info("Starting training of " + this.myName + " ...");
    try {
      this.sampleStream = new DocumentSampleStream(this.lineStream);
      this.model = DocumentCategorizerME.train("en", this.sampleStream, this.trainParams, new DoccatFactory());
    } catch (IOException e) {
      LOG.error("IOError while training a custom " + this.myName + " model " + this.modelDescr);
      //e.printStackTrace();
      throw new RuntimeException("IOError while training a custom " + this.myName + " model " + this.modelDescr, e);
    }
  }

  public String validate() {
    LOG.info("Starting validation of " + this.myName + " ...");
    String result = "";
    try {
      if (this.sampleStream==null)
        this.sampleStream = new DocumentSampleStream(this.lineStream);
      DoccatCrossValidator evaluator = new DoccatCrossValidator(this.lang, this.trainParams, new DoccatFactory(), null);
      // the second argument of 'evaluate()' gives number of folds (n): number of times the training-testing will be run (with data splitting train:test = (n-1):1)
      evaluator.evaluate(this.sampleStream, this.nFolds);
      result = "Accuracy = " + this.decFormat.format(evaluator.getDocumentAccuracy());
      LOG.info("Validation: " + result);
    } catch (Exception ex) {
      LOG.error("Error while validating " + this.myName + " model.");
      ex.printStackTrace();
    }
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


