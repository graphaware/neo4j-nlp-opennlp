/*
 *
 *
 */
package com.graphaware.nlp.processor.opennlp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Properties;
import java.util.Map;
import java.text.DecimalFormat;

import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.model.BaseModel;

import com.graphaware.nlp.util.GenericModelParameters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author vla
 */
public class OpenNLPGenericModelTool {

  protected BaseModel model;
  protected TrainingParameters trainParams;
  protected final String modelDescr;
  protected final DecimalFormat decFormat;
  protected int nFolds;

  protected final String fileIn;
  protected ObjectStream<String> lineStream;
  protected String fileValidate;
  protected ObjectStream<String> lineStreamValidate;

  private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);


  public OpenNLPGenericModelTool(String file, String modelDescr) {
    this.model = null;
    this.fileValidate = null;
    this.nFolds = 10;
    this.fileIn = file;
    this.modelDescr = modelDescr;
    this.decFormat = new DecimalFormat("#0.00"); // for formating validation results with precision 2 decimals

    this.setDefParams();
    this.lineStream = this.openFile(this.fileIn);
    this.lineStreamValidate = this.openFile(this.fileValidate);
  }

  public OpenNLPGenericModelTool(String file, String modelDescr, Map<String, String> params) {
    this(file, modelDescr);
    this.setTrainingParameters(params);
  }

  // override this method in your child-class if you want different defaults
  protected void setDefParams() {
    this.trainParams = TrainingParameters.defaultParams();
  }

  private ObjectStream<String> openFile(String fileName) {
    if (fileName==null)
      return null;
    //ImprovisedInputStreamFactory dataIn = null;
    ObjectStream<String> lStream = null;
    try {
      ImprovisedInputStreamFactory dataIn = new ImprovisedInputStreamFactory(null, "", fileName);
      lStream = new PlainTextByLineStream(dataIn, "UTF-8");
    } catch (IOException ex) {
      LOG.error("Failure while opening file " + fileName, ex);
      throw new RuntimeException("Failure while opening file " + fileName, ex);
    }
    return lStream;
  }

  protected void setTrainingParameters(Map<String, String> params) {
    if (params==null) return;

    // first: set default training parameters
    this.setDefParams();

    // now add/override-by user-defined parameters
    if (params.containsKey(GenericModelParameters.TRAIN_ALG)) {
      this.trainParams.put(TrainingParameters.ALGORITHM_PARAM, params.get(GenericModelParameters.TRAIN_ALG)); // default: MAXENT
      LOG.info("Training parameter " + TrainingParameters.ALGORITHM_PARAM + " set to " + params.get(GenericModelParameters.TRAIN_ALG));
    }
    if (params.containsKey(GenericModelParameters.TRAIN_TYPE)) {
      this.trainParams.put(TrainingParameters.TRAINER_TYPE_PARAM, params.get(GenericModelParameters.TRAIN_TYPE));
      LOG.info("Training parameter " + TrainingParameters.TRAINER_TYPE_PARAM + " set to " + params.get(GenericModelParameters.TRAIN_TYPE));
    }
    if (params.containsKey(GenericModelParameters.TRAIN_CUTOFF)) {
      this.trainParams.put(TrainingParameters.CUTOFF_PARAM, params.get(GenericModelParameters.TRAIN_CUTOFF));
      LOG.info("Training parameter " + TrainingParameters.CUTOFF_PARAM + " set to " + params.get(GenericModelParameters.TRAIN_CUTOFF));
    }
    if (params.containsKey(GenericModelParameters.TRAIN_ITER)) {
      this.trainParams.put(TrainingParameters.ITERATIONS_PARAM, params.get(GenericModelParameters.TRAIN_ITER));
      LOG.info("Training parameter " + TrainingParameters.ITERATIONS_PARAM + " set to " + params.get(GenericModelParameters.TRAIN_ITER));
    }
    if (params.containsKey(GenericModelParameters.TRAIN_THREADS)) {
      this.trainParams.put(TrainingParameters.THREADS_PARAM, params.get(GenericModelParameters.TRAIN_THREADS));
      LOG.info("Training parameter " + TrainingParameters.THREADS_PARAM + " set to " + params.get(GenericModelParameters.TRAIN_THREADS));
    }
    if (params.containsKey(GenericModelParameters.VALIDATE_FOLDS)) {
      try {
        this.nFolds = Integer.parseInt(params.get(GenericModelParameters.VALIDATE_FOLDS));
      } catch (Exception ex) {
        LOG.warn("Wrong specification of argument " + GenericModelParameters.VALIDATE_FOLDS + ", using default.");
      }
      LOG.info("n-folds for crossvalidation set to %d.", this.nFolds);
    }
    if (params.containsKey(GenericModelParameters.VALIDATE_FILE)) {
      this.fileValidate = (String) params.get(GenericModelParameters.VALIDATE_FILE);
      LOG.info("Using valudation file " + params.get(GenericModelParameters.VALIDATE_FILE));
    }
  }

  protected void closeInputFiles() {
    try {
      if (this.lineStream!=null)
        this.lineStream.close();
    } catch (IOException ex) {
      LOG.warn("Attept to close input line-stream from source file " + this.fileIn + " failed.");
    }

    try {
      if (this.lineStreamValidate!=null)
        this.lineStreamValidate.close();
    } catch (IOException ex) {
      LOG.warn("Attept to close input line-stream from source file " + this.fileValidate + " failed.");
    }
  }

  public void saveModel(String file) {
    if (this.model==null) {
      LOG.error("Can't save training results to a " + file + ": model is null");
      return;
    }

    try {
      BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(file));
      this.model.serialize(modelOut);
      modelOut.close();
    } catch (IOException ex) {
      LOG.error("Error saving model to file " + file, ex);
      throw new RuntimeException("Error saving model to file " + file, ex);
    }

    //this.closeInputFile();

    return;
  }

  public BaseModel getModel() {
    return this.model;
  }


  class ImprovisedInputStreamFactory implements InputStreamFactory {
    private File inputSourceFile;
    private String inputSourceStr;

    ImprovisedInputStreamFactory(Properties properties, String property, String defaultValue) {
      this.inputSourceFile = null;
      this.inputSourceStr = defaultValue;
      if (properties!=null) this.inputSourceStr = properties.getProperty(property, defaultValue);
      try {
        if (this.inputSourceStr.startsWith("file://"))
          this.inputSourceFile = new File(new URI(this.inputSourceStr));
        else if (this.inputSourceStr.startsWith("/"))
          this.inputSourceFile = new File(this.inputSourceStr);
      } catch (Exception ex) {
        LOG.error("Error while loading model from " + this.inputSourceStr);
        throw new RuntimeException("Error while loading model from " + this.inputSourceStr);
      }
    }

    @Override
    public InputStream createInputStream() throws IOException {
      LOG.debug("Creating input stream from " + this.inputSourceFile.getPath());
      //return getClass().getClassLoader().getResourceAsStream(this.inputSourceFile.getPath());
      return new FileInputStream(this.inputSourceFile.getPath());
    }

    /*public void closeInputStream() {
      try {
        if (this.is!=null)
        this.is.close();
      } catch (IOException ex) {
        LOG.warn("Attept to close input stream failed.");
      }
    }*/
  }

}
