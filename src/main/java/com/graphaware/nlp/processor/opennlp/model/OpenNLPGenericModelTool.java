/*
 *
 *
 */
package com.graphaware.nlp.processor.opennlp.model;

import com.graphaware.nlp.processor.opennlp.OpenNLPPipeline;
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

public class OpenNLPGenericModelTool {

    protected BaseModel model;
    protected TrainingParameters trainParams;
    protected final String modelDescr;
    protected final String lang;
    protected final DecimalFormat decFormat;
    protected int nFolds;

    protected final String fileIn;
    protected String fileValidate;

    private static final Logger LOG = LoggerFactory.getLogger(OpenNLPPipeline.class);

    public OpenNLPGenericModelTool(String file, String modelDescr, String lang) {
        this.fileValidate = null;
        this.fileIn = file;
        this.nFolds = 10;
        this.modelDescr = modelDescr;
        this.lang = lang;
        this.decFormat = new DecimalFormat("#0.00"); // for formating validation results with precision 2 decimals

        this.setDefParams();
    }

    public OpenNLPGenericModelTool(String file, String modelDescr, String lang, Map<String, Object> params) {
        this(file, modelDescr, lang);
        this.setTrainingParameters(params);
    }

    /*
     * This constructor needed for invoking test() method only (model is provided as an argument of train() )
     */
    public OpenNLPGenericModelTool() {
        this(null, null, null);
        this.model = null;
    }

    // override this method in your child-class if you want different defaults
    protected void setDefParams() {
        this.trainParams = TrainingParameters.defaultParams();
    }

    protected ObjectStream<String> openFile(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            LOG.error("File name is null or empty.");
            return null;
        }
        ObjectStream<String> lStream = null;
        try {
            ImprovisedInputStreamFactory dataIn = new ImprovisedInputStreamFactory(null, "", fileName);
            lStream = new PlainTextByLineStream(dataIn, "UTF-8");
        } catch (IOException ex) {
            LOG.error("Failure while opening file " + fileName, ex);
            throw new RuntimeException("Failure while opening file " + fileName, ex);
        }

        if (lStream == null)
            throw new RuntimeException("Failure while opening file " + fileName + ": input stream is null.");
        return lStream;
    }

    private void setTrainingParameters(Map<String, Object> params) {
        this.setDefParams();
        if (params == null || params.isEmpty()) {
            LOG.error("Map of parameters is null or empty. Using default values.");
            return;
        }

        // now add/override-by user-defined parameters
        if (params.containsKey(GenericModelParameters.TRAIN_ALG)) {
            String val = objectToString(params, GenericModelParameters.TRAIN_ALG);
            this.trainParams.put(TrainingParameters.ALGORITHM_PARAM, val); // default: MAXENT
            LOG.info("Training parameter " + TrainingParameters.ALGORITHM_PARAM + " set to " + val);
        }
        if (params.containsKey(GenericModelParameters.TRAIN_TYPE)) {
            String val = objectToString(params, GenericModelParameters.TRAIN_TYPE);
            this.trainParams.put(TrainingParameters.TRAINER_TYPE_PARAM, val);
            LOG.info("Training parameter " + TrainingParameters.TRAINER_TYPE_PARAM + " set to " + val);
        }
        if (params.containsKey(GenericModelParameters.TRAIN_CUTOFF)) {
            String val = objectToString(params, GenericModelParameters.TRAIN_CUTOFF);
            this.trainParams.put(TrainingParameters.CUTOFF_PARAM, val);
            LOG.info("Training parameter " + TrainingParameters.CUTOFF_PARAM + " set to " + val);
        }
        if (params.containsKey(GenericModelParameters.TRAIN_ITER)) {
            String val = objectToString(params, GenericModelParameters.TRAIN_ITER);
            this.trainParams.put(TrainingParameters.ITERATIONS_PARAM, val);
            LOG.info("Training parameter " + TrainingParameters.ITERATIONS_PARAM + " set to " + val);
        }
        if (params.containsKey(GenericModelParameters.TRAIN_THREADS)) {
            String val = objectToString(params, GenericModelParameters.TRAIN_THREADS);
            this.trainParams.put(TrainingParameters.THREADS_PARAM, val);
            LOG.info("Training parameter " + TrainingParameters.THREADS_PARAM + " set to " + val);
        }
        if (params.containsKey(GenericModelParameters.VALIDATE_FOLDS)) {
            this.nFolds = objectToInt(params, GenericModelParameters.VALIDATE_FOLDS);
            LOG.info("n-folds for crossvalidation set to %d.", this.nFolds);
        }
        if (params.containsKey(GenericModelParameters.VALIDATE_FILE)) {
            this.fileValidate = objectToString(params, GenericModelParameters.VALIDATE_FILE);
            LOG.info("Using valudation file " + fileValidate);
        }
    }

    private String objectToString(Map<String, Object> params, String key) {
        String result = null;
        if (params.get(key) instanceof String)
            result = (String) params.get(key);
        else if (params.get(key) instanceof Long)
            result = ((Long) params.get(key)).toString();
        else if (params.get(key) instanceof Integer)
            result = ((Integer) params.get(key)).toString();
        else
            throw new RuntimeException("Wrong format of parameter " + key);
        return result;
    }

    private int objectToInt(Map<String, Object> params, String key) {
        int result;
        if (params.get(key) instanceof String)
            result = Integer.parseInt((String) params.get(key));
        else if (params.get(key) instanceof Long)
            result = ((Long) params.get(key)).intValue();
        else if (params.get(key) instanceof Integer)
            result = ((Integer) params.get(key)).intValue();
        else
            throw new RuntimeException("Wrong format of parameter " + key);
        return result;
    }

    protected void closeInputFiles() {
//        try {
//            if (this.lineStream != null) {
//                this.lineStream.close();
//            }
//        } catch (IOException ex) {
//            LOG.warn("Attept to close input line-stream from source file " + this.fileIn + " failed.");
//        }
//
//        try {
//            if (this.lineStreamValidate != null) {
//                this.lineStreamValidate.close();
//            }
//        } catch (IOException ex) {
//            LOG.warn("Attept to close input line-stream from source file " + this.fileValidate + " failed.");
//        }
    }

    public void saveModel(String file) {
        if (this.model == null) {
            LOG.error("Can't save training results to a " + file + ": model is null");
            return;
        }
        try {
            LOG.info("Saving model to file: " + file);
            BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(file));
            this.model.serialize(modelOut);
            modelOut.close();
        } catch (IOException ex) {
            LOG.error("Error saving model to file " + file, ex);
            throw new RuntimeException("Error saving model to file " + file, ex);
        }

        //this.closeInputFile();
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
            if (properties != null) {
                this.inputSourceStr = properties.getProperty(property, defaultValue);
            }
            try {
                if (this.inputSourceStr.startsWith("file://")) {
                    this.inputSourceFile = new File(new URI(this.inputSourceStr.replace("file://", "")));
                } else if (this.inputSourceStr.startsWith("/")) {
                    this.inputSourceFile = new File(this.inputSourceStr);
                }
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
