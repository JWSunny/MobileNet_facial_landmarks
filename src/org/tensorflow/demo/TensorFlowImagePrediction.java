package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.List;
import java.util.Vector;


public class TensorFlowImagePrediction implements Prediction {

    private static final String TAG = "TensorFlowImagePrediction";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private float[] newoutputs;
    private String[] outputNames;
    private boolean logStats = false;
    private TensorFlowInferenceInterface inferenceInterface;
    private TensorFlowImagePrediction() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Prediction createPrec(
            AssetManager assetManager,
            String modelFilename,
            int inputSize,
            String inputName,
            String outputName) {
        TensorFlowImagePrediction c = new TensorFlowImagePrediction();
        c.inputName = inputName;
        c.outputName = outputName;
        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);

        final int landmarks = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " landmarks, output layer size is " + landmarks);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
        int aa = c.intValues.length;
        c.floatValues = new float[inputSize * inputSize * 3];
        c.outputs = new float[landmarks];
        c.newoutputs = new float[landmarks];
        return c;
    }

    @Override
    public float[] recognizeBitmap(Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("predictionImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.

        // 因为输入是 128*128*3的图片；所以首先是将 bitmap---> 128*128的尺寸
        float scaleWidth = ((float) 128) / bitmap.getWidth();
        float scaleHeight = ((float) 128) / bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap newbitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // 主要代码修改--- cnn_facial_landmarks 预测
        // 图片的输入是 112,112,3的格式，并将图片数据转化成 float 类型
        bitmap.getPixels(intValues, 0, newbitmap.getWidth(), 0, 0, newbitmap.getWidth(), newbitmap.getHeight());
        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
            floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
            floatValues[i * 3 + 2] =  val & 0xFF;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        // Failed to run TensorFlow inference with inputs:[input_image_tensor], outputs:[logits/BiasAdd]
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();


        float new_scaleWidth = bitmap.getWidth() / ((float) 128) ;
        float new_scaleHeight = bitmap.getHeight() / ((float) 128) ;


        // bitmap 图中进行 点的标注
        // 预测的点是进行了归一化处理的，需进行还原操作；并实现在扣取人脸框中的标记；
        for(int i=0; i<outputs.length; i++){
            if(i%2==0){
                newoutputs[i] = outputs[i]*inputSize*new_scaleWidth;
            }else{
                newoutputs[i] = outputs[i]*inputSize*new_scaleHeight;
            }

        }

//        Canvas canvas = new Canvas(bitmap);
//        Paint paint = new Paint();
//        paint.setColor(Color.RED);
//        paint.setStrokeWidth((float) 5.0);
//        canvas.drawPoints(newoutputs, paint);

        Trace.endSection(); //// "recognizeImage"
        return newoutputs;
    }

    @Override
    public Bitmap convertBitmap(Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("predictionImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.

        // 因为输入是 128*128*3的图片；所以首先是将 bitmap---> 128*128的尺寸
        float scaleWidth = ((float) 128) / bitmap.getWidth();
        float scaleHeight = ((float) 128) / bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap newbitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // 主要代码修改--- cnn_facial_landmarks 预测
        // 图片的输入是 112,112,3的格式，并将图片数据转化成 float 类型
        bitmap.getPixels(intValues, 0, newbitmap.getWidth(), 0, 0, newbitmap.getWidth(), newbitmap.getHeight());
        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
            floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
            floatValues[i * 3 + 2] =  val & 0xFF;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        // Failed to run TensorFlow inference with inputs:[input_image_tensor], outputs:[logits/BiasAdd]
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();


        float new_scaleWidth = bitmap.getWidth() / ((float) 128) ;
        float new_scaleHeight = bitmap.getHeight() / ((float) 128) ;


        // bitmap 图中进行 点的标注
        // 预测的点是进行了归一化处理的，需进行还原操作；并实现在扣取人脸框中的标记；
        for(int i=0; i<outputs.length; i++){
            if(i%2==0){
                newoutputs[i] = outputs[i]*inputSize*new_scaleWidth;
            }else{
                newoutputs[i] = outputs[i]*inputSize*new_scaleHeight;
            }

        }

        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStrokeWidth((float) 5.0);
        canvas.drawPoints(newoutputs, paint);

        Trace.endSection(); //// "recognizeImage"
        return bitmap;
    }

    @Override
    public float[] processLocation(Bitmap bitmap, RectF rect) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("predictionImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.

        // 因为输入是 128*128*3的图片；所以首先是将 bitmap---> 128*128的尺寸

        Bitmap newbitmap = null;
        float newwidth;
        float newheight;

        float width = rect.right - rect.left;
        float height = rect.bottom - rect.top;
        float diff = (height - width)/2;
        int delta = Math.round(Math.abs(diff)/2);

        if(diff == 0){
            newwidth = rect.right - rect.left;
            newheight = rect.bottom - rect.top;
        }else if (diff>0){
            Log.i("fail diff>0","[*] fail diff>0");
            rect.left -= delta;
            rect.right += delta;
            if (diff%2==1){
                rect.right += 1;
            }
            newwidth = rect.right - rect.left;
            newheight = rect.bottom - rect.top;
        }else {
            Log.i("fail diff <0","[*] fail diff<0");
            rect.top -= delta;
            rect.bottom += delta;
            if(diff%2==1){
                rect.bottom += 1;
            }
            newwidth = rect.right - rect.left;
            newheight = rect.bottom - rect.top;
        }

        float scaleWidth = ((float) 128) / newwidth;
        float scaleHeight = ((float) 128) / newheight;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        newbitmap = Bitmap.createBitmap(bitmap, Math.round(rect.left), Math.round(rect.top), Math.round(newwidth), Math.round(newheight), matrix, true);


        // 主要代码修改--- cnn_facial_landmarks 预测
        // 图片的输入是 112,112,3的格式，并将图片数据转化成 float 类型
        bitmap.getPixels(intValues, 0, newbitmap.getWidth(), 0, 0, newbitmap.getWidth(), newbitmap.getHeight());
        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
            floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
            floatValues[i * 3 + 2] =  val & 0xFF;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        // Failed to run TensorFlow inference with inputs:[input_image_tensor], outputs:[logits/BiasAdd]
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();


        ////  128*128的 bitmap 继续转化为 人脸检测的位置 location（求取人脸框中，关键点的位置坐标）
        float new_scaleWidth = (rect.right - rect.left) / ((float) 128);
        float new_scaleHeight = (rect.bottom - rect.top) / ((float) 128) ;


//        Matrix new_matrix = new Matrix();
//        new_matrix.postScale(scaleWidth, scaleHeight);
//        Bitmap new_bitmap = Bitmap.createBitmap(newbitmap, 0, 0, 128, 128, new_matrix, true);


        // bitmap 图中进行 点的标注
        // 预测的点是进行了归一化处理的，需进行还原操作；并实现在扣取人脸框中的标记；
        for(int i=0; i<outputs.length; i++){
            if(i%2==0){
                newoutputs[i] = outputs[i]*inputSize*new_scaleWidth;
            }else{
                newoutputs[i] = outputs[i]*inputSize*new_scaleHeight;
            }

        }

//        Canvas canvas = new Canvas(bitmap);
//        Paint paint = new Paint();
//        paint.setColor(Color.RED);
//        paint.setStrokeWidth((float) 5.0);
//        canvas.drawPoints(newoutputs, paint);

        Trace.endSection(); //// "recognizeImage"
        return newoutputs;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

}
