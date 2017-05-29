import java.util.ArrayList;
import java.util.List;
import java.util.*;
import java.lang.Error;

public class ANN {

    //INITIALIZE WEIGTHS FROM A RANDOM DISTRIBUTION
    private static Double[] weightInit(int numberOfWeights) {
        List<Double> weightsArrayList = new ArrayList<Double>();

        for (int i = 0; i < numberOfWeights; i++) {
            weightsArrayList.add(Math.random());
        }
        Double[] weights = new Double[weightsArrayList.size()];
        weights = weightsArrayList.toArray(weights);

        return weights;
    }

    //RELU ACTIVATION FUNCTION
    private static double relu(double data) {
        double relu = Math.max(0, data);

        return relu;
    }

    //CROSS ENTROPY LOSS FUNCTION
    private static double cross_entropy(double[] prediction, double[] target) {
        double totalLoss = 0;
        for (int i = 0; i < prediction.length; i++) {
            totalLoss += target[i] * (-Math.log(prediction[i]));
        }

        double averageLoss = totalLoss / prediction.length;
        return averageLoss;
    }

    //ACTIVATION FUNCTION: RELU FOR HIDDEN LAYERS, SOFTMAX FOR OUTPUT LAYER
    private static double activation(double[] data, String activationFunction, Double[] weights) {
        double weightMultiplication = 0;

        for (int i = 0; i < data.length; i++) {
            weightMultiplication += data[i] * weights[i];
        }

        if (activationFunction == "relu") {
            double activation = relu(weightMultiplication);

            return activation;
        } else {
            throw new Error("No activation function was declared");
        }
    }

    private static Double[] hiddenLayer(int numberOfUnits, String activationFunction, double[] data) {
        List<Double> unitsArrayList = new ArrayList<Double>();

        for (int i = 0; i < numberOfUnits; i++) {
            Double[] weights = weightInit(7);
            unitsArrayList.add(activation(data, activationFunction, weights));
        }

        Double[] units = new Double[unitsArrayList.size()];
        units = unitsArrayList.toArray(units);

        return units;
    }

    private static Double[] outputLayer(int numberOfUnits, double[] data) {

        //SOFTMAX ACTIVATION FUNCTION FOR OUTPUT PREDICTION
        List<Double> outputUnitsArrayList = new ArrayList<Double>();

        for (int i = 0; i < data.length; i++) {
            double exponentPlaceholder = 0;

           // data.forEach( (value) ->  Math.exp(value));
            for(int k = 0; k < data.length; k++) {
                exponentPlaceholder += Math.exp(data[k]);
            }

            double softmax = Math.exp(data[i]) / exponentPlaceholder ;
            outputUnitsArrayList.add(softmax);
        }

        Double[] output = new Double[outputUnitsArrayList.size()];
        output = outputUnitsArrayList.toArray(output);

        return output;
    }

    public static void main(String[] args) {
        double[] predictions = {0.38, 0.5, 0.9};
        double[] targets = {1, 1, 1};
        double[] data = {10, 1, 2, 1, 4, 5,4};
        Double[] weights = weightInit(7);
        System.out.println(cross_entropy(predictions, targets));
        System.out.println(activation(data, "relu", weights));
        System.out.println(hiddenLayer(3, "relu", data)[0]);
        System.out.println(hiddenLayer(3, "relu", data)[1]);
        System.out.println(outputLayer(2, data)[0]);
        System.out.println(outputLayer(2, data)[1]);
    }
}

