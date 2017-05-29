import java.util.ArrayList;
import java.util.List;
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

    //RELU ACTIVATION FUNCTION WITH UPPER BOUND FOR ACTIVATION
    private static double relu(double data) {
        double relu = Math.max(0, data/(data*data));

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
    private static double activation(Double[] data, String activationFunction, Double[] weights) {
        double weightMultiplication = 0;

        for (int i = 0; i < data.length; i++) {
            weightMultiplication += data[i] * weights[i];
        }

        if (activationFunction == "relu") {
            double activation = relu(weightMultiplication);

            return activation;
        }
        else {
            throw new Error("No activation function was declared");
        }
    }

    //HIDDEN LAYER: COMPUTES ACTIVATIONS OF ALL UNITS THE LAYER
    private static Double[] hiddenLayer(int numberOfUnits, String activationFunction, Double[] data) {
        List<Double> unitsArrayList = new ArrayList<Double>();

        for (int i = 0; i < numberOfUnits; i++) {
            Double[] weights = weightInit(data.length);
            unitsArrayList.add(activation(data, activationFunction, weights));
        }

        Double[] units = new Double[unitsArrayList.size()];
        units = unitsArrayList.toArray(units);

        return units;
    }

    private static Double[] outputLayer(int numberOfUnits, Double[] data) {
        List<Double> unitsArrayList = new ArrayList<Double>();

        //MULTIPLY WEIGHTS WITH INPUT DATA
        for (int i = 0; i < numberOfUnits; i++) {
            Double[] weights = weightInit(data.length);
            double weightMultiplication = 0;

            for (int j = 0; j < data.length; j++) {
                weightMultiplication += data[j] * weights[j];
            }
            unitsArrayList.add(weightMultiplication);
        }

        Double[] units = new Double[unitsArrayList.size()];
        units = unitsArrayList.toArray(units);

        //SOFTMAX ACTIVATION FUNCTION FOR OUTPUT PREDICTION
        List<Double> outputUnitsArrayList = new ArrayList<Double>();

        for (int i = 0; i < numberOfUnits; i++) {
            double exponentPlaceholder = 0;

           // data.forEach( (value) ->  Math.exp(value));
            for(int k = 0; k < units.length; k++) {
                exponentPlaceholder += Math.exp(units[k]);
                System.out.println(units[k]);
            }

            double softmax = Math.exp(units[i]) / exponentPlaceholder ;
            outputUnitsArrayList.add(softmax);
        }

        Double[] output = new Double[outputUnitsArrayList.size()];
        output = outputUnitsArrayList.toArray(output);

        return output;
    }

    public static void main(String[] args) {
        Double[] inputLayer = {0., 1., 0.5, 0.4, 0.9, 0.2,0.4};
        Double[] hiddenLayer1 = hiddenLayer(10, "relu", inputLayer);
        Double[] hiddenLayer2 = hiddenLayer(10, "relu", hiddenLayer1);
        Double[] outputs = outputLayer(2, hiddenLayer2);

        System.out.println(outputs[0]);
        System.out.println(outputs[1]);
    }
}

