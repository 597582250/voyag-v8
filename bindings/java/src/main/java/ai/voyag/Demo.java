package ai.voyag;

import ai.onnxruntime.OrtException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;
import java.util.Map;

/** CLI demo for the Java binding.
 *
 * <p>Usage:
 * <pre>
 *   mvn -q compile exec:java -Dexec.args="&lt;model-dir&gt; &lt;utterance&gt;"
 * </pre>
 */
public final class Demo {

    public static void main(String[] args) throws IOException, OrtException {
        if (args.length < 2) {
            System.err.println("usage: mvn -q compile exec:java -Dexec.args=\"<model-dir> <utterance>\"");
            System.exit(2);
        }
        try (VoyagPredictor predictor = VoyagPredictor.create(args[0])) {
            Map<String, Object> result = predictor.predict(args[1], null);
            Gson gson = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();
            System.out.println(gson.toJson(result));
        }
    }
}
