import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

// 2DO: compile as .jar

class K_neural {


    K_base base = new K_base();

    int hm_cores = Runtime.getRuntime().availableProcessors();
    ExecutorService pool = Executors.newFixedThreadPool(hm_cpu);


    static Object make_model(String model_type, int in_size, ArrayList<Integer> hidden_sizes, int out_size) {

        if (model_type.equals("gru")) return new GRU(int in_size, ArrayList<Integer> hidden_sizes, int out_size);
        if (model_type.equals("lstm")) return new LSTM(int in_size, ArrayList<Integer> hidden_sizes, int out_size);

    }

    static LOAD(String source) {

        // todo : do.

    }

    static SAVE(String destination) {

        // todo : do.

    }

    private class R implements Callable<GRU, ArrayList<Double[][]>>{

        GRU model;
        ArrayList<Double[][]> sequence;

        R(model,  sequence) {

            this.model = model;
            this.sequence = sequence;

        }

        ArrayList<Double[][]> call() {

            return model.respond_to(sequence);

        }

    }

    static ArrayList<ArrayList<Double[][]>> batch_response(GRU model, ArrayList<ArrayList<Double[][]>> batch) {

        int batch_size = batch.size();

        Future<ArrayList<Double[][]>>[] promises = new Future[batch_size];

        ctr = -1;
        for (ArrayList<Double[][]> data : batch) {
            ctr++;

            promises[ctr] = pool.submit(new R(model, data));

        }

        ArrayList<Double[][]>[] responses = new ArrayList<Double[][]>[batch_size];

        for (int i = 0; i < batch_size; i++)

            responses[i] = promises.get(i).get();

        return responses;

    }
    
}

