import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

// 2DO: compile as .jar

class K_api {

    static K_base base = new K_base();

    K_api(){ }


    static Object make_model(String model_type, int in_size, int[] hidden_sizes, int out_size) {

        if (model_type.equals("gru")) return new GRU(in_size,  hidden_sizes, out_size);
        if (model_type.equals("lstm")) return new LSTM(in_size, hidden_sizes, out_size);
        else return null;

    }

//    static ArrayList<ArrayList<Double[][]>> train_model(ArrayList<ArrayList<Double[][]>> model, ArrayList<ArrayList<Double[][]>> data, int batch_size) {
//
//        int hm_batches = data.size() / batch_size;
//        int hm_leftover = data.size() % batch_size;
//
//        for (int i = 0; i < hm_batches; i++) {
//
//
//
//        }
//
//
//    }

    static ArrayList<ArrayList<Double[][]>> load(String source) {

        // todo : do.
        return null;

    }

    static void save(ArrayList<ArrayList<Double[][]>> model, String destination) {

        // todo : do.

    }

    private static class R implements Callable<ArrayList<Double[][]>>{

        GRU model;
        ArrayList<Double[][]> input;
        ArrayList<Double[][]> response;

        R(GRU model,  ArrayList<Double[][]>[] data) {

            this.model = model;
            this.input = data[0];
            this.response = data[1];

        }

        @Override
        public ArrayList<Double[][]> call() {

            ArrayList<Double[][]> output = model.respond_to(input);

            double loss = 0;

            for (int i = 0; i < response.size(); i++)

                loss += base.sum(base.cross_entropy(response.get(i), output.get(i)));


            // TODO : return batch_grads

            return null;

        }

    }

    static int hm_cores = Runtime.getRuntime().availableProcessors();
    static ExecutorService pool = Executors.newFixedThreadPool(hm_cores);

    static ArrayList<Double[][]>[] batch_process(GRU model, ArrayList<ArrayList<Double[][]>[]> batch) {

        int batch_size = batch.size();

        // Future<ArrayList<Double[][]>>[] promises = new Future[batch_size];

//        int ctr = -1;
//        for (ArrayList<Double[][]> data : batch) {
//            ctr++;
//
//            promises[ctr] = pool.submit(new R(model, data));
//
//        }

        ArrayList<R> tasks = new ArrayList<>();

        for (ArrayList<Double[][]>[] data : batch)

            tasks.add(new R(model, data));

        try {

            List<Future<ArrayList<Double[][]>>> promises = pool.invokeAll(tasks);

            ArrayList<Double[][]>[] batch_grads = new ArrayList[batch_size];

            for (Future promise : promises)

                while(!promise.isDone()) {System.out.println("Empty promises..");}

            for (int i = 0; i < batch_size; i++)

                try { batch_grads[i] = promises.get(i).get(); }
                catch (Exception e) { e.printStackTrace(); }

            return batch_grads;


        } catch (Exception e) { e.printStackTrace(); return null; }

    }

    static private boolean is_done(List<Future<ArrayList<Double[][]>>> promises) {

        for (Future promise : promises)

            if (promise != null)

                try {
                    if (promise.get() == null) return false;
                }
                catch (InterruptedException e) { e.printStackTrace(); return false; }
                catch (ExecutionException e) { e.printStackTrace(); }

        return true;

    }
    
}

class GRU {

    K_base base = new K_base();

    private ArrayList<ArrayList<Double[][]>> layers = new ArrayList<>();
    private ArrayList<Double[][]> states = new ArrayList<>();


    GRU(int in_size, int[] hidden_sizes, int out_size) {

        int hm_layers = hidden_sizes.length+1;

        for (int l = 0; l < hm_layers; l++) {

            ArrayList<Double[][]> layer = new ArrayList<>();
            int i,o;

            if      (l == 0)           { i = in_size ; o = hidden_sizes[0]; }
            else if (l == hm_layers-1) { i = hidden_sizes[l-1] ; o = out_size; }
            else                       { i = hidden_sizes[l-1] ; o = hidden_sizes[l]; }

            // @params:
            //"wf1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wf2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wk1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wk2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wi" : randn(in_size, layer_size,    requires_grad=True, dtype=float32),

            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));

            layers.add(layer);
            states.add(base.zeros(1,o));

        }

    }

    private void zero_states() {

        int ctr = -1;
        for (ArrayList<Double[][]> layer : layers) {
            ctr++;

            int layer_size = base.size(layer.get(0))[1];
            Double[][] zero_state = base.zeros(1, layer_size);
            states.set(ctr, zero_state);

        }

    }

    private Double[][] propogate_layer(ArrayList<Double[][]> layer, Double[][] in, int layer_ctr) {

        Double[][] state  = states.get(layer_ctr);
        Double[][] focus  = base.sigm(base.add(base.matmul(in,layer.get(0)),base.matmul(state,layer.get(1))));
        Double[][] keep   = base.sigm(base.add(base.matmul(in,layer.get(2)),base.matmul(state,layer.get(3))));
        Double[][] interm = base.tanh(base.add(base.matmul(in,layer.get(4)),base.mul(state,focus)));
        Double[][] new_state = base.add(base.mul(keep,interm),base.mul(base.sub_scalar(1,keep),state));
        states.set(layer_ctr, new_state);

        return new_state;

    }

    private Double[][] propogate_model(Double[][] in) {


        int ctr = -1;
        for (ArrayList<Double[][]> layer : layers) {
            ctr++;

            in = propogate_layer(layer, in, ctr);

        }

        return in;

    }

    ArrayList<Double[][]> respond_to(ArrayList<Double[][]> sequence) {

        ArrayList<Double[][]> response = new ArrayList<>();

        for (Double[][] timestep : sequence)

            response.add(propogate_model(timestep));

        this.zero_states();

        return response;

    }

}

class LSTM {

    K_base base = new K_base();

    private ArrayList<ArrayList<Double[][]>> layers = new ArrayList<>();
    private ArrayList<Double[][]> states = new ArrayList<>();


    LSTM(int in_size, int[] hidden_sizes, int out_size) {

        int hm_layers = hidden_sizes.length+1;

        for (int l = 0; l < hm_layers; l++) {

            ArrayList<Double[][]> layer = new ArrayList<>();
            int i, o;

            if      (l == 0)           { i = in_size ; o = hidden_sizes[0]; }
            else if (l == hm_layers-1) { i = hidden_sizes[l-1] ; o = hidden_sizes[l]; }
            else                       { i = hidden_sizes[l] ; o = out_size; }

            // @params:
            //"wf1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wf2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wk1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wk2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"wi1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"wi2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),
            //"ws1": randn(in_size, layer_size,    requires_grad=True, dtype=float32),
            //"ws2": randn(layer_size, layer_size, requires_grad=True, dtype=float32),

            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));
            layer.add(base.randn(i,o));
            layer.add(base.randn(o,o));

            layers.add(layer);
            states.add(base.zeros(1,o));

        }

    }

    private void zero_states() {

        int ctr = -1;
        for (ArrayList<Double[][]> layer : layers) {
            ctr++;

            int layer_size = base.size(layer.get(0))[1];
            Double[][] zero_state = base.zeros(1, layer_size);
            states.set(ctr, zero_state);

        }

    }

    private Double[][] propogate_layer(ArrayList<Double[][]> layer, Double[][] in, int layer_ctr) {

        Double[][] state  = states.get(layer_ctr);
        Double[][] forget = base.sigm(base.add(base.matmul(in,layer.get(0)),base.matmul(state,layer.get(1))));
        Double[][] keep   = base.sigm(base.add(base.matmul(in,layer.get(2)),base.matmul(state,layer.get(3))));
        Double[][] interm = base.tanh(base.add(base.matmul(in,layer.get(4)),base.mul(state,layer.get(5))));
        Double[][] show   = base.sigm(base.add(base.matmul(in,layer.get(6)),base.matmul(state,layer.get(7))));
        Double[][] new_state = base.add(base.mul(keep,interm),base.mul(forget,state));
        states.set(layer_ctr, new_state);
        Double[][] out = base.mul(show,base.tanh(new_state));

        return out;

    }

    private Double[][] propogate_model(Double[][] in) {

        int ctr = -1;
        for (ArrayList<Double[][]> layer : layers) {
            ctr++;

            in = propogate_layer(layer, in, ctr);

        }

        return in;

    }

    ArrayList<Double[][]> respond_to(ArrayList<Double[][]> sequence) {

        ArrayList<Double[][]> response = new ArrayList<>();

        for (Double[][] timestep : sequence)

            response.add(propogate_model(timestep));

        zero_states();

        return response;

    }

}

