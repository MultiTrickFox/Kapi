package k.kapi;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import java.util.concurrent.*;


class K_Layer {


    static class Dense {


        K_Tensor w;

        String act_fn;


        Dense(int incoming_size, int outgoing_size, String act_fn) {

            this.w = K_Tensor.randn(incoming_size, outgoing_size);
            this.w.requires_grad = true;
            this.act_fn = act_fn;

        }

        Dense(Dense layer) {

            this.w = new K_Tensor(layer.w.matrix);
            this.w.requires_grad = true;
            this.act_fn = layer.act_fn;

        }


    }

    static K_Tensor propogate(Dense layer, K_Tensor incoming) {

        K_Tensor out = K_Tensor.matmul(incoming, layer.w);

        switch (layer.act_fn) {

            case "sigm":
                return K_Tensor.sigm(out);

            case "tanh":
                return K_Tensor.tanh(out);

            case "elu":
                return K_Tensor.elu(out);

            default:
                System.out.println("Available act_fn params: sigm/tanh/elu, not " + layer.act_fn);
                return out;

        }

    }

    static void clear_grad(Dense layer) {

        layer.w.grad = K_Math.zeros(K_Math.size(layer.w.grad));

    }


    static class LSTM {


        K_Tensor wf1, wf2;
        K_Tensor wk1, wk2;
        K_Tensor wi1, wi2;
        K_Tensor ws1, ws2;

        K_Tensor state;


        LSTM(int incoming_size, int outgoing_size) {

            this.wf1 = K_Tensor.randn(incoming_size, outgoing_size); this.wf1.requires_grad = true;
            this.wf2 = K_Tensor.randn(outgoing_size, outgoing_size); this.wf2.requires_grad = true;

            this.wk1 = K_Tensor.randn(incoming_size, outgoing_size); this.wk1.requires_grad = true;
            this.wk2 = K_Tensor.randn(outgoing_size, outgoing_size); this.wk2.requires_grad = true;

            this.wi1 = K_Tensor.randn(incoming_size, outgoing_size); this.wi1.requires_grad = true;
            this.wi2 = K_Tensor.randn(outgoing_size, outgoing_size); this.wi2.requires_grad = true;

            this.ws1 = K_Tensor.randn(incoming_size, outgoing_size); this.ws1.requires_grad = true;
            this.ws2 = K_Tensor.randn(outgoing_size, outgoing_size); this.ws2.requires_grad = true;

            this.state = K_Tensor.zeros(1, outgoing_size);

        }

        LSTM(LSTM layer) {

            this.wf1 = new K_Tensor(layer.wf1.matrix); this.wf1.requires_grad = true;
            this.wf2 = new K_Tensor(layer.wf2.matrix); this.wf2.requires_grad = true;

            this.wk1 = new K_Tensor(layer.wk1.matrix); this.wk1.requires_grad = true;
            this.wk2 = new K_Tensor(layer.wk2.matrix); this.wk2.requires_grad = true;

            this.wi1 = new K_Tensor(layer.wi1.matrix); this.wi1.requires_grad = true;
            this.wi2 = new K_Tensor(layer.wi2.matrix); this.wi2.requires_grad = true;

            this.ws1 = new K_Tensor(layer.ws1.matrix); this.ws1.requires_grad = true;
            this.ws2 = new K_Tensor(layer.ws2.matrix); this.ws2.requires_grad = true;

            this.state = K_Tensor.zeros(K_Tensor.size(layer.state));

        }


    }

    static K_Tensor propogate(LSTM layer, K_Tensor incoming) {

        K_Tensor forget = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wf1),K_Tensor.matmul(layer.state , layer.wf2)));
        K_Tensor keep   = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wk1),K_Tensor.matmul(layer.state , layer.wk2)));
        K_Tensor interm = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wi1),K_Tensor.matmul(layer.state , layer.wi2)));
        K_Tensor show   = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.ws1),K_Tensor.matmul(layer.state , layer.ws2)));

        layer.state = K_Tensor.add(K_Tensor.mul(forget, layer.state), K_Tensor.mul(keep, interm));

        return K_Tensor.mul(show, K_Tensor.tanh(layer.state));

    }

    static Float[][][] collect_grad(LSTM layer) {

        return new Float[][][]{

                layer.wf1.grad, layer.wf2.grad,
                layer.wk1.grad, layer.wk2.grad,
                layer.wi1.grad, layer.wi2.grad,
                layer.ws1.grad, layer.ws2.grad,

        };

    }

    static void apply_grad(LSTM layer, Float[][][] layer_grads) {

        layer.wf1.grad = K_Math.add(layer.wf1.grad, layer_grads[0]);
        layer.wf2.grad = K_Math.add(layer.wf2.grad, layer_grads[1]);

        layer.wk1.grad = K_Math.add(layer.wk1.grad, layer_grads[2]);
        layer.wk2.grad = K_Math.add(layer.wk2.grad, layer_grads[3]);

        layer.wi1.grad = K_Math.add(layer.wi1.grad, layer_grads[4]);
        layer.wi2.grad = K_Math.add(layer.wi2.grad, layer_grads[5]);

        layer.ws1.grad = K_Math.add(layer.ws1.grad, layer_grads[6]);
        layer.ws2.grad = K_Math.add(layer.ws2.grad, layer_grads[7]);

    }

    static void learn_from_grad(LSTM layer, float learning_rate) {

        layer.wf1.matrix = K_Math.sub(layer.wf1.matrix, K_Math.mul_scalar(learning_rate, layer.wf1.grad));
        layer.wf2.matrix = K_Math.sub(layer.wf2.matrix, K_Math.mul_scalar(learning_rate, layer.wf2.grad));

        layer.wk1.matrix = K_Math.sub(layer.wk1.matrix, K_Math.mul_scalar(learning_rate, layer.wk1.grad));
        layer.wk2.matrix = K_Math.sub(layer.wk2.matrix, K_Math.mul_scalar(learning_rate, layer.wk2.grad));

        layer.wi1.matrix = K_Math.sub(layer.wi1.matrix, K_Math.mul_scalar(learning_rate, layer.wi1.grad));
        layer.wi2.matrix = K_Math.sub(layer.wi2.matrix, K_Math.mul_scalar(learning_rate, layer.wi2.grad));

        layer.ws1.matrix = K_Math.sub(layer.ws1.matrix, K_Math.mul_scalar(learning_rate, layer.ws1.grad));
        layer.ws2.matrix = K_Math.sub(layer.ws2.matrix, K_Math.mul_scalar(learning_rate, layer.ws2.grad));

    }

    static void clear_grad(LSTM layer) {

        layer.wf1.grad = K_Math.zeros(K_Math.size(layer.wf1.grad));
        layer.wf2.grad = K_Math.zeros(K_Math.size(layer.wf2.grad));

        layer.wk1.grad = K_Math.zeros(K_Math.size(layer.wk1.grad));
        layer.wk2.grad = K_Math.zeros(K_Math.size(layer.wk2.grad));

        layer.wi1.grad = K_Math.zeros(K_Math.size(layer.wi1.grad));
        layer.wi2.grad = K_Math.zeros(K_Math.size(layer.wi2.grad));

        layer.ws1.grad = K_Math.zeros(K_Math.size(layer.ws1.grad));
        layer.ws2.grad = K_Math.zeros(K_Math.size(layer.ws2.grad));

    }

    static void clear_state(LSTM layer) {

        layer.state = K_Tensor.zeros(1, K_Tensor.size(layer.wf2,1));

    }


    // extras

    static void xavierize(Dense layer) {

        float xav_val = (float) Math.sqrt(6.0 / (K_Math.size(layer.w.matrix,0)+K_Math.size(layer.w.matrix,1)));

        layer.w.matrix = K_Math.mul_scalar(xav_val, layer.w.matrix);

    }

    static void xavierize(LSTM layer) {

        float xav_val = (float) Math.sqrt(6.0 / (K_Math.size(layer.wf1.matrix,0)+K_Math.size(layer.wf1.matrix,1)));

        layer.wf1.matrix = K_Math.mul_scalar(xav_val, layer.wf1.matrix);
        layer.wf2.matrix = K_Math.mul_scalar(xav_val, layer.wf2.matrix);

        layer.wk1.matrix = K_Math.mul_scalar(xav_val, layer.wk1.matrix);
        layer.wk2.matrix = K_Math.mul_scalar(xav_val, layer.wk2.matrix);

        layer.wi1.matrix = K_Math.mul_scalar(xav_val, layer.wi1.matrix);
        layer.wi2.matrix = K_Math.mul_scalar(xav_val, layer.wi2.matrix);

        layer.ws1.matrix = K_Math.mul_scalar(xav_val, layer.ws1.matrix);
        layer.ws2.matrix = K_Math.mul_scalar(xav_val, layer.ws2.matrix);

    }


}


class K_Model {


    // FeedForward

    static List<Object> FeedForward(int in_size, int[] hidden_sizes, int out_size, String activation_fn) {

        List<Object> model = new ArrayList<>();

        int ctr = -1;
        for (int hidden_size : hidden_sizes) {
            ctr++;

            if (ctr == 0)

                model.add(new K_Layer.Dense(in_size, hidden_size, activation_fn));

            else

                model.add(new K_Layer.Dense(hidden_sizes[ctr - 1], hidden_size, activation_fn));

        }

        model.add(new K_Layer.Dense(hidden_sizes[hidden_sizes.length-1], out_size, activation_fn));

        return model;

    }


    // Recurrent

    static ArrayList<K_Layer.LSTM> LSTM(int in_size, int[] hidden_sizes, int out_size) {

        ArrayList<K_Layer.LSTM> model = new ArrayList<>();

        int ctr = -1;
        for (int hidden_size : hidden_sizes) {
            ctr++;

            if (ctr == 0)

                model.add(new K_Layer.LSTM(in_size, hidden_size));

            else

                model.add(new K_Layer.LSTM(hidden_sizes[ctr-1], hidden_size));

        }

        model.add(new K_Layer.LSTM(hidden_sizes[hidden_sizes.length-1], out_size));

        for (K_Layer.LSTM layer : model)

            K_Layer.xavierize(layer);

        return model;

    }

    static K_Tensor[] propogate(ArrayList<K_Layer.LSTM> model, K_Tensor[] incoming) {

        K_Tensor[] response = new K_Tensor[incoming.length];

        int ctr = -1;
        for (K_Tensor timestep : incoming) {
            ctr++;

            for (K_Layer.LSTM layer : model)

                timestep = K_Layer.propogate(layer, timestep);

            response[ctr] = timestep;

        }

        return response;

    }

    static ArrayList<Float[][][]> collect_grads(ArrayList<K_Layer.LSTM> model) {

        ArrayList<Float[][][]> layers_grads = new ArrayList<>();

        for (K_Layer.LSTM layer : model) {

            layers_grads.add(K_Layer.collect_grad(layer));

        }

        return layers_grads;

    }

    static void apply_grads(ArrayList<K_Layer.LSTM> model, ArrayList<Float[][][]> grads) {

        Float[][][] layer_grads;

        int ctr = -1;
        for (K_Layer.LSTM layer : model) {
            ctr++;

            K_Layer.apply_grad(layer, grads.get(ctr));

        }

    }

    static void learn_from_grads(ArrayList<K_Layer.LSTM> model, float learning_rate) {

        for (K_Layer.LSTM layer : model) {

            K_Layer.learn_from_grad(layer, learning_rate);

        }

    }

    static void clear_grads(ArrayList<K_Layer.LSTM> model) {

        for (K_Layer.LSTM layer : model)

            K_Layer.clear_grad(layer);

    }

    static void clear_states(ArrayList<K_Layer.LSTM> model) {

        for (K_Layer.LSTM layer : model)

            K_Layer.clear_state(layer);

    }
    
    
    // Generic FNs


    static K_Tensor propogate(List<Object> model, K_Tensor timestep) {

        for (Object layer : model)

            if (layer instanceof K_Layer.Dense)

                timestep = K_Layer.propogate((K_Layer.Dense) layer, timestep);

            else if (layer instanceof K_Layer.LSTM)

                timestep = K_Layer.propogate((K_Layer.LSTM) layer, timestep);

        return timestep;

    }

    static K_Tensor[] propogate(List<Object> model, K_Tensor[] incoming) {

        K_Tensor[] response = new K_Tensor[incoming.length];

        int ctr = -1;
        for (K_Tensor timestep : incoming) {
            ctr++;

            response[ctr] = propogate(model, timestep);

        }

        return response;

    }

    static ArrayList<Float[][][]> collect_grads(List<Object> model) {

        ArrayList<Float[][][]> layers_grads = new ArrayList<>();

        for (Object layer : model) {

            if (layer instanceof K_Layer.Dense)

                layers_grads.add(new Float[][][]{((K_Layer.Dense) layer).w.grad});

            else if (layer instanceof K_Layer.LSTM)

                layers_grads.add(K_Layer.collect_grad((K_Layer.LSTM) layer));

        }
        
        return layers_grads;

    }

    static void apply_grads(List<Object> model, ArrayList<Float[][][]> grads) {

        Float[][][] layer_grads;

        int ctr = -1;
        for (Object layer : model) {
            ctr++;

            if (layer instanceof K_Layer.Dense) {

                K_Layer.Dense layer_ = (K_Layer.Dense) layer;

                layer_.w.grad = K_Math.add(layer_.w.grad, grads.get(ctr)[0]);

            } else if (layer instanceof K_Layer.LSTM) {

                K_Layer.LSTM layer_ = (K_Layer.LSTM) layer;

                layer_grads = grads.get(ctr);

                K_Layer.apply_grad(layer_, layer_grads);

            }

        }

    }

    static void learn_from_grads(List<Object> model, float learning_rate) {

        for (Object layer : model)

            if (layer instanceof K_Layer.Dense) {

                K_Layer.Dense layer_ = (K_Layer.Dense) layer;

                layer_.w.matrix = K_Math.sub(layer_.w.matrix, K_Math.mul_scalar(learning_rate, layer_.w.grad));

            }

            else if (layer instanceof K_Layer.LSTM) {

                K_Layer.LSTM layer_ = (K_Layer.LSTM) layer;

                K_Layer.learn_from_grad(layer_, learning_rate);
                
            }

    }

    static void clear_grads(List<Object> model) {

        for (Object layer : model)

            if (layer instanceof K_Layer.Dense)

                K_Layer.clear_grad((K_Layer.Dense) layer);

            else if (layer instanceof K_Layer.LSTM)

                K_Layer.clear_grad((K_Layer.LSTM) layer);

    }

    static ArrayList<K_Tensor> collect_states(List<Object> model) {

        ArrayList<K_Tensor> layer_states = new ArrayList<>();

        for (Object layer : model) {

            if (layer instanceof K_Layer.Dense)

                layer_states.add(null);

            else if (layer instanceof K_Layer.LSTM)

                layer_states.add(((K_Layer.LSTM) layer).state);

        }

        return layer_states;

    }

    static void apply_states(List<Object> model, ArrayList<K_Tensor> states) {

        int ctr = -1;
        for (Object layer : model) {
            ctr++;

            if (layer instanceof K_Layer.LSTM)

                ((K_Layer.LSTM) layer).state = states.get(ctr);

        }

    }
    

}


class K_Api{


    static Object[] loss_and_grad_from_datapoint(ArrayList<K_Layer.LSTM> model, ArrayList<Float[][]> datapoint) {

        K_Tensor[] datapoint_t = new K_Tensor[datapoint.size()];

        int ctr = -1;
        for (Float[][] timestep : datapoint) {
            ctr++;

            datapoint_t[ctr] = new K_Tensor(timestep);

        }

        K_Tensor[] response = K_Model.propogate(model, datapoint_t);

        K_Tensor[] datapoint_t_forloss = new K_Tensor[datapoint_t.length-1];
        System.arraycopy(datapoint_t, 1, datapoint_t_forloss, 0, datapoint_t_forloss.length);

        K_Tensor[] response_forloss = new K_Tensor[response.length-1];
        System.arraycopy(response, 0, response_forloss, 0, response_forloss.length);

        float loss = K_Tensor.fill_grads(K_Utils.sequence_loss(datapoint_t_forloss, response_forloss));

        ArrayList<Float[][][]> grads = K_Model.collect_grads(model);

        return new Object[]{loss, grads};

    }

    static Object[] loss_and_grad_from_batch(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> batch) {

        float batch_loss = 0;

        ArrayList<Float[][][]> batch_grad = K_Model.collect_grads(model);

        ArrayList<DatapointTask> tasks = new ArrayList<>();

        for (ArrayList<Float[][]> data : batch)

            tasks.add(new DatapointTask(model, data));

        try {

            List<Future<ArrayList<Float[][]>>> promises = threadpool.invokeAll(tasks);

            for (Future promise : promises)

                while(!promise.isDone()) {System.out.println("Empty promises..");}

            float loss;
            ArrayList<Float[][][]> layer_grads;

            for (int i = 0; i < batch.size(); i++)

                try {

                    loss = (float) tasks.get(i).result[0];
                    layer_grads = (ArrayList<Float[][][]>) tasks.get(i).result[1];

                    batch_loss += loss;

                    int ctr = -1;
                    for (Float[][][] layer_grad : layer_grads) {
                        ctr++;

                        Float[][][] batch_grad_layer = batch_grad.get(ctr);

                        int ctr2 = -1;
                        for (Float[][] weight_grad : layer_grad) {
                            ctr2++;

                            batch_grad_layer[ctr2] = K_Math.add(weight_grad, batch_grad_layer[ctr2]);

                        }

                    }

                }
                catch (Exception e) { e.printStackTrace(); }

        } catch (Exception e) { e.printStackTrace(); return null; }

        return new Object[]{batch_loss, batch_grad};

    }

    static float train_on_batch(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> batch, float learning_rate) {

        Object[] result = loss_and_grad_from_batch(model, batch);

        float batch_loss = (float) result[0];

        ArrayList<Float[][][]> batch_grad = (ArrayList<Float[][][]>) result[1];

        K_Model.apply_grads(model, batch_grad);

        K_Model.learn_from_grads(model, learning_rate/batch.size());

        K_Model.clear_grads(model);

        return batch_loss;

    }

    static float Train_On_Dataset(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, float learning_rate) {

        float ep_loss = 0;

        for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

            ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

        return ep_loss;

    }

    static void Train_On_Dataset(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, float learning_rate, int hm_epochs) {

        float ep_loss;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + " Loss " + ep_loss);

        }

    }

    static void Train_On_Dataset(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset, float train_ratio, float test_ratio, int batch_size, float learning_rate, int hm_epochs, int test_per_epochs) {

        assert train_ratio + test_ratio <= 1;

        ArrayList<ArrayList<ArrayList<Float[][]>>> sets = K_Utils.split_dataset(dataset, train_ratio, test_ratio);
        dataset = sets.get(0); ArrayList<ArrayList<Float[][]>> dataset2 = sets.get(1);

        float ep_loss;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + "\n\tTrain Loss: " + ep_loss + ((i + 1) % test_per_epochs == 0 ? "\n\tTest Loss: " + calc_test_loss(K_Utils.copy(model), dataset2) : ""));

        }

    }

    static float calc_test_loss(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset) {

        return (float) loss_and_grad_from_batch(model, dataset)[0];

    }


    // Generic Modelling


    static List<Object> Generate_Generic_Model(int[] sizes, String[] layer_types, String dense_act_fn) {

        assert layer_types.length == sizes.length-1;

        List<Object> layers = new ArrayList<>();

        int ctr = -1;
        for (String layer_type : layer_types) {
            ctr++;

            switch(layer_type) {

                case "dense":

                    K_Layer.Dense layer_dense = new K_Layer.Dense(sizes[ctr],sizes[ctr+1],dense_act_fn);

                    K_Layer.xavierize(layer_dense);

                    layers.add(layer_dense);

                    break;

                case "lstm":

                    K_Layer.LSTM layer_lstm = new K_Layer.LSTM(sizes[ctr],sizes[ctr+1]);

                    K_Layer.xavierize(layer_lstm);

                    layers.add(layer_lstm);

                    break;

                default:
                    System.out.println("Available layer params: dense/lstm, not " + layer_type);

            }

        }

        return layers;

    } ; static List<Object> Generate_Generic_Model(int[] sizes, String[] layer_types) { return Generate_Generic_Model(sizes, layer_types, "elu"); }


    static Object[] loss_and_grad_from_datapoint(List<Object> model, ArrayList<Float[][]> datapoint) {

        K_Tensor[] datapoint_t = new K_Tensor[datapoint.size()];

        int ctr = -1;
        for (Float[][] timestep : datapoint) {
            ctr++;

            datapoint_t[ctr] = new K_Tensor(timestep);

        }

        K_Tensor[] response = K_Model.propogate(model, datapoint_t);

        K_Tensor[] datapoint_t_forloss = new K_Tensor[datapoint_t.length-1];
        System.arraycopy(datapoint_t, 1, datapoint_t_forloss, 0, datapoint_t_forloss.length);
                                                                     // omg java, just do [:-1]
        K_Tensor[] response_forloss = new K_Tensor[response.length-1];
        System.arraycopy(response, 0, response_forloss, 0, response_forloss.length);

        float loss = K_Tensor.fill_grads(K_Utils.sequence_loss(datapoint_t_forloss, response_forloss));

        ArrayList<Float[][][]> grads = K_Model.collect_grads(model);

        return new Object[]{loss, grads};

    }

    static class DatapointTask implements Callable<ArrayList<Float[][]>> {

        Object[] result;

        ArrayList<K_Layer.LSTM> model;

        List<Object> model_generic;

        ArrayList<Float[][]> datapoint;

        DatapointTask(List<Object> model, ArrayList<Float[][]> datapoint) {

            this.model_generic = K_Utils.copy(model);

            this.datapoint = datapoint;

        }

        DatapointTask(ArrayList<K_Layer.LSTM> model, ArrayList<Float[][]> datapoint) {

            this.model = K_Utils.copy(model);

            this.datapoint = datapoint;

        }

        @Override
        public ArrayList<Float[][]> call() {

            if (this.model != null)

                this.result = loss_and_grad_from_datapoint(this.model, this.datapoint);

            else

                this.result = loss_and_grad_from_datapoint(this.model_generic, this.datapoint);

            return null;

        }

    }

    public static ExecutorService threadpool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    static Object[] loss_and_grad_from_batch(List<Object> model, ArrayList<ArrayList<Float[][]>> batch) {

        float batch_loss = 0;

        ArrayList<Float[][][]> batch_grad = K_Model.collect_grads(model);

        ArrayList<DatapointTask> tasks = new ArrayList<>();

        for (ArrayList<Float[][]> data : batch)

            tasks.add(new DatapointTask(model, data));

        try {

            List<Future<ArrayList<Float[][]>>> promises = threadpool.invokeAll(tasks);

            for (Future promise : promises)

                while(!promise.isDone()) {System.out.println("Empty promises..");}

            float loss;
            ArrayList<Float[][][]> layer_grads;

            for (int i = 0; i < batch.size(); i++)

                try {

                    loss = (float) tasks.get(i).result[0];
                    layer_grads = (ArrayList<Float[][][]>) tasks.get(i).result[1];

                    batch_loss += loss;

                    int ctr = -1;
                    for (Float[][][] layer_grad : layer_grads) {
                        ctr++;

                        Float[][][] batch_grad_layer = batch_grad.get(ctr);

                        int ctr2 = -1;
                        for (Float[][] weight_grad : layer_grad) {
                            ctr2++;

                            batch_grad_layer[ctr2] = K_Math.add(weight_grad, batch_grad_layer[ctr2]);

                        }

                    }

                }
                catch (Exception e) { e.printStackTrace(); }

        } catch (Exception e) { e.printStackTrace(); return null; }

        return new Object[]{batch_loss, batch_grad};

    }


    static float train_on_batch(List<Object> model, ArrayList<ArrayList<Float[][]>> batch, float learning_rate) {

        Object[] result = loss_and_grad_from_batch(model, batch);

        float batch_loss = (float) result[0];

        ArrayList<Float[][][]> batch_grad = (ArrayList<Float[][][]>) result[1];

        K_Model.apply_grads(model, batch_grad);

        K_Model.learn_from_grads(model, learning_rate/batch.size());

        K_Model.clear_grads(model);

        return batch_loss;

    }

    static float Train_On_Dataset(List<Object> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, float learning_rate) {

        float ep_loss = 0;

        for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

            ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

        return ep_loss;

    }

    static void Train_On_Dataset(List<Object> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, float learning_rate, int hm_epochs) {

        float ep_loss;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + " Loss " + ep_loss);

        }

    }

    static void Train_On_Dataset(List<Object> model, ArrayList<ArrayList<Float[][]>> dataset, float train_ratio, float test_ratio, int batch_size, float learning_rate, int hm_epochs, int test_per_epochs) {

        assert train_ratio + test_ratio <= 1;

        ArrayList<ArrayList<ArrayList<Float[][]>>> sets = K_Utils.split_dataset(dataset, train_ratio, test_ratio);
        dataset = sets.get(0); ArrayList<ArrayList<Float[][]>> dataset2 = sets.get(1);

        float ep_loss;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Utils.batchify(K_Utils.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + "\n\tTrain Loss: " + ep_loss + ((i + 1) % test_per_epochs == 0 ? "\n\tTest Loss: " + calc_test_loss(K_Utils.copy(model), dataset2) : ""));

        }

    }

    static float calc_test_loss(List<Object> model, ArrayList<ArrayList<Float[][]>> dataset) {

        return (float) loss_and_grad_from_batch(model, dataset)[0];

    }


}


class K_Utils {


    static ArrayList<ArrayList<Float[][]>> shuffle(ArrayList<ArrayList<Float[][]>> items) {

        Collections.shuffle(items);

        return items;

    }

    static ArrayList<ArrayList<ArrayList<Float[][]>>> batchify(ArrayList<ArrayList<Float[][]>> dataset, int batch_size) {

        ArrayList<ArrayList<ArrayList<Float[][]>>> batches = new ArrayList<>();

        ArrayList<ArrayList<Float[][]>> batch;

        for (int i = 0; i < dataset.size()/batch_size; i++) {

            batch = new ArrayList<>();

            for (int j = 0; j < batch_size; j++)

                batch.add(dataset.get(i*batch_size+j));

            batches.add(batch);

        }

        return batches;

    }

    static ArrayList<ArrayList<ArrayList<ArrayList<Float[][]>>>> batchify2(ArrayList<ArrayList<ArrayList<Float[][]>>> dataset, int batch_size) {

        ArrayList<ArrayList<ArrayList<ArrayList<Float[][]>>>> batches = new ArrayList<>();

        ArrayList<ArrayList<ArrayList<Float[][]>>> batch;

        for (int i = 0; i < dataset.size()/batch_size; i++) {

            batch = new ArrayList<>();

            for (int j = 0; j < batch_size; j++)

                batch.add(dataset.get(i*batch_size+j));

            batches.add(batch);

        }

        return batches;

    }

    static ArrayList<ArrayList<ArrayList<Float[][]>>> split_dataset(ArrayList<ArrayList<Float[][]>> dataset, float train_ratio, float test_ratio) {

        K_Utils.shuffle(dataset);

        ArrayList<ArrayList<ArrayList<Float[][]>>> train_and_test = new ArrayList<>();
        ArrayList<ArrayList<Float[][]>> training_set = new ArrayList<>();
        ArrayList<ArrayList<Float[][]>> testing_set = new ArrayList<>();

        int hm_data = dataset.size();
        int hm_train = (int) Math.floor(hm_data * train_ratio);
        int hm_test = (int) Math.floor(hm_data * test_ratio);

        for (int i = 0; i < hm_train; i++)

            training_set.add(dataset.get(i));

        for (int i = hm_train; i < hm_train+hm_test; i++)

            testing_set.add(dataset.get(i));

        train_and_test.add(training_set);
        train_and_test.add(testing_set);

        return train_and_test;

    }

    static K_Tensor sequence_loss(K_Tensor[] sample, K_Tensor[] response) {

        K_Tensor loss = null;

        for (int t = 0; t < response.length; t++)

            if (t == 0)

                loss = K_Tensor.mean_square(response[t], sample[t]);

            else

                loss = K_Tensor.add(loss, K_Tensor.mean_square(response[t], sample[t]));

        return loss;

    }


    // Extra


    static ArrayList<K_Layer.LSTM> copy(ArrayList<K_Layer.LSTM> model) {

        ArrayList<K_Layer.LSTM> model_copy = new ArrayList<>();

        for (K_Layer.LSTM layer : model)

            model_copy.add(new K_Layer.LSTM(layer));

        return model_copy;

    }

    static List<Object> copy(List<Object> model) {

        List<Object> model_copy = new ArrayList<>();

        for (Object layer : model)

            if (layer instanceof K_Layer.Dense)

                model_copy.add(new K_Layer.Dense((K_Layer.Dense) layer));

            else if (layer instanceof K_Layer.LSTM)

                model_copy.add(new K_Layer.LSTM((K_Layer.LSTM) layer));

        return model_copy;

    }

    static List<Object> combine_models(List<Object> model1, List<Object> model2) {

        List<Object> combination = new ArrayList<>();

        combination.addAll(model1);
        combination.addAll(model2);

        return combination;

    }


}


class K_Dlc {


    static class Encoder_Decoder {


        List<Object> encoder;

        List<Object> decoder;


        Encoder_Decoder(int[] enc_sizes, String[] enc_layer_types, int[] dec_sizes, String[] dec_layer_types, String dense_act_fn) {

            this.encoder = K_Api.Generate_Generic_Model(enc_sizes,enc_layer_types,dense_act_fn);
            this.decoder = K_Api.Generate_Generic_Model(dec_sizes,dec_layer_types,dense_act_fn);

        }

        Encoder_Decoder(int[] encdec_sizes, String[] encdec_layer_types, String dense_act_fn) {

            this.encoder = K_Api.Generate_Generic_Model(encdec_sizes,encdec_layer_types,dense_act_fn);
            this.decoder = K_Api.Generate_Generic_Model(encdec_sizes,encdec_layer_types,dense_act_fn);

        }

        Encoder_Decoder(int[] encdec_sizes, String[] encdec_layer_types) {

            this.encoder = K_Api.Generate_Generic_Model(encdec_sizes,encdec_layer_types);
            this.decoder = K_Api.Generate_Generic_Model(encdec_sizes,encdec_layer_types);

        }

        Encoder_Decoder() { }


    }


    static K_Tensor[] respond_to(Encoder_Decoder model, K_Tensor[] incoming_sequence, int outgoing_length) {

        K_Tensor[] enc_outgoing_response = K_Model.propogate(model.encoder, incoming_sequence);

        ArrayList<K_Tensor> states = K_Model.collect_states(model.encoder);

        try_to_fit_states(model.decoder, states);

        ArrayList<K_Tensor> dec_outgoing_response = new ArrayList<>();

        K_Tensor prev_response_step = null;

        Object dec_last_layer = model.decoder.get(model.decoder.size()-1);

        if (dec_last_layer instanceof K_Layer.Dense)
            prev_response_step = K_Tensor.zeros(1, K_Tensor.size(((K_Layer.Dense) dec_last_layer).w, 1));
        else if (dec_last_layer instanceof K_Layer.LSTM)
            prev_response_step = K_Tensor.zeros(1, K_Tensor.size(((K_Layer.LSTM) dec_last_layer).wf1, 1));

        for (int t = 0; t < outgoing_length; t++) {

            prev_response_step = K_Model.propogate(model.decoder, prev_response_step);

            dec_outgoing_response.add(prev_response_step);

        }

        K_Tensor[] return_array = new K_Tensor[dec_outgoing_response.size()];

        for (int i = 0; i < return_array.length; i++)

            return_array[i] = dec_outgoing_response.get(i);

        return return_array;

    }

    private static void try_to_fit_states(List<Object> decoder, ArrayList<K_Tensor> states) {

        int i = -1;
        for (Object layer : decoder) {
            i++;

            if (layer instanceof K_Layer.LSTM) {

                K_Layer.LSTM layer_ = (K_Layer.LSTM) layer;

                if (i < states.size() && states.get(i) != null && K_Tensor.size(layer_.state, 0) == K_Tensor.size(states.get(i), 0) && K_Tensor.size(layer_.state, 1) == K_Tensor.size(states.get(i), 1))

                    layer_.state = states.get(i);

                else

                    for (K_Tensor state : states)

                        if (state != null && K_Tensor.size(layer_.state, 0) == K_Tensor.size(state, 0) && K_Tensor.size(layer_.state, 1) == K_Tensor.size(state, 1))

                            layer_.state = state;

            }

        }

    }


    static Object[] loss_and_grad_from_datapoint(Encoder_Decoder model, ArrayList<Float[][]> datapoint, ArrayList<Float[][]> label) {

        K_Tensor[] datapoint_t = new K_Tensor[datapoint.size()];

        int ctr = -1;
        for (Float[][] timestep : datapoint) {
            ctr++;

            datapoint_t[ctr] = new K_Tensor(timestep);

        }

        K_Tensor[] response = respond_to(model, datapoint_t, label.size());

        K_Tensor[] label_t = new K_Tensor[label.size()];

        for (int i = 0; i < label_t.length; i++)

            label_t[i] = new K_Tensor(label.get(i));

        float loss = K_Tensor.fill_grads(K_Utils.sequence_loss(response, label_t));

        ArrayList<ArrayList<Float[][][]>> grads = new ArrayList<>();
        grads.add(K_Model.collect_grads(model.encoder));
        grads.add(K_Model.collect_grads(model.decoder));

        return new Object[]{loss, grads};

    }

    static class DatapointTask implements Callable<ArrayList<Float[][]>> {

        Object[] result;

        Encoder_Decoder model;

        ArrayList<ArrayList<Float[][]>> datapoint;

        DatapointTask(Encoder_Decoder model, ArrayList<ArrayList<Float[][]>> datapoint) {

            this.model = new Encoder_Decoder();
            this.model.encoder = K_Utils.copy(model.encoder);
            this.model.decoder = K_Utils.copy(model.decoder);

            this.datapoint = datapoint;

        }

        @Override
        public ArrayList<Float[][]> call() {

            this.result = loss_and_grad_from_datapoint(this.model, this.datapoint.get(0), this.datapoint.get(1));

            return null;

        }

    }

    static Object[] loss_and_grad_from_batch(Encoder_Decoder model, ArrayList<ArrayList<ArrayList<Float[][]>>> batch) {

        float batch_loss = 0;

        ArrayList<Float[][][]> batch_grad_enc = K_Model.collect_grads(model.encoder);
        ArrayList<Float[][][]> batch_grad_dec = K_Model.collect_grads(model.decoder);

        ArrayList<DatapointTask> tasks = new ArrayList<>();

        for (ArrayList<ArrayList<Float[][]>> data : batch)

            tasks.add(new DatapointTask(model, data));

        try {

            List<Future<ArrayList<Float[][]>>> promises = K_Api.threadpool.invokeAll(tasks);

            for (Future promise : promises)

                while(!promise.isDone()) {System.out.println("Empty promises..");}

            float loss;
            ArrayList<ArrayList<Float[][][]>> grads;

            for (int i = 0; i < batch.size(); i++)

                try {

                    loss = (float) tasks.get(i).result[0];
                    grads = (ArrayList<ArrayList<Float[][][]>>) tasks.get(i).result[1];

                    batch_loss += loss;

                    int ctr = -1;
                    for (Float[][][] layer_grad : grads.get(0)) {
                        ctr++;

                        Float[][][] batch_grad_layer = batch_grad_enc.get(ctr);

                        int ctr2 = -1;
                        for (Float[][] weight_grad : layer_grad) {
                            ctr2++;

                            batch_grad_layer[ctr2] = K_Math.add(weight_grad, batch_grad_layer[ctr2]);

                        }

                    }

                    ctr = -1;
                    for (Float[][][] layer_grad : grads.get(1)) {
                        ctr++;

                        Float[][][] batch_grad_layer = batch_grad_dec.get(ctr);

                        int ctr2 = -1;
                        for (Float[][] weight_grad : layer_grad) {
                            ctr2++;

                            batch_grad_layer[ctr2] = K_Math.add(weight_grad, batch_grad_layer[ctr2]);

                        }

                    }

                }

                catch (Exception e) { e.printStackTrace(); }

        } catch (Exception e) { e.printStackTrace(); return null; }

        return new Object[]{batch_loss, new Object[]{batch_grad_enc, batch_grad_dec}};

    }

    static float train_on_batch(Encoder_Decoder model, ArrayList<ArrayList<ArrayList<Float[][]>>> batch, float learning_rate) {

        Object[] result = loss_and_grad_from_batch(model, batch);

        float loss = (float) result[0];

        ArrayList<Float[][][]> batch_grad_enc = (ArrayList<Float[][][]>) ((Object[]) result[1])[0];
        ArrayList<Float[][][]> batch_grad_dec = (ArrayList<Float[][][]>) ((Object[]) result[1])[1];

        K_Model.apply_grads(model.encoder, batch_grad_enc);
        K_Model.learn_from_grads(model.encoder, learning_rate/batch.size());
        K_Model.clear_grads(model.encoder);

        K_Model.apply_grads(model.decoder, batch_grad_dec);
        K_Model.learn_from_grads(model.decoder, learning_rate/batch.size());
        K_Model.clear_grads(model.decoder);

        return loss;

    }


}
