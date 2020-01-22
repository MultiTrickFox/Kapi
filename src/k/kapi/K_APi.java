package k.kapi;

import java.util.Arrays;
import java.util.Collections;
import java.util.ArrayList;
import java.util.List;


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

        Dense(int incoming_size, int outgoing_size) {

            this.w = K_Tensor.randn(incoming_size, outgoing_size);
            this.w.requires_grad = true;
            this.act_fn = "sigm";

        }


    }

    static K_Tensor propogate(Dense layer, K_Tensor incoming) {

        K_Tensor out = K_Tensor.matmul(incoming, layer.w);

        switch (layer.act_fn) {

            case "sigm":
                return K_Tensor.sigm(out);

            case "tanh":
                return K_Tensor.tanh(out);

            //case "relu": // todo : do someday
            //return K_Tensor.relu(out)

        }

        return null;

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


}


class K_Model {


    // FeedForward

    static List<K_Layer.Dense> FeedForward(int in_size, int[] hidden_sizes, int out_size, String activation_fn) {

        List<K_Layer.Dense> model = new ArrayList<>();

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

    static K_Tensor propogate(List<K_Layer.Dense> model, K_Tensor incoming) {

        for (K_Layer.Dense layer : model)

            incoming = K_Layer.propogate(layer, incoming);

        return incoming;

    }

    static ArrayList<Float[][]> collect_grads(List<K_Layer.Dense> model) {

        ArrayList<Float[][]> layers_grads = new ArrayList<>();

        for (K_Layer.Dense layer : model)

            layers_grads.add(layer.w.grad);

        return layers_grads;

    }

    static void apply_grads(List<K_Layer.Dense> model, ArrayList<Float[][]> grads) {

        Float[][] layer_grads;

        int ctr = -1;
        for (K_Layer.Dense layer : model) {
            ctr++;

            layer.w.grad = K_Math.add(layer.w.grad, grads.get(ctr));

        }

    }

    static void learn_from_grads(List<K_Layer.Dense> model, float learning_rate) {

        for (K_Layer.Dense layer : model)

            layer.w.matrix = K_Math.sub(layer.w.matrix, K_Math.mul_scalar(learning_rate, layer.w.grad));

    }

    static void clear_grads(List<K_Layer.Dense> model) {

        for (K_Layer.Dense layer : model)

            K_Layer.clear_grad(layer);

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

    static ArrayList<K_Tensor> propogate_seq2seq(ArrayList<K_Layer.LSTM> encoder, ArrayList<K_Layer.LSTM> decoder, K_Tensor[] incoming) {

        ArrayList<K_Tensor> response = new ArrayList<>();

        for (K_Tensor timestep : incoming) {

            for (K_Layer.LSTM layer : encoder)

                timestep = K_Layer.propogate(layer, timestep);

            response.add(timestep); // TODO : do all of this (someday)

        }

        return response;

    }

    static ArrayList<Float[][][]> collect_grads(ArrayList<K_Layer.LSTM> model) {

        ArrayList<Float[][][]> layers_grads = new ArrayList<>();

        for (K_Layer.LSTM layer : model) {

            layers_grads.add(new Float[][][]{

                    layer.wf1.grad, layer.wf2.grad,
                    layer.wk1.grad, layer.wk2.grad,
                    layer.wi1.grad, layer.wi2.grad,
                    layer.ws1.grad, layer.ws2.grad,

            });

        }

        return layers_grads;

    }

    static void apply_grads(ArrayList<K_Layer.LSTM> model, ArrayList<Float[][][]> grads) {

        Float[][][] layer_grads;

        int ctr = -1;
        for (K_Layer.LSTM layer : model) {
            ctr++;

            layer_grads = grads.get(ctr);

            layer.wf1.grad = K_Math.add(layer.wf1.grad, layer_grads[0]);
            layer.wf2.grad = K_Math.add(layer.wf2.grad, layer_grads[1]);

            layer.wk1.grad = K_Math.add(layer.wk1.grad, layer_grads[2]);
            layer.wk2.grad = K_Math.add(layer.wk2.grad, layer_grads[3]);

            layer.wi1.grad = K_Math.add(layer.wi1.grad, layer_grads[4]);
            layer.wi2.grad = K_Math.add(layer.wi2.grad, layer_grads[5]);

            layer.ws1.grad = K_Math.add(layer.ws1.grad, layer_grads[6]);
            layer.ws2.grad = K_Math.add(layer.ws2.grad, layer_grads[7]);

        }

    }

    static void learn_from_grads(ArrayList<K_Layer.LSTM> model, float learning_rate) {

        for (K_Layer.LSTM layer : model) {

            layer.wf1.matrix = K_Math.sub(layer.wf1.matrix, K_Math.mul_scalar(learning_rate, layer.wf1.grad));
            layer.wf2.matrix = K_Math.sub(layer.wf2.matrix, K_Math.mul_scalar(learning_rate, layer.wf2.grad));

            layer.wk1.matrix = K_Math.sub(layer.wk1.matrix, K_Math.mul_scalar(learning_rate, layer.wk1.grad));
            layer.wk2.matrix = K_Math.sub(layer.wk2.matrix, K_Math.mul_scalar(learning_rate, layer.wk2.grad));

            layer.wi1.matrix = K_Math.sub(layer.wi1.matrix, K_Math.mul_scalar(learning_rate, layer.wi1.grad));
            layer.wi2.matrix = K_Math.sub(layer.wi2.matrix, K_Math.mul_scalar(learning_rate, layer.wi2.grad));

            layer.ws1.matrix = K_Math.sub(layer.ws1.matrix, K_Math.mul_scalar(learning_rate, layer.ws1.grad));
            layer.ws2.matrix = K_Math.sub(layer.ws2.matrix, K_Math.mul_scalar(learning_rate, layer.ws2.grad));

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

        float loss = K_Tensor.fill_grads(K_Util.sequence_loss(datapoint_t, response));

        ArrayList<Float[][][]> grads = K_Model.collect_grads(model);

        return new Object[]{loss, grads};

    }

    static Object[] loss_and_grad_from_batch(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> batch) {

        float batch_loss = 0;

        ArrayList<Float[][][]> batch_grad = K_Model.collect_grads(model);

        for (ArrayList<Float[][]> sample : batch) {

            Object[] result = loss_and_grad_from_datapoint(K_Util.copy(model), sample);

            // parallelize here.



            batch_loss += (float) result[0]; // TODO : this section comes after parallel

            int ctr = -1;
            for (Float[][][] layer_grad : (ArrayList<Float[][][]>) result[1]) {
                ctr++;

                Float[][][] batch_grad_layer = batch_grad.get(ctr);

                int ctr2 = -1;
                for (Float[][] weight_grad : layer_grad) {
                    ctr2++;

                    batch_grad_layer[ctr] = K_Math.add(weight_grad, batch_grad_layer[ctr2]);

                }

            }

        }

        return new Object[]{batch_loss, batch_grad};

    }


    static float train_on_batch(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> batch, float learning_rate) {

        Object[] result = loss_and_grad_from_batch(model, batch);

        float batch_loss = (float) result[0];

        ArrayList<Float[][][]> batch_grad = (ArrayList<Float[][][]>) result[1];

        K_Model.apply_grads(model, batch_grad);

        K_Model.learn_from_grads(model, learning_rate/batch.size());

        K_Model.clear_grads(model);
        K_Model.clear_states(model);

        return batch_loss;

    }

    static float train_on_dataset(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, float learning_rate) {

        float ep_loss = 0;

        for (ArrayList<ArrayList<Float[][]>> batch : K_Util.batchify(K_Util.shuffle(dataset), batch_size))

            ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

        return ep_loss;

    }

    static void train_on_dataset(ArrayList<K_Layer.LSTM> model, ArrayList<ArrayList<Float[][]>> dataset, int batch_size, int hm_epochs, float learning_rate) {

        float ep_loss;

        for (int i = 0; i < hm_epochs; i++) {

            ep_loss = 0;

            for (ArrayList<ArrayList<Float[][]>> batch : K_Util.batchify(K_Util.shuffle(dataset), batch_size))

                ep_loss += K_Api.train_on_batch(model, batch, learning_rate);

            System.out.println("Epoch " + i + " Loss " + ep_loss);

        }

    }


}


class K_Util {


    static void xavierize(List<K_Layer.Dense> model) {

//        layer.w = ...; // TODO : do

    }

    static void xavierize(ArrayList<K_Layer.LSTM> model) {

//        layer.w = ...; // TODO : do

    }

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

    static K_Tensor sequence_loss(Float[][][] sample, K_Tensor[] response) {

        K_Tensor loss = K_Tensor.zeros(K_Tensor.size(response[0]));

        for (int t = 0; t < response.length-1; t++)

            if (t == 0)

                loss = K_Tensor.mean_square(response[t],sample[t+1]);

            else

                loss = K_Tensor.add(loss, K_Tensor.mean_square(response[t],sample[t+1]));

        return loss;

    }

    static K_Tensor sequence_loss(K_Tensor[] sample, K_Tensor[] response) {

        K_Tensor loss = K_Tensor.zeros(K_Tensor.size(response[0]));

        for (int t = 0; t < response.length-1; t++)

            if (t == 0)

                loss = K_Tensor.mean_square(response[t],sample[t+1]);

            else

                loss = K_Tensor.add(loss, K_Tensor.mean_square(response[t],sample[t+1]));

        return loss;

    }


    // Extra

    static List<K_Layer.Dense> copy(List<K_Layer.Dense> model) {

        List<K_Layer.Dense> model_copy = new ArrayList<>();

        for (K_Layer.Dense layer : model)

            model_copy.add(new K_Layer.Dense(layer));

        return model_copy;

    }

    static ArrayList<K_Layer.LSTM> copy(ArrayList<K_Layer.LSTM> model) {

        ArrayList<K_Layer.LSTM> model_copy = new ArrayList<>();

        for (K_Layer.LSTM layer : model)

            model_copy.add(new K_Layer.LSTM(layer));

        return model_copy;

    }


}
