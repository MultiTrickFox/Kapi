package k.kapi;

import java.util.ArrayList;
import java.util.List;


class LayerDense {


    int prev_size;
    int this_size;

    K_Tensor w;

    String act_fn;


    LayerDense(int incoming_size, int outgoing_size, String act_fn) {

        this.prev_size = incoming_size;
        this.this_size = outgoing_size;

        this.w = K_Tensor.randn(incoming_size, outgoing_size);
        this.w.requires_grad = true;
        this.act_fn = act_fn;

    }

    LayerDense(LayerDense layer) {

        this.prev_size = layer.prev_size;
        this.this_size = layer.this_size;

        this.w = new K_Tensor(layer.w.matrix);
        this.w.requires_grad = true;
        this.act_fn = layer.act_fn;

    }

    LayerDense(int incoming_size, int outgoing_size) {

        this.prev_size = incoming_size;
        this.this_size = outgoing_size;

        this.w = K_Tensor.randn(incoming_size, outgoing_size);
        this.w.requires_grad = true;
        this.act_fn = "sigm";

    }


}


class LayerLSTM {


    int prev_size;
    int this_size;

    K_Tensor wf1, wf2;
    K_Tensor wk1, wk2;
    K_Tensor wi1, wi2;
    K_Tensor ws1, ws2;

    K_Tensor state;


    LayerLSTM(int incoming_size, int outgoing_size) {

        this.prev_size = incoming_size;
        this.this_size = outgoing_size;

        this.wf1 = K_Tensor.randn(prev_size, this_size); this.wf1.requires_grad = true;
        this.wf2 = K_Tensor.randn(this_size, this_size); this.wf2.requires_grad = true;

        this.wk1 = K_Tensor.randn(prev_size, this_size); this.wk1.requires_grad = true;
        this.wk2 = K_Tensor.randn(this_size, this_size); this.wk2.requires_grad = true;

        this.wi1 = K_Tensor.randn(prev_size, this_size); this.wi1.requires_grad = true;
        this.wi2 = K_Tensor.randn(this_size, this_size); this.wi2.requires_grad = true;

        this.ws1 = K_Tensor.randn(prev_size, this_size); this.ws1.requires_grad = true;
        this.ws2 = K_Tensor.randn(this_size, this_size); this.ws2.requires_grad = true;

        this.state = K_Tensor.zeros(1, this_size);

    }

    LayerLSTM(LayerLSTM layer) {

        this.prev_size = layer.prev_size;
        this.this_size = layer.this_size;

        this.wf1 = new K_Tensor(layer.wf1.matrix); this.wf1.requires_grad = true;
        this.wf2 = new K_Tensor(layer.wf2.matrix); this.wf2.requires_grad = true;

        this.wk1 = new K_Tensor(layer.wk1.matrix); this.wk1.requires_grad = true;
        this.wk2 = new K_Tensor(layer.wk2.matrix); this.wk2.requires_grad = true;

        this.wi1 = new K_Tensor(layer.wi1.matrix); this.wi1.requires_grad = true;
        this.wi2 = new K_Tensor(layer.wi2.matrix); this.wi2.requires_grad = true;

        this.ws1 = new K_Tensor(layer.ws1.matrix); this.ws1.requires_grad = true;
        this.ws2 = new K_Tensor(layer.ws2.matrix); this.ws2.requires_grad = true;

        this.state = K_Tensor.zeros(1, this_size);

    }


}



class K_Api {


    // FeedForward

    static List<LayerDense> FeedForward(int in_size, int[] hidden_sizes, int out_size, String activation_fn) {

        List<LayerDense> model = new ArrayList<>();

        int ctr = -1;
        for (int hidden_size : hidden_sizes) {
            ctr++;

            if (ctr == 0)

                model.add(new LayerDense(in_size, hidden_size, activation_fn));

            else

                model.add(new LayerDense(hidden_sizes[ctr-1], hidden_size, activation_fn));

        }

        model.add(new LayerDense(hidden_sizes[hidden_sizes.length-1], out_size, activation_fn));

        return model;

    }

    static K_Tensor propogate(List<LayerDense> model, K_Tensor incoming) {

        for (LayerDense layer : model)

            incoming = propogate(layer, incoming);

        return incoming;

    }

    static K_Tensor propogate(LayerDense layer, K_Tensor incoming) {

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


    // Recurrent

    static ArrayList<LayerLSTM> LSTM(int in_size, int[] hidden_sizes, int out_size) {

        ArrayList<LayerLSTM> model = new ArrayList<>();

        int ctr = -1;
        for (int hidden_size : hidden_sizes) {
            ctr++;

            if (ctr == 0)

                model.add(new LayerLSTM(in_size, hidden_size));

            else

                model.add(new LayerLSTM(hidden_sizes[ctr-1], hidden_size));

        }

        model.add(new LayerLSTM(hidden_sizes[hidden_sizes.length-1], out_size));

        return model;

    }

    static ArrayList<K_Tensor> propogate(ArrayList<LayerLSTM> model, K_Tensor[] incoming) {

        ArrayList<K_Tensor> response = new ArrayList<>();

        for (K_Tensor timestep : incoming) {

            for (LayerLSTM layer : model)

                timestep = propogate(layer, timestep);

            response.add(timestep);

        }

        return response;

    }

    static ArrayList<K_Tensor> propogate_seq2seq(ArrayList<LayerLSTM> encoder, ArrayList<LayerLSTM> decoder, K_Tensor[] incoming) {

        ArrayList<K_Tensor> response = new ArrayList<>();

        for (K_Tensor timestep : incoming) {

            for (LayerLSTM layer : encoder)

                timestep = propogate(layer, timestep);

            response.add(timestep); // TODO : do all of this (someday)

        }

        return response;

    }

    static K_Tensor propogate(LayerLSTM layer, K_Tensor incoming) {

        K_Tensor forget = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wf1),K_Tensor.matmul(layer.state , layer.wf2)));
        K_Tensor keep   = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wk1),K_Tensor.matmul(layer.state , layer.wk2)));
        K_Tensor interm = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.wi1),K_Tensor.matmul(layer.state , layer.wi2)));
        K_Tensor show   = K_Tensor.sigm(K_Tensor.add(K_Tensor.matmul(incoming, layer.ws1),K_Tensor.matmul(layer.state , layer.ws2)));

        layer.state = K_Tensor.add(K_Tensor.mul(forget, layer.state), K_Tensor.mul(keep, interm));

        return K_Tensor.mul(show, K_Tensor.tanh(layer.state));

    }


    // Util Fns

    static void xavierize(List<LayerDense> model) {

//        layer.w = ...; // TODO : do

    }

    static void xavierize(ArrayList<LayerLSTM> model) {

//        layer.w = ...; // TODO : do

    }


    // Extras

    static List<LayerDense> copy(List<LayerDense> model) {

        List<LayerDense> model_copy = new ArrayList<>();

        for (LayerDense layer : model)

            model_copy.add(new LayerDense(layer));

        return model_copy;

    }

    static ArrayList<LayerLSTM> copy(ArrayList<LayerLSTM> model) {

        ArrayList<LayerLSTM> model_copy = new ArrayList<>();

        for (LayerLSTM layer : model)

            model_copy.add(new LayerLSTM(layer));

        return model_copy;

    }

    static LayerDense copy(LayerDense layer) {

        return new LayerDense(layer);

    }


}


class K_Trainer{


    static ArrayList<K_Tensor[]> create_fake_data(int in_size, int out_size, int hm_data, int max_length) {

        ArrayList<K_Tensor[]> dataset = new ArrayList<>();

        K_Tensor[] sequence;

        for (int i = 0; i < hm_data; i++) {

            sequence = new K_Tensor[max_length];

            for (int t = 0; t < max_length; t++)

                sequence[t] = K_Tensor.randn(1, in_size);

            dataset.add(sequence);

        }

        return dataset;

    }

    static ArrayList<K_Tensor[][]> batchify(ArrayList<K_Tensor[]> dataset, int batch_size) {

        ArrayList<K_Tensor[][]> batches = new ArrayList<>();

        K_Tensor[][] batch;

        for (int i = 0; i < dataset.size()/batch_size; i++) {

            batch = new K_Tensor[batch_size][2];

            for (int j = 0; j < batch_size; j++)

                batch[j] = dataset.get(i*batch_size+j);

            batches.add(batch);

        }

        return batches;

    }

    static float batch_train(ArrayList<LayerLSTM> model, K_Tensor[][] batch, float learning_rate) { // TODO : Parallelize

        float batch_loss = 0;

        for (K_Tensor[] sample : batch) {

            ArrayList<K_Tensor> response = K_Api.propogate(model, sample); // todo : there will be K_api.copy(model) here

            K_Tensor loss = K_Tensor.zeros(K_Tensor.size(response.get(0)));

            for (int t = 0; t < response.size()-1; t++)

                if (t == 0)

                    loss = K_Tensor.mean_square(response.get(t),sample[t+1]);

                else

                    loss = K_Tensor.add(loss, K_Tensor.mean_square(response.get(t),sample[t+1]));

            batch_loss += K_Tensor.fill_grads(loss); //K_Tensor.fill_grads(K_Tensor.mean_square(response.get(t),sample[t+1]));

            K_Tensor.release_graph();

        }

        learn_from_grads(model, learning_rate/batch.length);

        K_Tensor.empty_grads();

        return batch_loss;

    }

    static ArrayList<Float[][][]> collect_grads(ArrayList<LayerLSTM> model) {

        ArrayList<Float[][][]> layers_grads = new ArrayList<>();

        for (LayerLSTM layer : model) {

            layers_grads.add(new Float[][][]{

                    layer.wf1.grad, layer.wf2.grad,
                    layer.wk1.grad, layer.wk2.grad,
                    layer.wi1.grad, layer.wi2.grad,
                    layer.ws1.grad, layer.ws2.grad,

            });

        }

        return layers_grads;

    }

    static void apply_grads(ArrayList<LayerLSTM> model, ArrayList<Float[][][]> grads) {

        Float[][][] layer_grads;

        int ctr = -1;
        for (LayerLSTM layer : model) {
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

    static void learn_from_grads(ArrayList<LayerLSTM> model, float learning_rate) {

        for (LayerLSTM layer : model) {

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


    static ArrayList<Float[][]> collect_grads(List<LayerDense> model) {

        ArrayList<Float[][]> layers_grads = new ArrayList<>();

        for (LayerDense layer : model)

            layers_grads.add(layer.w.grad);

        return layers_grads;

    }

    static void apply_grads(List<LayerDense> model, ArrayList<Float[][]> grads) {

        Float[][] layer_grads;

        int ctr = -1;
        for (LayerDense layer : model) {
            ctr++;

            layer.w.grad = K_Math.add(layer.w.grad, grads.get(ctr));

        }

    }

    static void learn_from_grads(List<LayerDense> model, float learning_rate) {

        for (LayerDense layer : model)

            layer.w.matrix = K_Math.sub(layer.w.matrix, K_Math.mul_scalar(learning_rate, layer.w.grad));

    }


}
