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

//    static void xavierize(ArrayList<LayerFF> model) {
//
////        layer.w = ...; // TODO : do
//        System.out.println();
//
//    }

//    static void xavierize(ArrayList<LayerLSTM> model) {
//
////        layer.w = ...; // TODO : do
//
//    }


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


    static ArrayList<K_Tensor[]> create_fake_data(int in_size, int out_size, int hm_data) {

        ArrayList<K_Tensor[]> dataset = new ArrayList<>();

        for (int i = 0; i < hm_data; i++)

        { K_Tensor input1 = K_Tensor.randn(1, in_size);
            K_Tensor label1 = K_Tensor.randn(1, out_size);
            dataset.add(new K_Tensor[]{input1, label1}); }

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

    static void batch_train(K_Tensor[][] batch, ArrayList<LayerLSTM> model, float learning_rate) {



    }

    static void batch_train(K_Tensor[][] batch, List<LayerDense> model) {



    }


}
