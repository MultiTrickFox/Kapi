package k.kapi;

import java.util.ArrayList;


class K_Layer {


    int prev_size;
    int this_size;

    K_Tensor w;

    String act_fn;


    K_Layer(int incoming_size, int outgoing_size, String act_fn) {

        this.prev_size = incoming_size;
        this.this_size = outgoing_size;

        this.w = K_Tensor.randn(incoming_size, outgoing_size);
        this.w.requires_grad = true;
        this.act_fn = act_fn;

    }

    K_Layer(K_Layer layer) {

        this.prev_size = layer.prev_size;
        this.this_size = layer.this_size;

        this.w = new K_Tensor(layer.w.matrix);
        this.w.requires_grad = true;
        this.act_fn = layer.act_fn;

    }

    K_Layer(int incoming_size, int outgoing_size) {

        this.prev_size = incoming_size;
        this.this_size = outgoing_size;

        this.w = K_Tensor.randn(incoming_size, outgoing_size);
        this.w.requires_grad = true;
        this.act_fn = "sigm";

    }


}


class K_NN {


    // FeedForward

    static ArrayList<K_Layer> FeedForward(int in_size, int[] hidden_sizes, int out_size, String activation_fn) {

        ArrayList<K_Layer> model = new ArrayList<>();

        int ctr = -1;
        for (int hidden_size : hidden_sizes) {
            ctr++;

            if (ctr == 0)

                model.add(new K_Layer(in_size, hidden_size, activation_fn));

            else

                model.add(new K_Layer(hidden_sizes[ctr-1], hidden_size, activation_fn));

        }

        model.add(new K_Layer(hidden_sizes[hidden_sizes.length-1], out_size, activation_fn));

        return model;

    }

    static K_Tensor propogate(ArrayList<K_Layer> model, K_Tensor incoming) {

        for (K_Layer layer : model)

            incoming = propogate(layer, incoming);

        return incoming;

    }

    static K_Tensor propogate(K_Layer layer, K_Tensor incoming) {

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


    // Util Fns

    static void xavierize(K_Layer layer) {

//        layer.w = ...; // TODO : do

    }


    // Extras

    static ArrayList<K_Layer> copy(ArrayList<K_Layer> model) {

        ArrayList<K_Layer> model_copy = new ArrayList<>();

        for (K_Layer layer : model)

            model_copy.add(new K_Layer(layer));

        return model_copy;

    }

    static K_Layer copy(K_Layer layer) {

        return new K_Layer(layer);

    }


}
