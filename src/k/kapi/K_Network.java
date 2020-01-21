package k.kapi;

import java.util.ArrayList;


class Neuron {


    K_Tensor weight_to_parents;

    float activation_value;


    Neuron(int hm_parents) {

        this.weight_to_parents = K_Tensor.randn(1,hm_parents);

        this.activation_value = 0.0f;

    }

    Neuron(K_Tensor weight_to_parents) {

        this.weight_to_parents = weight_to_parents;

        this.activation_value = 0.0f;

    }

    void propogate() {



    }

    K_Tensor propogate(K_Tensor from_parents) {

        K_Tensor activation_value = K_Tensor.sigm(K_Tensor.sum(K_Tensor.mul(from_parents, weight_to_parents)));

        return activation_value;

    }

    K_Tensor propogate(K_Tensor from_parents, String activation_type) { // todo: change when more functions added. (i.e. add relu)

        K_Tensor activation_value = activation_type.equals("sigm") ? K_Tensor.sigm(K_Tensor.sum(K_Tensor.mul(from_parents, weight_to_parents))) :
                                                                     K_Tensor.tanh(K_Tensor.sum(K_Tensor.mul(from_parents, weight_to_parents)));

        return activation_value;

    }


}
